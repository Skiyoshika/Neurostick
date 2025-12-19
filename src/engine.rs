// src/engine.rs
use crate::drivers::{SignalBatch, SignalBuffer, SpectrumBuilder};
use crate::model::neurogpt::CHANNEL_LABELS_10_20;
use crate::model::neurogpt::NeuroGPTSession;
use crate::model::neurogpt::AdaptiveGate;
use crate::openbci::OpenBciSession;
use crate::recorder::DataRecorder;
use crate::types::*;
use crate::vjoy::VJoyClient;
use std::f64::consts::PI;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

// =========================================================================
// 1. 内嵌 DSP 滤波器 (Biquad 实现) - 解决信号“脏”的问题
// =========================================================================
#[derive(Clone)]
struct Biquad {
    a0: f64, a1: f64, a2: f64,
    b0: f64, b1: f64, b2: f64,
    z1: f64, z2: f64,
}

impl Biquad {
    fn new_notch(fs: f64, freq: f64, q: f64) -> Self {
        let w0 = 2.0 * PI * freq / fs;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        let b0 = 1.0;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self {
            a0, a1, a2, b0, b1, b2, z1: 0.0, z2: 0.0,
        }
    }

    fn new_highpass(fs: f64, freq: f64, q: f64) -> Self {
        let w0 = 2.0 * PI * freq / fs;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self {
            a0, a1, a2, b0, b1, b2, z1: 0.0, z2: 0.0,
        }
    }

    fn process(&mut self, input: f64) -> f64 {
        // Transposed Direct Form II to keep state in z1/z2
        let a1 = self.a1 / self.a0;
        let a2 = self.a2 / self.a0;
        let b0 = self.b0 / self.a0;
        let b1 = self.b1 / self.a0;
        let b2 = self.b2 / self.a0;

        let out = b0 * input + self.z1;
        self.z1 = b1 * input - a1 * out + self.z2;
        self.z2 = b2 * input - a2 * out;
        out
    }
}

// 修正后的 Filter 结构体
struct SimpleFilter {
    // 级联滤波器：先高通，再陷波
    hp: Vec<BiquadState>, // Per channel
    notch: Vec<BiquadState>, // Per channel
    fs: f64,
}

#[derive(Clone)]
struct BiquadState {
    x1: f64, x2: f64, y1: f64, y2: f64,
    b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64,
}

impl BiquadState {
    fn process(&mut self, x: f64) -> f64 {
        let y = (self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2 
                 - self.a1 * self.y1 - self.a2 * self.y2) / self.a0;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }
}

impl SimpleFilter {
    fn new(channels: usize, fs: f64) -> Self {
        let mut hp = Vec::with_capacity(channels);
        let mut notch = Vec::with_capacity(channels);
        
        // 1. 3Hz 高通 (去漂移)
        let hp_coeffs = Self::calc_coeffs(fs, 3.0, 0.707, true);
        // 2. 50Hz 陷波 (去工频干扰 - 国内50Hz，如果是欧美改60Hz)
        let notch_coeffs = Self::calc_coeffs(fs, 50.0, 10.0, false);

        for _ in 0..channels {
            hp.push(hp_coeffs.clone());
            notch.push(notch_coeffs.clone());
        }
        Self { hp, notch, fs }
    }

    fn calc_coeffs(fs: f64, freq: f64, q: f64, is_highpass: bool) -> BiquadState {
        let w0 = 2.0 * PI * freq / fs;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        
        let (b0, b1, b2, a0, a1, a2) = if is_highpass {
            let a0 = 1.0 + alpha;
            (
                (1.0 + cos_w0) / 2.0, -(1.0 + cos_w0), (1.0 + cos_w0) / 2.0,
                a0, -2.0 * cos_w0, 1.0 - alpha
            )
        } else {
            // Notch
            let a0 = 1.0 + alpha;
            (
                1.0, -2.0 * cos_w0, 1.0,
                a0, -2.0 * cos_w0, 1.0 - alpha
            )
        };

        BiquadState { x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0, b0, b1, b2, a0, a1, a2 }
    }

    fn process_sample(&mut self, channel_idx: usize, sample: f64) -> f64 {
        if channel_idx >= self.hp.len() { return sample; }
        let s1 = self.hp[channel_idx].process(sample);
        self.notch[channel_idx].process(s1)
    }
}

// =========================================================================
// 2. 神经意图解码器 (逻辑判定)
// =========================================================================
fn process_neural_intent(
    data: &[f64],
    threshold: f64,
    calib_mode: bool,
    calib_max: &mut f64,
    start_time: Instant,
    calib_target: CalibrationTarget,
    tx: &Sender<BciMessage>,
) -> GamepadState {
    let mut gp = GamepadState::default();

    // 此时进来的 data 已经是滤波后的干净数据了
    let is_active = |idx: usize| -> bool { 
        data.get(idx).map(|&v| v.abs() > threshold).unwrap_or(false) 
    };
    let match_pattern = |indices: &[usize]| -> bool { indices.iter().all(|&i| is_active(i)) };

    // --- 游戏映射逻辑 (保持不变，但现在更准了) ---
    // 左摇杆 (WASD)
    if match_pattern(&[0, 4, 8]) { gp.ly += 1.0; } // W
    if match_pattern(&[1, 5, 9]) { gp.ly -= 1.0; } // S
    if match_pattern(&[2, 6, 10]) { gp.lx -= 1.0; } // A
    if match_pattern(&[3, 7, 11]) { gp.lx += 1.0; } // D

    // 动作键
    if match_pattern(&[0, 1, 2]) { gp.a = true; } 
    if match_pattern(&[3, 4, 5]) { gp.b = true; } 
    if match_pattern(&[6, 7, 8]) { gp.x = true; } 
    if match_pattern(&[9, 10, 11]) { gp.y = true; } 

    // 右摇杆 (IJKL)
    if match_pattern(&[12, 0]) { gp.ry += 1.0; }
    if match_pattern(&[13, 1]) { gp.ry -= 1.0; }
    if match_pattern(&[14, 2]) { gp.rx -= 1.0; }
    if match_pattern(&[15, 3]) { gp.rx += 1.0; }

    // 触发器/肩键
    if match_pattern(&[0, 15]) && gp.ry == 0.0 { gp.lb = true; }
    if match_pattern(&[2, 13]) && gp.rx == 0.0 { gp.rb = true; }
    if match_pattern(&[1, 14]) && gp.rx == 0.0 { gp.lt = true; }
    if match_pattern(&[3, 12]) && gp.ry == 0.0 { gp.rt = true; }

    // 校准逻辑
    if calib_mode {
        let max_s = data.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if max_s > *calib_max {
            *calib_max = max_s;
        }
        if start_time.elapsed().as_secs() >= 3 {
            tx.send(BciMessage::CalibrationResult(calib_target, *calib_max))
                .ok();
        }
    }

    gp
}

pub fn spawn_thread(tx: Sender<BciMessage>, rx_cmd: Receiver<GuiCommand>) {
    thread::spawn(move || {
        let vjd_status_name = |code: i32| -> &'static str {
            match code {
                0 => "VJD_STAT_OWN",
                1 => "VJD_STAT_FREE",
                2 => "VJD_STAT_BUSY",
                3 => "VJD_STAT_MISS",
                4 => "VJD_STAT_UNKN",
                _ => "VJD_STAT_?",
            }
        };

        tx.send(BciMessage::Log("Engine V14.2 (vJoy ownership diagnostics)".to_owned()))
            .ok();

        // --- 初始化 vJoy ---
        let joystick_res = VJoyClient::new(1);
        if let Err(e) = &joystick_res {
            tx.send(BciMessage::Log(format!("vJoy init failed: {e}"))).ok();
        }
        let mut joystick = joystick_res.ok();
        if joystick.is_some() {
            tx.send(BciMessage::VJoyStatus(true)).ok();
            tx.send(BciMessage::Log("✅ vJoy acquired (Device 1)".to_owned())).ok();
        } else {
            tx.send(BciMessage::VJoyStatus(false)).ok();
            tx.send(BciMessage::Log("⚠️ vJoy not found. Gamepad disabled.".to_owned())).ok();
        }

        // If the device isn't actually owned, probe other device IDs (users may enable a different vJoy device).
        let should_probe = joystick
            .as_ref()
            .and_then(|j| j.vjd_status())
            .map(|s| s != 0)
            .unwrap_or(true);
        if should_probe {
            if let Some(j) = &joystick {
                let status = j.vjd_status().unwrap_or(-999);
                let owner = j.owner_pid().unwrap_or(0);
                tx.send(BciMessage::Log(format!(
                    "vJoy not owned: id={}, status={} ({}), owner_pid={}",
                    j.device_id(),
                    status,
                    vjd_status_name(status),
                    owner
                )))
                .ok();
            }
            joystick = None;
            for id in 1..=16u32 {
                if let Ok(client) = VJoyClient::new(id) {
                    tx.send(BciMessage::Log(format!("vJoy acquired (Device {})", id))).ok();
                    joystick = Some(client);
                    break;
                }
            }
        }

        // Cache vJoy capabilities for Steam binding (mapping helper).
        let mut vjoy_buttons: u32 = 0;
        let mut vjoy_has_pov: bool = false;
        let mut vjoy_ls_axis_x: u32 = 0x30; // X
        let mut vjoy_ls_axis_y: u32 = 0x31; // Y
        let mut vjoy_rs_axis_x: u32 = 0x33; // Rx
        let mut vjoy_rs_axis_y: u32 = 0x34; // Ry
        let compute_vjoy_caps = |joy: &VJoyClient| {
            let buttons = joy.button_count().unwrap_or(0);
            let has_pov = joy.cont_pov_count().unwrap_or(0) > 0;

            let ls_candidates = [
                (0x30, 0x31), // X/Y
                (0x33, 0x34), // Rx/Ry
                (0x32, 0x35), // Z/Rz
                (0x35, 0x36), // Rz/Slider
                (0x36, 0x37), // Slider/Dial
            ];
            let mut ls_axis_x: u32 = 0x30;
            let mut ls_axis_y: u32 = 0x31;
            for (ax, ay) in ls_candidates {
                let okx = joy.axis_exists(ax).unwrap_or(false);
                let oky = joy.axis_exists(ay).unwrap_or(false);
                if okx && oky {
                    ls_axis_x = ax;
                    ls_axis_y = ay;
                    break;
                }
            }

            let rs_candidates = [
                (0x33, 0x34), // Rx/Ry
                (0x32, 0x35), // Z/Rz
                (0x35, 0x36), // Rz/Slider
                (0x36, 0x37), // Slider/Dial
            ];
            let mut rs_axis_x: u32 = 0x33;
            let mut rs_axis_y: u32 = 0x34;
            for (ax, ay) in rs_candidates {
                let okx = joy.axis_exists(ax).unwrap_or(false);
                let oky = joy.axis_exists(ay).unwrap_or(false);
                if okx && oky {
                    rs_axis_x = ax;
                    rs_axis_y = ay;
                    break;
                }
            }

            let enabled = joy.vjoy_enabled().unwrap_or(false);
            let status = joy.vjd_status().unwrap_or(-999);
            (
                buttons,
                has_pov,
                ls_axis_x,
                ls_axis_y,
                rs_axis_x,
                rs_axis_y,
                enabled,
                status,
            )
        };
        if let Some(joy) = &joystick {
            let status = joy.vjd_status().unwrap_or(-999);
            let owner = joy.owner_pid().unwrap_or(0);
            let self_pid = std::process::id();
            tx.send(BciMessage::Log(format!(
                "vJoy ownership: id={}, status={} ({}), owner_pid={}",
                joy.device_id(),
                status,
                vjd_status_name(status),
                owner
            )))
            .ok();
            if owner != 0 && owner != self_pid {
                tx.send(BciMessage::Log(format!(
                    "⚠️ vJoy owner_pid ({owner}) != this process ({self_pid}); another process may be holding vJoy."
                )))
                .ok();
            }
            let (buttons, has_pov, lsx, lsy, rsx, rsy, enabled, status) = compute_vjoy_caps(joy);
            vjoy_buttons = buttons;
            vjoy_has_pov = has_pov;
            vjoy_ls_axis_x = lsx;
            vjoy_ls_axis_y = lsy;
            vjoy_rs_axis_x = rsx;
            vjoy_rs_axis_y = rsy;
            tx.send(BciMessage::Log(format!(
                "vJoy: enabled={enabled}, status={} ({})",
                status,
                vjd_status_name(status)
            )))
            .ok();
            tx.send(BciMessage::Log(format!(
                "vJoy caps: buttons={vjoy_buttons}, pov={vjoy_has_pov}, LS axes=0x{vjoy_ls_axis_x:02X}/0x{vjoy_ls_axis_y:02X}, RS axes=0x{vjoy_rs_axis_x:02X}/0x{vjoy_rs_axis_y:02X}"
            )))
            .ok();
        }

        let mut recorder = DataRecorder::new();
        let mut openbci: Option<OpenBciSession> = None;
        let mut signal_buffer: Option<SignalBuffer> = None;
        
        // 默认采样率
        let mut current_sample_rate_hz: f32 = 250.0; 
        
        // --- 初始化 DSP 滤波器 ---
        let mut filters = SimpleFilter::new(16, current_sample_rate_hz as f64);
        let mut neurogpt_gate = AdaptiveGate::new();
        // Lazy-load NeuroGPT so Simulation-mode connect stays responsive (ONNX session creation can take seconds).
        let mut neurogpt: Option<NeuroGPTSession> = None;
        let mut neurogpt_last_error: Option<String> = None;
        let mut last_neurogpt_infer = Instant::now() - Duration::from_secs(10);
        let mut last_neurogpt_success = Option::<Instant>::None;
        let mut last_neurogpt_status_emit = Instant::now() - Duration::from_secs(10);
        let mut last_spectrum_at = Instant::now() - Duration::from_secs(10);
        let mut spectrum_fft_size: usize = 1024;
        let mut neurogpt_calib: Option<(Instant, Instant, f32)> = None; // (start,end,target_per_min)
        let mut neurogpt_calib_margins: Vec<f32> = Vec::new();
        let mut neurogpt_calib_top1: Vec<f32> = Vec::new();
        // GUI receives `BciMessage::NeuroGptTrigger`; we don't need to keep local state here.

        // NeuroGPT is loaded lazily on first stream or self-test so UI remains responsive.

        let mut current_mode = ConnectionMode::Simulation;
        let mut is_active = false;
        let mut is_streaming = false;
        let mut threshold = 150.0; // 默认阈值稍微调低，因为去了直流

        let mut sim_phase: f64 = 0.0;
        let mut current_sim_input = SimInputIntent::default();
        let mut mapping_helper: MappingHelperCommand = MappingHelperCommand::Off;
        let mut mapping_helper_until = Instant::now();
        let mut mapping_helper_step: usize = 0;
        let mut mapping_helper_last_step = Instant::now();
        let mut last_vjoy_error_log = Instant::now() - Duration::from_secs(10);
        let mut calib_mode = false;
        let mut calib_max_val = 0.0;
        let mut calib_start_time = Instant::now();
        let mut calib_target = CalibrationTarget::Relax;

        // 缓存区
        let mut raw_channel_data = vec![0.0f64; 16];
        let mut clean_channel_data = vec![0.0f64; 16];

        // 循环控制
        let mut last_vjoy_update = Instant::now();

        loop {
            // 1. 处理 GUI 命令 (非阻塞)
            while let Ok(cmd) = rx_cmd.try_recv() {
                match cmd {
                    GuiCommand::Connect(mode, port) => {
                        current_mode = mode;
                        if mode == ConnectionMode::Hardware {
                            match OpenBciSession::connect(&port) {
                                Ok(session) => {
                                    current_sample_rate_hz = session.sample_rate_hz();
                                    // 重置滤波器以匹配新采样率
                                    filters = SimpleFilter::new(16, current_sample_rate_hz as f64);
                                    openbci = Some(session);
                                    is_active = true;
                                    tx.send(BciMessage::Status(true)).ok();
                                    // Log how many EEG channels BrainFlow reports (should be 16 for Cyton+Daisy).
                                    let n = openbci
                                        .as_ref()
                                        .map(|s| s.eeg_channel_count())
                                        .unwrap_or(0);
                                    tx.send(BciMessage::Log(format!(
                                        "✅ OpenBCI Connected ({} Hz, eeg_ch={})",
                                        current_sample_rate_hz, n
                                    )))
                                    .ok();
                                    if n > 0 && n < 16 {
                                        tx.send(BciMessage::Log(
                                            "⚠️ BrainFlow reports <16 EEG channels. Daisy may not be detected or the link is unstable; check the Daisy connection, dongle distance, and USB interference."
                                                .to_owned(),
                                        ))
                                        .ok();
                                    }
                                }
                                Err(e) => { tx.send(BciMessage::Log(format!("❌ Failed: {}", e))).ok(); }
                            }
                        } else {
                            is_active = true;
                            tx.send(BciMessage::Status(true)).ok();
                            tx.send(BciMessage::Log("✅ Simulation Mode".to_owned())).ok();
                        }
                    }
                    GuiCommand::Disconnect => {
                        is_active = false; is_streaming = false;
                        if let Some(mut s) = openbci.take() {
                            let _ = s.stop_stream();
                            let _ = s.release();
                        }
                        tx.send(BciMessage::Status(false)).ok();
                    }
                    GuiCommand::StartStream => { if is_active { 
                        is_streaming = true; 
                        if let Some(s) = openbci.as_mut() {
                            if let Err(e) = s.start_stream() {
                                tx.send(BciMessage::Log(format!("❌ start_stream failed: {e}"))).ok();
                            }
                        }
                        tx.send(BciMessage::Log("🌊 Stream Started".to_owned())).ok();
                        // Do not load NeuroGPT here: ONNX session creation can block the engine loop and stop UI updates.
                        // The model will be loaded on-demand via the "NeuroGPT 自检" button (or when needed later).
                        tx.send(BciMessage::NeuroGptStatus(NeuroGptRuntimeStatus {
                            onnx_loaded: neurogpt.is_some(),
                            onnx_path: Some("model/neurogpt.onnx".to_owned()),
                            last_error: neurogpt_last_error.clone(),
                            last_infer_ms_ago: last_neurogpt_success
                                .map(|t| t.elapsed().as_millis() as u64),
                            gate: neurogpt_gate.params(),
                        }))
                        .ok();
                    }}
                    GuiCommand::StopStream => { 
                        is_streaming = false; 
                        if let Some(s) = openbci.as_mut() {
                            if let Err(e) = s.stop_stream() {
                                tx.send(BciMessage::Log(format!("❌ stop_stream failed: {e}"))).ok();
                            }
                        }
                        tx.send(BciMessage::Log("🛑 Stream Stopped".to_owned())).ok();
                    }
                    GuiCommand::SetThreshold(v) => threshold = v,
                    GuiCommand::SetFftSize(sz) => {
                        spectrum_fft_size = sz.clamp(32, 8192);
                    }
                    GuiCommand::StartCalibration(is_action) => {
                        calib_mode = true;
                        calib_max_val = 0.0;
                        calib_start_time = Instant::now();
                        calib_target = if is_action {
                            CalibrationTarget::Action
                        } else {
                            CalibrationTarget::Relax
                        };
                    }
                    GuiCommand::UpdateSimInput(input) => current_sim_input = input,
                    GuiCommand::StartRecording(l) => { recorder.start(&l); tx.send(BciMessage::RecordingStatus(true)).ok(); }
                    GuiCommand::StopRecording => { recorder.stop(); tx.send(BciMessage::RecordingStatus(false)).ok(); }
                    GuiCommand::InjectArtifact => { /* handled elsewhere / optional */ }
                    GuiCommand::SetMappingHelper(cmd) => {
                        mapping_helper = cmd;
                        // Steam's binding UI can miss very short pulses; keep a longer stable window.
                        mapping_helper_until = Instant::now() + Duration::from_millis(1200);
                        mapping_helper_step = 0;
                        mapping_helper_last_step = Instant::now();

                        // (Re)acquire vJoy when the user wants mapping helper.
                        if mapping_helper != MappingHelperCommand::Off {
                            if joystick.is_none() {
                                let joystick_res = VJoyClient::new(1);
                                if let Err(e) = &joystick_res {
                                    tx.send(BciMessage::Log(format!("vJoy init failed: {e}"))).ok();
                                }
                                joystick = joystick_res.ok();
                            }
                            if joystick.is_some() {
                                tx.send(BciMessage::VJoyStatus(true)).ok();
                                tx.send(BciMessage::Log("✅ vJoy ready".to_owned())).ok();
                                if let Some(joy) = &joystick {
                                    let (buttons, has_pov, lsx, lsy, rsx, rsy, enabled, status) =
                                        compute_vjoy_caps(joy);
                                    vjoy_buttons = buttons;
                                    vjoy_has_pov = has_pov;
                                    vjoy_ls_axis_x = lsx;
                                    vjoy_ls_axis_y = lsy;
                                    vjoy_rs_axis_x = rsx;
                                    vjoy_rs_axis_y = rsy;
                                    tx.send(BciMessage::Log(format!(
                                        "vJoy: enabled={enabled}, status={} ({})",
                                        status,
                                        vjd_status_name(status)
                                    )))
                                    .ok();
                                    tx.send(BciMessage::Log(format!(
                                        "vJoy caps: buttons={vjoy_buttons}, pov={vjoy_has_pov}, LS axes=0x{vjoy_ls_axis_x:02X}/0x{vjoy_ls_axis_y:02X}, RS axes=0x{vjoy_rs_axis_x:02X}/0x{vjoy_rs_axis_y:02X}"
                                    )))
                                    .ok();
                                }
                            } else {
                                tx.send(BciMessage::VJoyStatus(false)).ok();
                                tx.send(BciMessage::Log(
                                    "⚠️ vJoy unavailable (Device 1). If joy.cpl shows no movement: verify vJoyConf Device 1 is enabled and no other app is holding vJoy."
                                        .to_owned(),
                                ))
                                .ok();
                            }
                        }
                    }
                    GuiCommand::SetNeuroGptGate(p) => {
                        neurogpt_gate.set_params(p);
                        tx.send(BciMessage::NeuroGptStatus(NeuroGptRuntimeStatus {
                            onnx_loaded: neurogpt.is_some(),
                            onnx_path: Some("model/neurogpt.onnx".to_owned()),
                            last_error: neurogpt_last_error.clone(),
                            last_infer_ms_ago: last_neurogpt_success
                                .map(|t| t.elapsed().as_millis() as u64),
                            gate: neurogpt_gate.params(),
                        }))
                        .ok();
                    }
                    GuiCommand::NeuroGptSelfTest => {
                        if neurogpt.is_none() && neurogpt_last_error.is_none() {
                            match NeuroGPTSession::new() {
                                Ok(s) => {
                                    tx.send(BciMessage::Log(
                                        "✅ NeuroGPT ONNX session loaded (expects 250 timesteps; supports 250Hz/125Hz)"
                                            .to_owned(),
                                    ))
                                    .ok();
                                    neurogpt = Some(s);
                                }
                                Err(e) => {
                                    let msg = e.to_string();
                                    neurogpt_last_error = Some(msg.clone());
                                    tx.send(BciMessage::Log(format!(
                                        "ℹ️ NeuroGPT ONNX disabled (load failed): {msg}"
                                    )))
                                    .ok();
                                }
                            }
                            tx.send(BciMessage::NeuroGptStatus(NeuroGptRuntimeStatus {
                                onnx_loaded: neurogpt.is_some(),
                                onnx_path: Some("model/neurogpt.onnx".to_owned()),
                                last_error: neurogpt_last_error.clone(),
                                last_infer_ms_ago: last_neurogpt_success
                                    .map(|t| t.elapsed().as_millis() as u64),
                                gate: neurogpt_gate.params(),
                            }))
                            .ok();
                        }
                        if let Some(sess) = neurogpt.as_mut() {
                            let frame = crate::drivers::TimeSeriesFrame {
                                sample_rate_hz: 250.0,
                                channel_labels: (0..16).map(|i| format!("Ch{}", i + 1)).collect(),
                                samples: (0..16)
                                    .map(|ch| {
                                        let f = 8.0 + (ch as f32) * 0.5;
                                        (0..250)
                                            .map(|i| {
                                                let t = i as f32 / 250.0;
                                                (2.0 * std::f32::consts::PI * f * t).sin() * 10.0
                                            })
                                            .collect::<Vec<f32>>()
                                    })
                                    .collect(),
                            };
                            match sess.predict_command(&frame) {
                                Ok((idx, probs, _cmd)) => {
                                    tx.send(BciMessage::Log(format!(
                                        "NeuroGPT self-test OK: argmax={idx}, probs={:?}",
                                        probs
                                    )))
                                    .ok();
                                    last_neurogpt_success = Some(Instant::now());
                                    tx.send(BciMessage::NeuroGptStatus(NeuroGptRuntimeStatus {
                                        onnx_loaded: true,
                                        onnx_path: Some("model/neurogpt.onnx".to_owned()),
                                        last_error: None,
                                        last_infer_ms_ago: Some(0),
                                        gate: neurogpt_gate.params(),
                                    }))
                                    .ok();
                                }
                                Err(e) => {
                                    tx.send(BciMessage::Log(format!(
                                        "NeuroGPT self-test failed: {e}"
                                    )))
                                    .ok();
                                }
                            }
                        } else {
                            tx.send(BciMessage::Log(
                                "NeuroGPT self-test skipped: ONNX session not loaded".to_owned(),
                            ))
                            .ok();
                        }
                    }
                    GuiCommand::NeuroGptCalibrateStart {
                        seconds,
                        target_triggers_per_min,
                    } => {
                        let secs = seconds.max(3).min(60);
                        let now = Instant::now();
                        neurogpt_calib = Some((now, now + Duration::from_secs(secs as u64), target_triggers_per_min));
                        neurogpt_calib_margins.clear();
                        neurogpt_calib_top1.clear();
                        tx.send(BciMessage::Log(format!(
                            "NeuroGPT calibration started: {}s, target={:.1}/min",
                            secs, target_triggers_per_min
                        )))
                        .ok();
                    }
                }
            }


            // Steam mapping helper: drive vJoy directly (no focus / no streaming dependency)
            if mapping_helper != MappingHelperCommand::Off {
                let now = Instant::now();
                let mut gp = GamepadState::default();

                if mapping_helper == MappingHelperCommand::AutoCycle {
                    if mapping_helper_last_step.elapsed() >= Duration::from_millis(900) {
                        // Steam controller binding (typical order):
                        // A,B,X,Y, Dpad L/R/U/D, LS L/R/U/D, LS click, RS L/R/U/D, RS click.
                        mapping_helper_step = (mapping_helper_step + 1) % 18;
                        mapping_helper_last_step = now;
                    }
                    match mapping_helper_step {
                        0 => gp.a = true,
                        1 => gp.b = true,
                        2 => gp.x = true,
                        3 => gp.y = true,
                        4 => gp.dpad_left = true,
                        5 => gp.dpad_right = true,
                        6 => gp.dpad_up = true,
                        7 => gp.dpad_down = true,
                        8 => gp.lx = -1.0,
                        9 => gp.lx = 1.0,
                        // Many gamepad APIs treat "up" as negative Y.
                        10 => gp.ly = -1.0,
                        11 => gp.ly = 1.0,
                        12 => gp.ls = true,
                        13 => gp.rx = -1.0,
                        14 => gp.rx = 1.0,
                        15 => gp.ry = -1.0,
                        16 => gp.ry = 1.0,
                        _ => gp.rs = true,
                    }
                } else if now <= mapping_helper_until {
                    match mapping_helper {
                        MappingHelperCommand::PulseA => gp.a = true,
                        MappingHelperCommand::PulseB => gp.b = true,
                        MappingHelperCommand::PulseX => gp.x = true,
                        MappingHelperCommand::PulseY => gp.y = true,
                        MappingHelperCommand::PulseLB => gp.lb = true,
                        MappingHelperCommand::PulseRB => gp.rb = true,
                        MappingHelperCommand::PulseLT => gp.lt = true,
                        MappingHelperCommand::PulseRT => gp.rt = true,
                        MappingHelperCommand::PulseBack => gp.back = true,
                        MappingHelperCommand::PulseStart => gp.start = true,
                        MappingHelperCommand::PulseLeftStickClick => gp.ls = true,
                        MappingHelperCommand::PulseRightStickClick => gp.rs = true,
                        MappingHelperCommand::PulseDpadUp => gp.dpad_up = true,
                        MappingHelperCommand::PulseDpadDown => gp.dpad_down = true,
                        MappingHelperCommand::PulseDpadLeft => gp.dpad_left = true,
                        MappingHelperCommand::PulseDpadRight => gp.dpad_right = true,
                        MappingHelperCommand::PulseLeftStickUp => gp.ly = -1.0,
                        MappingHelperCommand::PulseLeftStickDown => gp.ly = 1.0,
                        MappingHelperCommand::PulseLeftStickLeft => gp.lx = -1.0,
                        MappingHelperCommand::PulseLeftStickRight => gp.lx = 1.0,
                        MappingHelperCommand::PulseRightStickUp => gp.ry = -1.0,
                        MappingHelperCommand::PulseRightStickDown => gp.ry = 1.0,
                        MappingHelperCommand::PulseRightStickLeft => gp.rx = -1.0,
                        MappingHelperCommand::PulseRightStickRight => gp.rx = 1.0,
                        MappingHelperCommand::AutoCycle | MappingHelperCommand::Off => {}
                    }
                }

                if let Some(joy) = &mut joystick {
                    let mut ok_all = true;
                    let safe_btn = |joy: &VJoyClient, max: u32, id: u8, down: bool| -> bool {
                        if max == 0 || (id as u32) <= max {
                            joy.set_button(id, down)
                        } else {
                            true
                        }
                    };

                    ok_all &= joy.set_button(1, gp.a);
                    ok_all &= joy.set_button(2, gp.b);
                    ok_all &= joy.set_button(3, gp.x);
                    ok_all &= joy.set_button(4, gp.y);
                    ok_all &= safe_btn(joy, vjoy_buttons, 5, gp.lb);
                    ok_all &= safe_btn(joy, vjoy_buttons, 6, gp.rb);
                    ok_all &= safe_btn(joy, vjoy_buttons, 7, gp.lt);
                    ok_all &= safe_btn(joy, vjoy_buttons, 8, gp.rt);

                    if vjoy_has_pov {
                        // Prefer POV hat for D-pad so Steam recognizes it reliably.
                        let pov = if gp.dpad_up {
                            0
                        } else if gp.dpad_right {
                            9000
                        } else if gp.dpad_down {
                            18000
                        } else if gp.dpad_left {
                            27000
                        } else {
                            -1
                        };
                        ok_all &= joy.set_cont_pov(1, pov);
                        ok_all &= safe_btn(joy, vjoy_buttons, 9, false);
                        ok_all &= safe_btn(joy, vjoy_buttons, 10, false);
                        ok_all &= safe_btn(joy, vjoy_buttons, 11, false);
                        ok_all &= safe_btn(joy, vjoy_buttons, 12, false);
                    } else {
                        // Fallback: use buttons if POV hat isn't enabled in vJoyConf.
                        ok_all &= safe_btn(joy, vjoy_buttons, 9, gp.dpad_up);
                        ok_all &= safe_btn(joy, vjoy_buttons, 10, gp.dpad_down);
                        ok_all &= safe_btn(joy, vjoy_buttons, 11, gp.dpad_left);
                        ok_all &= safe_btn(joy, vjoy_buttons, 12, gp.dpad_right);
                    }

                    ok_all &= safe_btn(joy, vjoy_buttons, 13, gp.back);
                    ok_all &= safe_btn(joy, vjoy_buttons, 14, gp.start);
                    ok_all &= safe_btn(joy, vjoy_buttons, 15, gp.ls);
                    ok_all &= safe_btn(joy, vjoy_buttons, 16, gp.rs);
                    let axis = |v: f32| -> i32 {
                        // Steam's binding UI can require near-extreme motion; use full vJoy range.
                        let v = v.clamp(-1.0, 1.0) as f64;
                        let min = 0.0;
                        let max = 32767.0;
                        let t = (v + 1.0) * 0.5; // [-1,1] -> [0,1]
                        (min + t * (max - min)) as i32
                    };
                    ok_all &= joy.set_axis(vjoy_ls_axis_x, axis(gp.lx));
                    ok_all &= joy.set_axis(vjoy_ls_axis_y, axis(gp.ly));
                    ok_all &= joy.set_axis(vjoy_rs_axis_x, axis(gp.rx));
                    ok_all &= joy.set_axis(vjoy_rs_axis_y, axis(gp.ry));

                    if !ok_all && last_vjoy_error_log.elapsed() >= Duration::from_secs(1) {
                        let enabled = joy.vjoy_enabled().unwrap_or(false);
                        let status = joy.vjd_status().unwrap_or(-999);
                        tx.send(BciMessage::Log(format!(
                            "vJoy write failed: enabled={enabled}, status={} ({})",
                            status,
                            vjd_status_name(status)
                        )))
                        .ok();
                        last_vjoy_error_log = Instant::now();
                    }
                }

                if last_vjoy_update.elapsed().as_millis() > 30 {
                    tx.send(BciMessage::GamepadUpdate(gp)).ok();
                    last_vjoy_update = Instant::now();
                }

                // Keep a light tick so Steam sees changes even if streaming is stopped.
                if !is_streaming {
                    thread::sleep(Duration::from_millis(16));
                }
            }

            // 2. 数据采集与处理
            if is_streaming {
                let mut has_new_data = false;

                if current_mode == ConnectionMode::Simulation {
                    // 模拟数据生成
                    sim_phase += 0.1;
                    let noise = (sim_phase * 0.5).sin() * 5.0; // 模拟一些底噪
                    
                    raw_channel_data.fill(0.0);
                    // ... (此处省略太长的模拟输入判定，保持原样即可，重点是后面)
                    // 为了演示简单，这里只保留一部分模拟逻辑
                    // Steam mapping helper (works even when Steam window is focused).
                    // SIM keyboard shortcuts require QNMDsol focus; this helper generates vJoy inputs in the background.
                    let mut sim = current_sim_input;
                    if mapping_helper == MappingHelperCommand::AutoCycle {
                        if mapping_helper_last_step.elapsed() >= Duration::from_millis(900) {
                            mapping_helper_step = (mapping_helper_step + 1) % 8;
                            mapping_helper_last_step = Instant::now();
                        }
                        sim = SimInputIntent::default();
                        match mapping_helper_step {
                            0 => sim.space = true, // A
                            1 => sim.key_z = true, // B
                            2 => sim.key_x = true, // X
                            3 => sim.key_c = true, // Y
                            4 => sim.w = true,     // LS up
                            5 => sim.s = true,     // LS down
                            6 => sim.a = true,     // LS left
                            _ => sim.d = true,     // LS right
                        }
                    } else if mapping_helper != MappingHelperCommand::Off
                        && Instant::now() <= mapping_helper_until
                    {
                        sim = SimInputIntent::default();
                        match mapping_helper {
                            MappingHelperCommand::PulseA => sim.space = true,
                            MappingHelperCommand::PulseB => sim.key_z = true,
                            MappingHelperCommand::PulseX => sim.key_x = true,
                            MappingHelperCommand::PulseY => sim.key_c = true,
                            MappingHelperCommand::PulseLeftStickUp => sim.w = true,
                            MappingHelperCommand::PulseLeftStickDown => sim.s = true,
                            MappingHelperCommand::PulseLeftStickLeft => sim.a = true,
                            MappingHelperCommand::PulseLeftStickRight => sim.d = true,
                            _ => {}
                        }
                    }

                    // Simulation input -> channel activation patterns expected by process_neural_intent.
                    let mut bump = |idx: usize| {
                        if let Some(v) = raw_channel_data.get_mut(idx) {
                            *v += 500.0;
                        }
                    };
                    if sim.w { for &i in &[0, 4, 8] { bump(i); } }
                    if sim.s { for &i in &[1, 5, 9] { bump(i); } }
                    if sim.a { for &i in &[2, 6, 10] { bump(i); } }
                    if sim.d { for &i in &[3, 7, 11] { bump(i); } }
                    if sim.space { for &i in &[0, 1, 2] { bump(i); } } // A
                    if sim.key_z { for &i in &[3, 4, 5] { bump(i); } } // B
                    if sim.key_x { for &i in &[6, 7, 8] { bump(i); } } // X
                    if sim.key_c { for &i in &[9, 10, 11] { bump(i); } } // Y
                    
                    // 模拟模式也加上一点随机漂移，测试滤波器
                    for v in raw_channel_data.iter_mut() { *v += noise; }
                    
                    has_new_data = true;
                    thread::sleep(Duration::from_millis(4)); // 250Hz approx
                } else if let Some(session) = openbci.as_mut() {
                    match session.next_sample() {
                        Ok(Some(sample)) => {
                            for (i, v) in sample.iter().take(16).enumerate() {
                                raw_channel_data[i] = *v;
                            }
                            has_new_data = true;
                        }
                        Ok(None) => {
                            // 没有数据时短暂休眠，避免死循环烧CPU
                            // 关键优化：休眠时间要极短
                            thread::sleep(Duration::from_micros(500)); 
                        }
                        Err(_) => { thread::sleep(Duration::from_millis(10)); }
                    }
                }

                if has_new_data {
                    // === 关键步骤：实时滤波 ===
                    // OpenBCI 的原始数据可能有几万的直流偏置，必须滤掉
                    for i in 0..16 {
                        let filtered = filters.process_sample(i, raw_channel_data[i]);
                        // BrainFlow 返回的 Cyton 数据是伏特级别，UI/阈值逻辑使用微伏，统一缩放
                        clean_channel_data[i] = if current_mode == ConnectionMode::Hardware {
                            filtered * 1e6
                        } else {
                            filtered
                        };
                    }

                    // 录制原始数据(Raw)还是干净数据(Clean)? 
                    // 建议录制 Raw，方便以后调整算法。但为了演示效果，这里我们把 Clean 发给 UI
                    if recorder.is_recording() {
                        recorder.write_record(&raw_channel_data);
                    }

                    // === 发送数据给 UI 渲染 ===
                    // 初始化 Buffer (如果为空)
                    if signal_buffer.is_none() {
                        let labels: Vec<String> = if clean_channel_data.len() == 16 {
                            CHANNEL_LABELS_10_20.iter().map(|s| s.to_string()).collect()
                        } else {
                            (0..clean_channel_data.len())
                                .map(|i| format!("Ch{}", i + 1))
                                .collect()
                        };
                        signal_buffer = SignalBuffer::with_history_seconds(labels, current_sample_rate_hz, 10.0).ok();
                    }

                    if let Some(buf) = signal_buffer.as_mut() {
                        // 把 clean_channel_data 包装成 Batch
                        let batch = SignalBatch {
                            started_at: SystemTime::now(),
                            sample_rate_hz: current_sample_rate_hz,
                            channel_labels: buf.channel_labels().to_vec(),
                            samples: clean_channel_data.iter().map(|&v| vec![v as f32]).collect(),
                        };
                        buf.push_batch(&batch).ok();
                        
                        // 降低 UI 刷新频率，比如每 4 个采样发一次 GUI，或者只发最新的 snapshot
                        // 为了流畅度，这里每次都发，但 GUI 端要注意性能
                        let frame = buf.snapshot(5.0);
                        tx.send(BciMessage::DataFrame(frame.clone())).ok();

                        // Send spectrum on a lower cadence to keep FFT cost bounded.
                        if last_spectrum_at.elapsed() >= Duration::from_millis(250) {
                            last_spectrum_at = Instant::now();
                            let builder = SpectrumBuilder::with_size(spectrum_fft_size);
                            tx.send(BciMessage::Spectrum(builder.compute(&frame))).ok();
                        }
                    }

                    // === 神经解码 (使用干净数据) ===
                    let mut gp = process_neural_intent(
                        &clean_channel_data, 
                        threshold, 
                        calib_mode, 
                        &mut calib_max_val, 
                        calib_start_time, 
                        calib_target,
                        &tx
                    );

                    // === NeuroGPT (ONNX) 推理：8-30Hz 带通 + 125Hz->250Hz 插值 ===
                    if let (Some(sess), Some(buf)) = (neurogpt.as_mut(), signal_buffer.as_ref()) {
                        if last_neurogpt_infer.elapsed() >= Duration::from_millis(200) {
                            last_neurogpt_infer = Instant::now();
                            let one_sec = buf.snapshot(1.0);
                            if let Ok((idx, probs, cmd)) = sess.predict_command(&one_sec) {
                                tx.send(BciMessage::ModelPrediction(probs.clone())).ok();
                                last_neurogpt_success = Some(Instant::now());

                                // Calibration collection (margin + top1).
                                if let Some((start, end, _target)) = neurogpt_calib {
                                    let now = Instant::now();
                                    let progress01 = ((now.duration_since(start).as_secs_f32()
                                        / end.duration_since(start).as_secs_f32())
                                        .clamp(0.0, 1.0))
                                    .min(1.0);
                                    tx.send(BciMessage::NeuroGptCalibrationProgress { progress01 })
                                        .ok();
                                    if let Some((top1, top2)) = crate::model::neurogpt::top2_probs(&probs) {
                                        neurogpt_calib_top1.push(top1);
                                        neurogpt_calib_margins.push((top1 - top2).max(0.0));
                                    }
                                }

                                if let Some(cmd) = neurogpt_gate.decide(&probs, cmd) {
                                    tx.send(BciMessage::NeuroGptTrigger(idx)).ok();
                                    match cmd {
                                        MappingHelperCommand::PulseLeftStickLeft => gp.lx = -1.0,
                                        MappingHelperCommand::PulseLeftStickRight => gp.lx = 1.0,
                                        MappingHelperCommand::PulseLeftStickUp => gp.ly = -1.0, // Forward
                                        _ => {}
                                    }
                                }
                            }
                        }

                        // Finalize calibration if time elapsed.
                        if let Some((start, end, target_per_min)) = neurogpt_calib {
                            if Instant::now() >= end {
                                neurogpt_calib = None;
                                let n = neurogpt_calib_margins.len().max(1) as f32;
                                let mean = neurogpt_calib_margins.iter().copied().sum::<f32>() / n;
                                let var = neurogpt_calib_margins
                                    .iter()
                                    .copied()
                                    .map(|x| {
                                        let d = x - mean;
                                        d * d
                                    })
                                    .sum::<f32>()
                                    / n;

                                // Estimate desired exceed probability based on inference rate (~5Hz) and target rate.
                                let infer_hz = 5.0;
                                let expected_infers = (infer_hz * end.duration_since(start).as_secs_f32())
                                    .max(1.0);
                                let target_total = (target_per_min.max(0.0) / 60.0) * end.duration_since(start).as_secs_f32();
                                let exceed_p = (target_total / expected_infers).clamp(0.001, 0.5);
                                let quantile_p = (1.0 - exceed_p).clamp(0.5, 0.999);

                                neurogpt_calib_margins.sort_by(|a, b| a.total_cmp(b));
                                let q_idx = ((quantile_p * (neurogpt_calib_margins.len() - 1).max(1) as f32)
                                    .round() as usize)
                                    .min(neurogpt_calib_margins.len().saturating_sub(1));
                                let q = neurogpt_calib_margins.get(q_idx).copied().unwrap_or(mean);

                                let std = var.max(1e-6).sqrt();
                                let mut p = neurogpt_gate.params();
                                p.k_sigma = ((q - mean) / std).clamp(0.5, 5.0);
                                // Set an absolute floor on probability based on observed distribution.
                                if !neurogpt_calib_top1.is_empty() {
                                    neurogpt_calib_top1.sort_by(|a, b| a.total_cmp(b));
                                    let p_idx = ((0.8 * (neurogpt_calib_top1.len() - 1).max(1) as f32).round()
                                        as usize)
                                        .min(neurogpt_calib_top1.len().saturating_sub(1));
                                    let p80 = neurogpt_calib_top1[p_idx];
                                    p.min_prob = p80.clamp(0.4, 0.9);
                                }
                                neurogpt_gate.set_params(p);
                                neurogpt_gate.reset_baseline(mean, var);
                                tx.send(BciMessage::Log(format!(
                                    "NeuroGPT calibration done: mean_margin={:.4}, std={:.4}, k_sigma={:.2}, min_prob={:.2}",
                                    mean,
                                    std,
                                    p.k_sigma,
                                    p.min_prob
                                )))
                                .ok();
                                tx.send(BciMessage::NeuroGptStatus(NeuroGptRuntimeStatus {
                                    onnx_loaded: true,
                                    onnx_path: Some("model/neurogpt.onnx".to_owned()),
                                    last_error: None,
                                    last_infer_ms_ago: last_neurogpt_success
                                        .map(|t| t.elapsed().as_millis() as u64),
                                    gate: neurogpt_gate.params(),
                                }))
                                .ok();
                                tx.send(BciMessage::NeuroGptCalibrationProgress { progress01: 0.0 })
                                    .ok();
                            }
                        }
                        if last_neurogpt_status_emit.elapsed() >= Duration::from_secs(2) {
                            last_neurogpt_status_emit = Instant::now();
                            tx.send(BciMessage::NeuroGptStatus(NeuroGptRuntimeStatus {
                                onnx_loaded: true,
                                onnx_path: Some("model/neurogpt.onnx".to_owned()),
                                last_error: None,
                                last_infer_ms_ago: last_neurogpt_success
                                    .map(|t| t.elapsed().as_millis() as u64),
                                gate: neurogpt_gate.params(),
                            }))
                            .ok();
                        }
                    }

                    // === 驱动 vJoy ===
                    // 只有当状态发生改变 或 每隔一定时间才更新，减少系统调用开销
                    // 这里为了响应速度，每帧都更新
                    if let Some(joy) = &mut joystick {
                        joy.set_button(1, gp.a);
                        joy.set_button(2, gp.b);
                        joy.set_axis(0x30, (16384.0 + gp.lx * 16000.0) as i32);
                        joy.set_axis(0x31, (16384.0 + gp.ly * 16000.0) as i32);
                        // ... 其他按键映射同理
                    }
                    
                    // 发送手柄状态给 UI 显示
                    if last_vjoy_update.elapsed().as_millis() > 30 {
                        tx.send(BciMessage::GamepadUpdate(gp)).ok();
                        last_vjoy_update = Instant::now();
                    }
                }
            } else {
                // 未推流时，降低 CPU 占用
                thread::sleep(Duration::from_millis(50));
            }
        }
    });
}
