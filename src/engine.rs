// src/engine.rs
use crate::drivers::{SignalBatch, SignalBuffer};
use crate::openbci::OpenBciSession;
use crate::recorder::DataRecorder;
use crate::types::*;
use crate::vjoy::VJoyClient;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant, SystemTime};

/// 🧠 神经意图解码器 (接收端：检查特征)
fn process_neural_intent(
    data: &[f64],
    threshold: f64,
    calib_mode: bool,
    calib_max: &mut f64,
    start_time: Instant,
    tx: &Sender<BciMessage>,
) -> GamepadState {
    let mut gp = GamepadState::default();

    let is_active =
        |idx: usize| -> bool { data.get(idx).map(|&v| v.abs() > threshold).unwrap_or(false) };

    // 辅助：检查组合模式
    let match_pattern = |indices: &[usize]| -> bool { indices.iter().all(|&i| is_active(i)) };

    // =========================================================================
    // 1. 解码表 (Decoder) - 必须与下面的生成表 (Encoder) 完全一致
    // =========================================================================

    // --- 左摇杆 (WASD) ---
    if match_pattern(&[0, 4, 8]) {
        gp.ly += 1.0;
    } // W
    if match_pattern(&[1, 5, 9]) {
        gp.ly -= 1.0;
    } // S
    if match_pattern(&[2, 6, 10]) {
        gp.lx -= 1.0;
    } // A
    if match_pattern(&[3, 7, 11]) {
        gp.lx += 1.0;
    } // D

    // --- 动作键 (Space/ZXC) ---
    // 修复：这里定义了每个键需要的通道组合
    if match_pattern(&[0, 1, 2]) {
        gp.a = true;
    } // Space
    if match_pattern(&[3, 4, 5]) {
        gp.b = true;
    } // Z
    if match_pattern(&[6, 7, 8]) {
        gp.x = true;
    } // X
    if match_pattern(&[9, 10, 11]) {
        gp.y = true;
    } // C

    // --- 右摇杆 (IJKL) ---
    // 修复：使用独特的跨通道组合
    let rs_up = match_pattern(&[12, 0]);
    let rs_down = match_pattern(&[13, 1]);
    let rs_left = match_pattern(&[14, 2]);
    let rs_right = match_pattern(&[15, 3]);

    if rs_up {
        gp.ry += 1.0;
    } // I (Up)
    if rs_down {
        gp.ry -= 1.0;
    } // K (Down)
    if rs_left {
        gp.rx -= 1.0;
    } // J (Left)
    if rs_right {
        gp.rx += 1.0;
    } // L (Right)

    // --- 肩键/扳机 (QEUO) ---
    if match_pattern(&[0, 15]) && !(rs_up || rs_right) {
        gp.lb = true;
    } // U
    if match_pattern(&[2, 13]) && !(rs_left || rs_down) {
        gp.rb = true;
    } // O
    if match_pattern(&[1, 14]) && !(rs_left || rs_down) {
        gp.lt = true;
    } // Q
    if match_pattern(&[3, 12]) && !(rs_up || rs_right) {
        gp.rt = true;
    } // E

    // --- D-Pad (方向键) ---
    if match_pattern(&[4, 12]) {
        gp.dpad_up = true;
    }
    if match_pattern(&[5, 13]) {
        gp.dpad_down = true;
    }
    if match_pattern(&[6, 14]) {
        gp.dpad_left = true;
    }
    if match_pattern(&[7, 15]) {
        gp.dpad_right = true;
    }

    // 2. 校准逻辑
    if calib_mode {
        let max_s = data.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if max_s > *calib_max {
            *calib_max = max_s;
        }
        if start_time.elapsed().as_secs() >= 3 {
            tx.send(BciMessage::CalibrationResult((), *calib_max)).ok();
        }
    }

    gp
}

pub fn spawn_thread(tx: Sender<BciMessage>, rx_cmd: Receiver<GuiCommand>) {
    thread::spawn(move || {
        tx.send(BciMessage::Log(
            "⚙️ Engine V13.1 (Synced Logic).".to_owned(),
        ))
        .ok();

        let mut joystick = match VJoyClient::new(1) {
            Ok(j) => {
                tx.send(BciMessage::VJoyStatus(true)).ok();
                Some(j)
            }
            Err(_) => {
                tx.send(BciMessage::VJoyStatus(false)).ok();
                None
            }
        };
        let mut last_vjoy_attempt = Instant::now();
        let mut vjoy_logged_failure = joystick.is_none();

        let mut recorder = DataRecorder::new();
        let mut openbci: Option<OpenBciSession> = None;

        let mut signal_buffer: Option<SignalBuffer> = None;
        let mut channel_labels: Vec<String> = Vec::new();
        let mut current_sample_rate_hz: f32 = 250.0;

        let mut current_mode = ConnectionMode::Simulation;
        let mut is_active = false;
        let mut is_streaming = false;
        let mut threshold = 200.0;

        let mut sim_phase = 0.0;
        let mut current_sim_input = SimInputIntent::default();

        let mut calib_mode = false;
        let mut calib_max_val = 0.0;
        let mut calib_start_time = Instant::now();
        let mut inject_artifact_frames = 0;

        let mut channel_data = vec![0.0f64; 16];

        loop {
            // 1. 消息处理
            for _ in 0..10 {
                if let Ok(cmd) = rx_cmd.try_recv() {
                    match cmd {
                        GuiCommand::Connect(mode, port) => {
                            if !is_active {
                                current_mode = mode;
                                if mode == ConnectionMode::Simulation {
                                    is_active = true;
                                    tx.send(BciMessage::Status(true)).ok();
                                    tx.send(BciMessage::Log("✅ Sim Connected".to_owned())).ok();
                                } else {
                                    match OpenBciSession::connect(&port) {
                                        Ok(session) => {
                                            current_sample_rate_hz = session.sample_rate_hz();
                                            openbci = Some(session);
                                            is_active = true;
                                            tx.send(BciMessage::Status(true)).ok();
                                            tx.send(BciMessage::Log(format!(
                                                "✅ OpenBCI Connected on {} ({} Hz)",
                                                port, current_sample_rate_hz
                                            )))
                                            .ok();
                                        }
                                        Err(e) => {
                                            tx.send(BciMessage::Log(format!(
                                                "❌ Connect Failed: {}",
                                                e
                                            )))
                                            .ok();
                                        }
                                    }
                                }
                            }
                        }
                        GuiCommand::Disconnect => {
                            is_active = false;
                            is_streaming = false;
                            signal_buffer = None;
                            channel_labels.clear();
                            if let Some(session) = openbci.as_mut() {
                                session.stop_stream().ok();
                            }
                            openbci = None;
                            current_sample_rate_hz = 250.0;
                            if recorder.is_recording() {
                                recorder.stop();
                                tx.send(BciMessage::RecordingStatus(false)).ok();
                            }
                            tx.send(BciMessage::Status(false)).ok();
                        }
                        GuiCommand::StartStream => {
                            if is_active {
                                is_streaming = true;
                                if current_mode == ConnectionMode::Hardware {
                                    if let Some(session) = openbci.as_mut() {
                                        if let Err(e) = session.start_stream() {
                                            tx.send(BciMessage::Log(format!(
                                                "❌ Start Stream Failed: {}",
                                                e
                                            )))
                                            .ok();
                                            is_streaming = false;
                                        }
                                    }
                                }
                                tx.send(BciMessage::Log("🌊 Stream Started".to_owned()))
                                    .ok();
                            }
                        }
                        GuiCommand::StopStream => {
                            is_streaming = false;
                            signal_buffer = None;
                            channel_labels.clear();
                            if current_mode == ConnectionMode::Hardware {
                                if let Some(session) = openbci.as_mut() {
                                    session.stop_stream().ok();
                                }
                            }
                            tx.send(BciMessage::Log("🛑 Stream Stopped".to_owned()))
                                .ok();
                        }
                        GuiCommand::SetThreshold(v) => threshold = v,
                        GuiCommand::StartCalibration(_) => {
                            calib_mode = true;
                            calib_max_val = 0.0;
                            calib_start_time = Instant::now();
                        }
                        GuiCommand::UpdateSimInput(input) => current_sim_input = input,
                        GuiCommand::StartRecording(label) => {
                            recorder.start(&label);
                            tx.send(BciMessage::RecordingStatus(true)).ok();
                        }
                        GuiCommand::StopRecording => {
                            recorder.stop();
                            tx.send(BciMessage::RecordingStatus(false)).ok();
                        }
                        GuiCommand::InjectArtifact => {
                            inject_artifact_frames = 20;
                            tx.send(BciMessage::Log("💉 Injecting...".to_owned())).ok();
                        }
                    }
                } else {
                    break;
                }
            }

            if is_streaming {
                channel_data.fill(0.0);
                // === 2. 信号生成 (生成端：必须生成解码器需要的组合) ===
                if current_mode == ConnectionMode::Simulation {
                    sim_phase += 0.1;
                    // 底噪
                    for (i, v) in channel_data.iter_mut().enumerate() {
                        *v = (sim_phase * (i as f64 * 0.1 + 1.0)).sin() * 2.0;
                    }

                    let amp = 1000.0;

                    // 左摇杆 (WASD) -> [0,4,8], [1,5,9], [2,6,10], [3,7,11]
                    if current_sim_input.w {
                        channel_data[0] += amp;
                        channel_data[4] += amp;
                        channel_data[8] += amp;
                    }
                    if current_sim_input.s {
                        channel_data[1] += amp;
                        channel_data[5] += amp;
                        channel_data[9] += amp;
                    }
                    if current_sim_input.a {
                        channel_data[2] += amp;
                        channel_data[6] += amp;
                        channel_data[10] += amp;
                    }
                    if current_sim_input.d {
                        channel_data[3] += amp;
                        channel_data[7] += amp;
                        channel_data[11] += amp;
                    }

                    // 动作键 (Space/ZXC) -> [0,1,2], [3,4,5], [6,7,8], [9,10,11]
                    // 修复：严格对齐解码器要求的通道
                    if current_sim_input.space {
                        channel_data[0] += amp;
                        channel_data[1] += amp;
                        channel_data[2] += amp;
                    } // A
                    if current_sim_input.key_z {
                        channel_data[3] += amp;
                        channel_data[4] += amp;
                        channel_data[5] += amp;
                    } // B
                    if current_sim_input.key_x {
                        channel_data[6] += amp;
                        channel_data[7] += amp;
                        channel_data[8] += amp;
                    } // X
                    if current_sim_input.key_c {
                        channel_data[9] += amp;
                        channel_data[10] += amp;
                        channel_data[11] += amp;
                    } // Y

                    // 右摇杆 (IJKL) -> [12,0], [13,1], [14,2], [15,3]
                    // 修复：注入对应的跨半球信号
                    if current_sim_input.up {
                        channel_data[12] += amp;
                        channel_data[0] += amp;
                    } // I
                    if current_sim_input.down {
                        channel_data[13] += amp;
                        channel_data[1] += amp;
                    } // K
                    if current_sim_input.left {
                        channel_data[14] += amp;
                        channel_data[2] += amp;
                    } // J
                    if current_sim_input.right {
                        channel_data[15] += amp;
                        channel_data[3] += amp;
                    } // L

                    // 肩键 (QEUO) -> [1,14], [3,12], [0,15], [2,13]
                    // 修复：注入对应的信号
                    if current_sim_input.u {
                        channel_data[0] += amp;
                        channel_data[15] += amp;
                    } // LB (U)
                    if current_sim_input.o {
                        channel_data[2] += amp;
                        channel_data[13] += amp;
                    } // RB (O)
                    if current_sim_input.q {
                        channel_data[1] += amp;
                        channel_data[14] += amp;
                    } // LT (Q)
                    if current_sim_input.e {
                        channel_data[3] += amp;
                        channel_data[12] += amp;
                    } // RT (E)

                    // 方向键 (Arrows) -> [4,12], [5,13], [6,14], [7,15]
                    // 修复：注入对应的信号
                    if current_sim_input.arrow_up {
                        channel_data[4] += amp;
                        channel_data[12] += amp;
                    }
                    if current_sim_input.arrow_down {
                        channel_data[5] += amp;
                        channel_data[13] += amp;
                    }
                    if current_sim_input.arrow_left {
                        channel_data[6] += amp;
                        channel_data[14] += amp;
                    }
                    if current_sim_input.arrow_right {
                        channel_data[7] += amp;
                        channel_data[15] += amp;
                    }

                    if inject_artifact_frames > 0 {
                        // 模拟惊吓：全脑激活
                        for v in channel_data.iter_mut() {
                            *v += amp;
                        }
                        inject_artifact_frames -= 1;
                    }

                    // 更短的忙等待 + 更长的休眠，降低 CPU 抢占
                    thread::sleep(Duration::from_millis(8));
                } else if let Some(session) = openbci.as_mut() {
                    match session.next_sample() {
                        Ok(Some(sample)) => {
                            for (i, v) in sample.iter().take(16).enumerate() {
                                channel_data[i] = *v;
                            }
                        }
                        Ok(None) => {
                            thread::sleep(Duration::from_millis(2));
                        }
                        Err(e) => {
                            tx.send(BciMessage::Log(format!("BrainFlow read failed: {e}")))
                                .ok();
                            thread::sleep(Duration::from_millis(20));
                        }
                    }
                }

                if recorder.is_recording() {
                    recorder.write_record(&channel_data);
                }

                // === 驱动层：更新滚动缓冲并推送给 GUI ===
                const DEFAULT_SAMPLE_RATE_HZ: f32 = 250.0;
                const HISTORY_SECONDS: f32 = 60.0;
                let sample_rate_hz = if current_mode == ConnectionMode::Hardware {
                    current_sample_rate_hz
                } else {
                    DEFAULT_SAMPLE_RATE_HZ
                };

                if signal_buffer.is_none() || channel_labels.len() != channel_data.len() {
                    channel_labels = (0..channel_data.len())
                        .map(|i| format!("Ch{}", i + 1))
                        .collect();
                    match SignalBuffer::with_history_seconds(
                        channel_labels.clone(),
                        sample_rate_hz,
                        HISTORY_SECONDS,
                    ) {
                        Ok(buf) => signal_buffer = Some(buf),
                        Err(e) => {
                            tx.send(BciMessage::Log(format!("Buffer init failed: {e}")))
                                .ok();
                        }
                    }
                }

                if let Some(buf) = signal_buffer.as_mut() {
                    let samples: Vec<Vec<f32>> =
                        channel_data.iter().map(|v| vec![*v as f32]).collect();
                    let batch = SignalBatch {
                        started_at: SystemTime::now(),
                        sample_rate_hz,
                        samples,
                        channel_labels: channel_labels.clone(),
                    };
                    match buf.push_batch(&batch) {
                        Ok(()) => {
                            let frame = buf.snapshot(HISTORY_SECONDS);
                            tx.send(BciMessage::DataFrame(frame)).ok();
                        }
                        Err(e) => {
                            tx.send(BciMessage::Log(format!("Buffer push failed: {e}")))
                                .ok();
                        }
                    }
                }

                // === 3. 解码 (这里会调用上面对齐过的逻辑) ===
                let gp = process_neural_intent(
                    &channel_data,
                    threshold,
                    calib_mode,
                    &mut calib_max_val,
                    calib_start_time,
                    &tx,
                );

                // 4. 执行
                if let Some(joy) = &mut joystick {
                    joy.set_button(1, gp.a);
                    joy.set_button(2, gp.b);
                    joy.set_button(3, gp.x);
                    joy.set_button(4, gp.y);
                    joy.set_button(5, gp.lb);
                    joy.set_button(6, gp.rb);
                    joy.set_button(7, gp.lt);
                    joy.set_button(8, gp.rt);
                    joy.set_button(9, gp.dpad_up);
                    joy.set_button(10, gp.dpad_down);
                    joy.set_button(11, gp.dpad_left);
                    joy.set_button(12, gp.dpad_right);

                    let to_axis = |v: f32| (16384.0 + v * 16000.0) as i32;
                    joy.set_axis(0x30, to_axis(gp.lx));
                    joy.set_axis(0x31, to_axis(gp.ly));
                    joy.set_axis(0x32, to_axis(gp.rx));
                    joy.set_axis(0x33, to_axis(gp.ry));
                } else if last_vjoy_attempt.elapsed().as_secs() >= 5 {
                    // 自动重连 vJoy，确保开机时即可映射
                    last_vjoy_attempt = Instant::now();
                    match VJoyClient::new(1) {
                        Ok(j) => {
                            joystick = Some(j);
                            vjoy_logged_failure = false;
                            tx.send(BciMessage::VJoyStatus(true)).ok();
                            tx.send(BciMessage::Log("vJoy reconnected".to_owned()))
                                .ok();
                        }
                        Err(e) => {
                            if !vjoy_logged_failure {
                                tx.send(BciMessage::Log(format!(
                                    "vJoy connect failed: {e}"
                                )))
                                .ok();
                                vjoy_logged_failure = true;
                            }
                            tx.send(BciMessage::VJoyStatus(false)).ok();
                        }
                    }
                }

                if sim_phase as i32 % 2 == 0 {
                    tx.send(BciMessage::GamepadUpdate(gp)).ok();
                }
            } else {
                // 即便未采集也尝试维持 vJoy 连接，保证打开应用即可映射
                if joystick.is_none() && last_vjoy_attempt.elapsed().as_secs() >= 5 {
                    last_vjoy_attempt = Instant::now();
                    match VJoyClient::new(1) {
                        Ok(j) => {
                            joystick = Some(j);
                            vjoy_logged_failure = false;
                            tx.send(BciMessage::VJoyStatus(true)).ok();
                            tx.send(BciMessage::Log("vJoy connected".to_owned()))
                                .ok();
                        }
                        Err(e) => {
                            if !vjoy_logged_failure {
                                tx.send(BciMessage::Log(format!(
                                    "vJoy connect failed: {e}"
                                )))
                                .ok();
                                vjoy_logged_failure = true;
                            }
                            tx.send(BciMessage::VJoyStatus(false)).ok();
                        }
                    }
                }
                thread::sleep(Duration::from_millis(50));
            }
        }
    });
}
