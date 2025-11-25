// src/engine.rs
use crate::types::*;
use crate::vjoy::VJoyClient;
use crate::recorder::DataRecorder;
use libloading::{Library, Symbol};
use std::ffi::CString;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

/// 🧠 意图特征解码器
/// 这里模拟了 AI 模型的推断过程：从复杂的 16 通道信号中识别特定的“模式”
fn decode_neural_intent(
    data: &[f64], 
    threshold: f64, 
    calib_mode: bool,
    calib_max: &mut f64,
    start_time: Instant,
    tx: &Sender<BciMessage>
) -> GamepadState {
    let mut gp = GamepadState::default();
    
    // 辅助函数：检查特征组合是否满足
    // 只有当 indices 中列出的所有通道的信号强度都超过阈值时，才返回 true
    let check_pattern = |indices: &[usize]| -> bool {
        indices.iter().all(|&idx| data[idx].abs() > threshold)
    };

    // =========================================================================
    // 1. 脑波特征映射表 (Brainwave Feature Mapping)
    // 这里定义了每个“动作意图”对应的“脑区协同模式”
    // =========================================================================

    // --- 左摇杆 (移动意图) ---
    // 模拟运动皮层 (C3/C4) 的协同模式
    if check_pattern(&[0, 4]) { gp.ly += 1.0; } // W (前进): 激活 Ch0 + Ch4
    if check_pattern(&[1, 5]) { gp.ly -= 1.0; } // S (后退): 激活 Ch1 + Ch5
    if check_pattern(&[2, 6]) { gp.lx -= 1.0; } // A (向左): 激活 Ch2 + Ch6
    if check_pattern(&[3, 7]) { gp.lx += 1.0; } // D (向右): 激活 Ch3 + Ch7

    // --- 右摇杆 (视角/注意力意图) ---
    // 模拟枕叶 (O1/O2) 视觉区的协同
    if check_pattern(&[8, 12])  { gp.ry += 1.0; } // I (看上): Ch8 + Ch12
    if check_pattern(&[9, 13])  { gp.ry -= 1.0; } // K (看下): Ch9 + Ch13
    if check_pattern(&[10, 14]) { gp.rx -= 1.0; } // J (看左): Ch10 + Ch14
    if check_pattern(&[11, 15]) { gp.rx += 1.0; } // L (看右): Ch11 + Ch15

    // --- 动作键 ABXY (高频爆发指令) ---
    // 模拟更复杂的跨脑区协同，需要3个通道同时激活
    if check_pattern(&[0, 1, 2]) { gp.a = true; } // Space (跳跃/确认): 额叶强激活
    if check_pattern(&[2, 3, 4]) { gp.b = true; } // Z (B键)
    if check_pattern(&[4, 5, 6]) { gp.x = true; } // X (攻击/物品)
    if check_pattern(&[6, 7, 0]) { gp.y = true; } // C (Y键)

    // --- 肩键/扳机 (特殊功能) ---
    // 模拟特定频率的信号组合
    if check_pattern(&[8, 9, 10])    { gp.lb = true; } // U (LB): 防御
    if check_pattern(&[10, 11, 12])  { gp.rb = true; } // O (RB): 轻攻击
    if check_pattern(&[12, 13, 14])  { gp.lt = true; } // Q (LT): 战技
    if check_pattern(&[13, 14, 15])  { gp.rt = true; } // E (RT): 重攻击

    // --- D-Pad (辅助指令) ---
    // 模拟跨半球的长距离连接 (Cross-Hemisphere Sync)
    if check_pattern(&[0, 15]) { gp.dpad_up = true; }    // Up: 首尾呼应
    if check_pattern(&[3, 12]) { gp.dpad_down = true; }  // Down
    if check_pattern(&[4, 11]) { gp.dpad_left = true; }  // Left
    if check_pattern(&[7, 8])  { gp.dpad_right = true; } // Right

    // 2. 校准逻辑
    if calib_mode {
        let max_s = data.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        if max_s > *calib_max { *calib_max = max_s; }
        if start_time.elapsed().as_secs() >= 3 {
            tx.send(BciMessage::CalibrationResult((), *calib_max)).ok();
        }
    }

    gp
}

pub fn spawn_thread(tx: Sender<BciMessage>, rx_cmd: Receiver<GuiCommand>) {
    thread::spawn(move || {
        tx.send(BciMessage::Log("⚙️ Core Engine v9.0 (Neural Pattern).".to_owned())).ok();
        
        let mut joystick = match VJoyClient::new(1) {
            Ok(j) => { tx.send(BciMessage::VJoyStatus(true)).ok(); Some(j) },
            Err(_) => { tx.send(BciMessage::VJoyStatus(false)).ok(); None }
        };

        let mut recorder = DataRecorder::new();
        // 即使没有DLL也能运行模拟模式
        let lib_opt = unsafe { Library::new("BoardController.dll").ok() };
        
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

        loop {
            // 1. 消息处理 (保持高效，每帧最多处理10条)
            for _ in 0..10 { 
                if let Ok(cmd) = rx_cmd.try_recv() {
                    match cmd {
                        GuiCommand::Connect(mode) => {
                            if !is_active {
                                current_mode = mode;
                                if mode == ConnectionMode::Simulation {
                                    is_active = true;
                                    tx.send(BciMessage::Status(true)).ok();
                                    tx.send(BciMessage::Log("✅ Sim Connected".to_owned())).ok();
                                } else if let Some(lib) = &lib_opt {
                                    unsafe {
                                        let prepare: Symbol<unsafe extern "C" fn(i32, *const i8) -> i32> = lib.get(b"prepare_session").unwrap();
                                        let p_str = r#"{"serial_port":"COM4","timeout":3,"master_board":-100,"file":"","file_anc":"","file_aux":"","ip_address":"","ip_address_anc":"","ip_address_aux":"","ip_port":0,"ip_port_anc":0,"ip_port_aux":0,"ip_protocol":0,"mac_address":"","other_info":"","serial_number":""}"#;
                                        let params = CString::new(p_str).unwrap();
                                        if prepare(2, params.as_ptr()) == 0 {
                                            is_active = true;
                                            tx.send(BciMessage::Status(true)).ok();
                                            tx.send(BciMessage::Log("✅ Hardware Connected".to_owned())).ok();
                                        } else {
                                            tx.send(BciMessage::Log("❌ Connect Failed".to_owned())).ok();
                                        }
                                    }
                                }
                            }
                        },
                        GuiCommand::Disconnect => { 
                            is_active = false; is_streaming = false; 
                            if recorder.is_recording() { recorder.stop(); tx.send(BciMessage::RecordingStatus(false)).ok(); }
                            tx.send(BciMessage::Status(false)).ok(); 
                        },
                        GuiCommand::StartStream => { 
                            if is_active { 
                                is_streaming = true; 
                                if current_mode == ConnectionMode::Hardware {
                                    if let Some(lib) = &lib_opt { unsafe { let start: Symbol<unsafe extern "C" fn(i32, *const i8) -> i32> = lib.get(b"start_stream").unwrap(); let e = CString::new("").unwrap(); start(45000, e.as_ptr()); } }
                                }
                                tx.send(BciMessage::Log("🌊 Stream Started".to_owned())).ok();
                            } 
                        },
                        GuiCommand::StopStream => { 
                            is_streaming = false; 
                            if current_mode == ConnectionMode::Hardware {
                                if let Some(lib) = &lib_opt { unsafe { let stop: Symbol<unsafe extern "C" fn(i32) -> i32> = lib.get(b"stop_stream").unwrap(); stop(2); } }
                            }
                            tx.send(BciMessage::Log("🛑 Stream Stopped".to_owned())).ok();
                        },
                        GuiCommand::SetThreshold(v) => threshold = v,
                        GuiCommand::StartCalibration(_) => { calib_mode = true; calib_max_val = 0.0; calib_start_time = Instant::now(); },
                        GuiCommand::UpdateSimInput(input) => current_sim_input = input,
                        GuiCommand::StartRecording(label) => { recorder.start(&label); tx.send(BciMessage::RecordingStatus(true)).ok(); },
                        GuiCommand::StopRecording => { recorder.stop(); tx.send(BciMessage::RecordingStatus(false)).ok(); },
                        GuiCommand::InjectArtifact => { inject_artifact_frames = 20; tx.send(BciMessage::Log("💉 Injecting...".to_owned())).ok(); }
                    }
                } else {
                    break; 
                }
            }

            // 2. 数据循环
            if is_streaming {
                // 严格限制为 16 通道 (对应 Cyton+Daisy)
                let mut channel_data = vec![0.0f64; 16];

                // === 模拟信号生成：将按键意图转化为特定的脑波组合 ===
                if current_mode == ConnectionMode::Simulation {
                    sim_phase += 0.1;
                    // 基础底噪 (Alpha波模拟)
                    for i in 0..16 { channel_data[i] = (sim_phase * (i as f64 * 0.1 + 1.0)).sin() * 5.0; }
                    
                    let amp = 1000.0; // 强激活信号
                    
                    // 模拟：按下 W -> 激活 Ch0 和 Ch4
                    if current_sim_input.w { channel_data[0] += amp; channel_data[4] += amp; }
                    if current_sim_input.s { channel_data[1] += amp; channel_data[5] += amp; }
                    if current_sim_input.a { channel_data[2] += amp; channel_data[6] += amp; }
                    if current_sim_input.d { channel_data[3] += amp; channel_data[7] += amp; }

                    // 模拟：右摇杆 -> 激活后部通道
                    if current_sim_input.up    { channel_data[8] += amp; channel_data[12] += amp; }
                    if current_sim_input.down  { channel_data[9] += amp; channel_data[13] += amp; }
                    if current_sim_input.left  { channel_data[10] += amp; channel_data[14] += amp; }
                    if current_sim_input.right { channel_data[11] += amp; channel_data[15] += amp; }

                    // 模拟：功能键 -> 激活3个通道的复杂模式
                    if current_sim_input.space { channel_data[0] += amp; channel_data[1] += amp; channel_data[2] += amp; }
                    if current_sim_input.key_z { channel_data[2] += amp; channel_data[3] += amp; channel_data[4] += amp; }
                    if current_sim_input.key_x { channel_data[4] += amp; channel_data[5] += amp; channel_data[6] += amp; }
                    if current_sim_input.key_c { channel_data[6] += amp; channel_data[7] += amp; channel_data[0] += amp; }

                    // 模拟：肩键
                    if current_sim_input.u { channel_data[8] += amp; channel_data[9] += amp; channel_data[10] += amp; }
                    if current_sim_input.o { channel_data[10] += amp; channel_data[11] += amp; channel_data[12] += amp; }
                    if current_sim_input.q { channel_data[12] += amp; channel_data[13] += amp; channel_data[14] += amp; }
                    if current_sim_input.e { channel_data[13] += amp; channel_data[14] += amp; channel_data[15] += amp; }

                    // 模拟：方向键 (跨半球连接)
                    if current_sim_input.arrow_up    { channel_data[0] += amp; channel_data[15] += amp; }
                    if current_sim_input.arrow_down  { channel_data[3] += amp; channel_data[12] += amp; }
                    if current_sim_input.arrow_left  { channel_data[4] += amp; channel_data[11] += amp; }
                    if current_sim_input.arrow_right { channel_data[7] += amp; channel_data[8] += amp; }
                    
                    // 伪迹注入
                    if inject_artifact_frames > 0 {
                        // 模拟全脑惊吓反应 (所有通道激活)
                        for i in 0..16 { channel_data[i] += amp; }
                        inject_artifact_frames -= 1;
                    }

                    thread::sleep(Duration::from_millis(5));
                } 
                // === 硬件数据读取 ===
                else if let Some(lib) = &lib_opt {
                    unsafe {
                        let get_cnt: Symbol<unsafe extern "C" fn(i32, *mut i32) -> i32> = lib.get(b"get_board_data_count").unwrap();
                        let get_dat: Symbol<unsafe extern "C" fn(i32, *mut f64) -> i32> = lib.get(b"get_board_data").unwrap();
                        let get_row: Symbol<unsafe extern "C" fn(i32, *mut i32) -> i32> = lib.get(b"get_num_rows").unwrap();
                        
                        let mut count = 0; get_cnt(2, &mut count);
                        if count > 0 {
                            let mut rows = 0; get_row(2, &mut rows);
                            let mut buf = vec![0.0f64; (rows * count) as usize];
                            get_dat(count, buf.as_mut_ptr());
                            // 取最新一个采样点
                            for i in 0..count {
                                let current_sample_index = i as usize;
                                for c in 0..16 {
                                    // Cyton 数据通常从 index 1 开始
                                    let row_idx = (c + 1) as usize;
                                    let idx = row_idx * (count as usize) + current_sample_index;
                                    if idx < buf.len() { channel_data[c] = buf[idx]; }
                                }
                            }
                        }
                    }
                    thread::sleep(Duration::from_millis(5));
                }

                // 录制原始数据
                if recorder.is_recording() { recorder.write_record(&channel_data); }

                // === 解码意图 (Processing) ===
                // 将采集到(或模拟出)的复杂波形，解码为手柄指令
                let gp = decode_neural_intent(
                    &channel_data, threshold, 
                    calib_mode, &mut calib_max_val, calib_start_time, 
                    &tx
                );

                // === 执行 vJoy ===
                if let Some(joy) = &mut joystick {
                    joy.set_button(1, gp.a); joy.set_button(2, gp.b);
                    joy.set_button(3, gp.x); joy.set_button(4, gp.y);
                    joy.set_button(5, gp.lb); joy.set_button(6, gp.rb); 
                    joy.set_button(7, gp.lt); joy.set_button(8, gp.rt);
                    
                    // 映射 D-Pad
                    joy.set_button(9, gp.dpad_up); joy.set_button(10, gp.dpad_down);
                    joy.set_button(11, gp.dpad_left); joy.set_button(12, gp.dpad_right);
                    
                    let to_axis = |v: f32| (16384.0 + v * 16000.0) as i32;
                    joy.set_axis(0x30, to_axis(gp.lx)); 
                    joy.set_axis(0x31, to_axis(gp.ly)); 
                    joy.set_axis(0x32, to_axis(gp.rx)); 
                    joy.set_axis(0x33, to_axis(gp.ry)); 
                }

                // 发送反馈
                if sim_phase as i32 % 2 == 0 {
                    tx.send(BciMessage::GamepadUpdate(gp)).ok();
                    tx.send(BciMessage::DataPacket(channel_data)).ok();
                }

            } else {
                thread::sleep(Duration::from_millis(50));
            }
        }
    });
}