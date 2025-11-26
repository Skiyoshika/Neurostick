// src/engine.rs
use crate::openbci::OpenBciSession;
use crate::recorder::DataRecorder;
use crate::types::*;
use crate::vjoy::VJoyClient;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::{Duration, Instant};

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
    if match_pattern(&[12, 0]) {
        gp.ry += 1.0;
    } // I (Up)
    if match_pattern(&[13, 1]) {
        gp.ry -= 1.0;
    } // K (Down)
    if match_pattern(&[14, 2]) {
        gp.rx -= 1.0;
    } // J (Left)
    if match_pattern(&[15, 3]) {
        gp.rx += 1.0;
    } // L (Right)

    // --- 肩键/扳机 (QEUO) ---
    if match_pattern(&[0, 15]) {
        gp.lb = true;
    } // U
    if match_pattern(&[2, 13]) {
        gp.rb = true;
    } // O
    if match_pattern(&[1, 14]) {
        gp.lt = true;
    } // Q
    if match_pattern(&[3, 12]) {
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

        let mut recorder = DataRecorder::new();
        let mut openbci: Option<OpenBciSession> = None;

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
                                            openbci = Some(session);
                                            is_active = true;
                                            tx.send(BciMessage::Status(true)).ok();
                                            tx.send(BciMessage::Log(format!(
                                                "✅ OpenBCI Connected on {}",
                                                port
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
                            if let Some(session) = openbci.as_mut() {
                                session.stop_stream().ok();
                            }
                            openbci = None;
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
                let mut channel_data = vec![0.0f64; 16];

                // === 2. 信号生成 (生成端：必须生成解码器需要的组合) ===
                if current_mode == ConnectionMode::Simulation {
                    sim_phase += 0.1;
                    // 底噪
                    for i in 0..16 {
                        channel_data[i] = (sim_phase * (i as f64 * 0.1 + 1.0)).sin() * 2.0;
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
                        for i in 0..16 {
                            channel_data[i] += amp;
                        }
                        inject_artifact_frames -= 1;
                    }

                    thread::sleep(Duration::from_millis(5));
                } else if let Some(session) = openbci.as_mut() {
                    if let Some(sample) = session.next_sample() {
                        for (i, v) in sample.iter().take(16).enumerate() {
                            channel_data[i] = *v;
                        }
                    } else {
                        thread::sleep(Duration::from_millis(2));
                        continue;
                    }
                }

                if recorder.is_recording() {
                    recorder.write_record(&channel_data);
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
                }

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
