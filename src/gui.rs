// src/gui.rs
use crate::engine;
use crate::types::*;
// 引入我们刚刚写好的模块
use crate::visualizer; 

use eframe::egui;
use egui::{Color32, Vec2};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use std::sync::mpsc::{channel, Receiver, Sender};

pub struct QnmdSolApp {
    is_connected: bool, is_vjoy_active: bool, is_streaming: bool, is_recording: bool,
    connection_mode: ConnectionMode, follow_latest: bool,
    time: f64,
    wave_buffers: Vec<Vec<[f64; 2]>>,
    view_seconds: f64, display_gain: f64, vertical_spacing: f64,
    gamepad_target: GamepadState, gamepad_visual: GamepadState,
    calib_rest_max: f64, calib_act_max: f64, is_calibrating: bool, calib_timer: f32,
    trigger_threshold: f64, record_label: String, language: Language, has_started: bool,
    selected_tab: String, log_messages: Vec<String>,
    rx: Receiver<BciMessage>, tx_cmd: Sender<GuiCommand>,
}

impl Default for QnmdSolApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let (tx_cmd, rx_cmd) = channel();
        engine::spawn_thread(tx, rx_cmd);
        let buffers = vec![Vec::new(); 16];
        Self {
            is_connected: false, is_vjoy_active: false, is_streaming: false, is_recording: false,
            connection_mode: ConnectionMode::Simulation, follow_latest: true,
            time: 0.0, wave_buffers: buffers, view_seconds: 8.0, display_gain: 0.35, vertical_spacing: 420.0,
            gamepad_target: GamepadState::default(), gamepad_visual: GamepadState::default(),
            calib_rest_max: 0.0, calib_act_max: 0.0, is_calibrating: false, calib_timer: 0.0,
            selected_tab: "Monitor".to_owned(), log_messages: vec![],
            trigger_threshold: 200.0, record_label: Language::English.default_record_label().to_owned(),
            language: Language::English, has_started: false,
            rx, tx_cmd,
        }
    }
}

impl QnmdSolApp {
    fn text(&self, key: UiText) -> &'static str { self.language.text(key) }
    fn reset_localized_defaults(&mut self) { self.log_messages.clear(); self.log(self.text(UiText::Ready)); self.record_label = self.language.default_record_label().to_owned(); }
    fn log(&mut self, msg: &str) { self.log_messages.push(format!("> {}", msg)); if self.log_messages.len() > 8 { self.log_messages.remove(0); } }
    fn lerp(current: f32, target: f32, speed: f32) -> f32 { current + (target - current) * speed }

    fn show_start_screen(&mut self, ctx: &egui::Context) {
        let mut visuals = egui::Visuals::light();
        visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(245, 245, 247);
        visuals.window_fill = Color32::from_rgb(250, 250, 252);
        ctx.set_visuals(visuals);
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.vertical_centered(|ui| {
                ui.add_space(100.0);
                ui.heading(egui::RichText::new("Welcome to QNMDsol world！").size(30.0).color(Color32::from_rgb(30, 30, 35)));
                ui.add_space(10.0);
                ui.label(egui::RichText::new(self.text(UiText::LanguagePrompt)).color(Color32::DARK_GRAY));
                ui.add_space(30.0);
                egui::Frame::none().fill(Color32::from_rgb(255, 255, 255)).rounding(egui::Rounding::same(24.0)).stroke(egui::Stroke::new(1.0, Color32::from_gray(220))).inner_margin(egui::style::Margin::symmetric(30.0, 20.0)).show(ui, |ui| {
                    ui.vertical_centered(|ui| {
                        ui.label(egui::RichText::new(self.text(UiText::StartSubtitle)).size(16.0));
                        ui.add_space(16.0);
                        ui.horizontal(|ui| {
                            if ui.add(egui::Button::new(egui::RichText::new("中文").size(16.0)).min_size(Vec2::new(140.0, 42.0)).rounding(egui::Rounding::same(12.0))).clicked() { self.language = Language::Chinese; self.has_started = true; self.reset_localized_defaults(); }
                            ui.add_space(14.0);
                            if ui.add(egui::Button::new(egui::RichText::new("English").size(16.0)).min_size(Vec2::new(140.0, 42.0)).rounding(egui::Rounding::same(12.0))).clicked() { self.language = Language::English; self.has_started = true; self.reset_localized_defaults(); }
                        });
                    });
                });
            });
        });
    }
}

impl eframe::App for QnmdSolApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.has_started { self.show_start_screen(ctx); return; }

        if self.connection_mode == ConnectionMode::Simulation {
            let mut input = SimInputIntent::default();
            if ctx.input(|i| i.key_down(egui::Key::W)) { input.w = true; }
            if ctx.input(|i| i.key_down(egui::Key::S)) { input.s = true; }
            if ctx.input(|i| i.key_down(egui::Key::A)) { input.a = true; }
            if ctx.input(|i| i.key_down(egui::Key::D)) { input.d = true; }
            if ctx.input(|i| i.key_down(egui::Key::Space)) { input.space = true; }
            if ctx.input(|i| i.key_down(egui::Key::Z)) { input.key_z = true; }
            if ctx.input(|i| i.key_down(egui::Key::X)) { input.key_x = true; }
            if ctx.input(|i| i.key_down(egui::Key::C)) { input.key_c = true; }
            if ctx.input(|i| i.key_down(egui::Key::I)) { input.up = true; }
            if ctx.input(|i| i.key_down(egui::Key::K)) { input.down = true; }
            if ctx.input(|i| i.key_down(egui::Key::J)) { input.left = true; }
            if ctx.input(|i| i.key_down(egui::Key::L)) { input.right = true; }
            if ctx.input(|i| i.key_down(egui::Key::Q)) { input.q = true; }
            if ctx.input(|i| i.key_down(egui::Key::E)) { input.e = true; }
            if ctx.input(|i| i.key_down(egui::Key::U)) { input.u = true; }
            if ctx.input(|i| i.key_down(egui::Key::O)) { input.o = true; }
            if ctx.input(|i| i.key_down(egui::Key::ArrowUp)) { input.arrow_up = true; }
            if ctx.input(|i| i.key_down(egui::Key::ArrowDown)) { input.arrow_down = true; }
            if ctx.input(|i| i.key_down(egui::Key::ArrowLeft)) { input.arrow_left = true; }
            if ctx.input(|i| i.key_down(egui::Key::ArrowRight)) { input.arrow_right = true; }
            self.tx_cmd.send(GuiCommand::UpdateSimInput(input)).ok();
        }

        let mut msg_count = 0;
        while let Ok(msg) = self.rx.try_recv() {
            msg_count += 1;
            if msg_count > 20 {
                match msg { BciMessage::GamepadUpdate(gp) => self.gamepad_target = gp, _ => continue, }
            } else {
                match msg {
                    BciMessage::Log(s) => self.log(&s),
                    BciMessage::Status(b) => self.is_connected = b,
                    BciMessage::VJoyStatus(b) => self.is_vjoy_active = b,
                    BciMessage::GamepadUpdate(gp) => self.gamepad_target = gp,
                    BciMessage::RecordingStatus(b) => self.is_recording = b,
                    BciMessage::DataPacket(data) => {
                        self.time += 0.02;
                        for (i, val) in data.iter().enumerate().take(self.wave_buffers.len()) {
                            if i < self.wave_buffers.len() {
                                let offset = i as f64 * self.vertical_spacing;
                                self.wave_buffers[i].push([self.time, *val * self.display_gain + offset]);
                                if self.wave_buffers[i].len() > 500 { self.wave_buffers[i].remove(0); }
                            }
                        }
                    }
                    BciMessage::CalibrationResult(_, max) => {
                        self.is_calibrating = false;
                        if self.calib_rest_max == 0.0 {
                            self.calib_rest_max = max; self.log(&format!("Base: {:.1}", max));
                        } else {
                            self.calib_act_max = max; self.log(&format!("Act: {:.1}", max));
                            let new = (self.calib_rest_max + self.calib_act_max) * 0.6;
                            self.trigger_threshold = new;
                            self.tx_cmd.send(GuiCommand::SetThreshold(new)).unwrap();
                            self.log(&format!("Threshold: {:.1}", new));
                        }
                    }
                }
            }
        }
        
        let speed = 0.3;
        self.gamepad_visual.lx = Self::lerp(self.gamepad_visual.lx, self.gamepad_target.lx, speed);
        self.gamepad_visual.ly = Self::lerp(self.gamepad_visual.ly, self.gamepad_target.ly, speed);
        self.gamepad_visual.rx = Self::lerp(self.gamepad_visual.rx, self.gamepad_target.rx, speed);
        self.gamepad_visual.ry = Self::lerp(self.gamepad_visual.ry, self.gamepad_target.ry, speed);
        self.gamepad_visual.a = self.gamepad_target.a;
        self.gamepad_visual.b = self.gamepad_target.b;
        self.gamepad_visual.x = self.gamepad_target.x;
        self.gamepad_visual.y = self.gamepad_target.y;
        self.gamepad_visual.lb = self.gamepad_target.lb;
        self.gamepad_visual.rb = self.gamepad_target.rb;
        self.gamepad_visual.lt = self.gamepad_target.lt;
        self.gamepad_visual.rt = self.gamepad_target.rt;
        self.gamepad_visual.dpad_up = self.gamepad_target.dpad_up;
        self.gamepad_visual.dpad_down = self.gamepad_target.dpad_down;
        self.gamepad_visual.dpad_left = self.gamepad_target.dpad_left;
        self.gamepad_visual.dpad_right = self.gamepad_target.dpad_right;

        if self.is_streaming { ctx.request_repaint(); }
        if self.is_calibrating { self.calib_timer -= ctx.input(|i| i.stable_dt); if self.calib_timer < 0.0 { self.calib_timer = 0.0; } ctx.request_repaint(); }

        let mut visuals = egui::Visuals::dark();
        visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(10, 10, 15);
        ctx.set_visuals(visuals);

        egui::SidePanel::left("L").min_width(340.0).show(ctx, |ui| {
            ui.add_space(10.0);
            ui.heading(self.text(UiText::Title));
            ui.label(self.text(UiText::Subtitle));
            ui.separator();
            ui.horizontal(|ui| {
                let sim_label = self.text(UiText::Sim);
                let real_label = self.text(UiText::Real);
                ui.selectable_value(&mut self.connection_mode, ConnectionMode::Simulation, sim_label);
                ui.selectable_value(&mut self.connection_mode, ConnectionMode::Hardware, real_label);
            });
            let btn_txt = if self.is_connected { self.text(UiText::Disconnect) } else { self.text(UiText::Connect) };
            if ui.button(btn_txt).clicked() {
                if !self.is_connected { self.tx_cmd.send(GuiCommand::Connect(self.connection_mode)).unwrap(); } else { self.tx_cmd.send(GuiCommand::Disconnect).unwrap(); }
            }
            if self.is_connected {
                let stream_btn = if self.is_streaming { self.text(UiText::StopStream) } else { self.text(UiText::StartStream) };
                if ui.button(stream_btn).clicked() {
                    if self.is_streaming { self.tx_cmd.send(GuiCommand::StopStream).unwrap(); self.is_streaming = false; }
                    else { self.tx_cmd.send(GuiCommand::StartStream).unwrap(); self.is_streaming = true; }
                }
                if ui.button(self.text(UiText::ResetView)).clicked() { for buf in &mut self.wave_buffers { buf.clear(); } self.time = 0.0; }
                let follow_label = if self.follow_latest { self.text(UiText::FollowOn) } else { self.text(UiText::FollowOff) };
                if ui.button(follow_label).clicked() { self.follow_latest = !self.follow_latest; }
            }
            ui.add_space(20.0);
            ui.label(self.text(UiText::Controller));
            
            // === 核心：调用分离出来的 visualizer 模块 ===
            visualizer::draw_xbox_controller(ui, &self.gamepad_visual);
            
            ui.add_space(20.0);
            ui.separator();
            ui.label(self.text(UiText::Data));
            ui.text_edit_singleline(&mut self.record_label);
            let can_record = self.is_connected && self.is_streaming && self.connection_mode == ConnectionMode::Hardware;
            let rec_btn_text = if self.is_recording { self.text(UiText::StopRecording) } else { self.text(UiText::StartRecording) };
            let rec_btn_col = if self.is_recording { Color32::RED } else { if can_record { Color32::DARK_GRAY } else { Color32::from_rgb(30,30,30) } };
            if ui.add_enabled(can_record, egui::Button::new(egui::RichText::new(rec_btn_text).color(Color32::WHITE)).fill(rec_btn_col)).clicked() {
                if self.is_recording { self.tx_cmd.send(GuiCommand::StopRecording).unwrap(); } else { self.tx_cmd.send(GuiCommand::StartRecording(self.record_label.clone())).unwrap(); }
            }
            if self.is_recording { ui.label(egui::RichText::new(self.text(UiText::Recording)).color(Color32::RED).small()); }
            else if self.connection_mode == ConnectionMode::Simulation { ui.label(egui::RichText::new(self.text(UiText::HardwareRequired)).color(Color32::YELLOW).small()); }
            ui.add_space(10.0);
            egui::ScrollArea::vertical().max_height(100.0).show(ui, |ui| { for m in &self.log_messages { ui.monospace(m); } });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            if self.selected_tab == "Monitor" {
                ui.horizontal(|ui| {
                    if self.is_connected {
                        if self.connection_mode == ConnectionMode::Simulation && self.is_streaming { ui.label(egui::RichText::new(self.text(UiText::KeyHint)).strong().color(Color32::YELLOW)); }
                    } else { ui.label(self.text(UiText::ConnectFirst)); }
                });
                Plot::new("main_plot").view_aspect(2.0).include_y(0.0).auto_bounds_x().allow_drag(!self.follow_latest).allow_zoom(!self.follow_latest).show(ui, |plot_ui| {
                    let colors = [
                        Color32::from_rgb(0, 255, 255), Color32::from_rgb(0, 255, 255), Color32::from_rgb(0, 200, 255), Color32::from_rgb(0, 170, 255),
                        Color32::YELLOW, Color32::from_rgb(240, 220, 80), Color32::from_rgb(230, 200, 0), Color32::from_rgb(210, 170, 0),
                        Color32::from_rgb(255, 0, 255), Color32::from_rgb(230, 0, 200), Color32::from_rgb(200, 0, 170), Color32::from_rgb(170, 0, 140),
                        Color32::from_rgb(0, 255, 140), Color32::from_rgb(0, 230, 120), Color32::from_rgb(0, 200, 100), Color32::from_rgb(0, 170, 80),
                    ];
                    let mut min_y = f64::INFINITY; let mut max_y = f64::NEG_INFINITY;
                    for (i, buf) in self.wave_buffers.iter().enumerate() {
                        if !buf.is_empty() {
                            let col = colors.get(i).unwrap_or(&Color32::WHITE);
                            plot_ui.line(Line::new(PlotPoints::new(buf.clone())).name(format!("Ch{}", i + 1)).color(*col));
                            if let Some((min, max)) = buf.iter().fold(None, |acc, p| match acc { None => Some((p[1], p[1])), Some((mi, ma)) => Some((mi.min(p[1]), ma.max(p[1]))), }) { min_y = min_y.min(min); max_y = max_y.max(max); }
                        }
                    }
                    if self.follow_latest && self.time > 0.0 {
                        let x_max = self.time;
                        let x_min = (x_max - self.view_seconds).max(0.0);
                        let margin = self.vertical_spacing * 0.3;
                        let y_min = if min_y.is_finite() { min_y - margin } else { -1000.0 };
                        let y_max = if max_y.is_finite() { max_y + margin } else { self.vertical_spacing * self.wave_buffers.len() as f64 };
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max([x_min, y_min], [x_max + 0.5, y_max]));
                    }
                });
                ui.label(format!("{} {:.1}", self.text(UiText::Threshold), self.trigger_threshold));
            } else {
                ui.heading(self.text(UiText::Calibration));
                if self.is_connected && self.is_streaming {
                    if ui.button("1. Record Relax (3s)").clicked() { self.calib_rest_max = 0.0; self.is_calibrating = true; self.calib_timer = 3.0; self.tx_cmd.send(GuiCommand::StartCalibration(false)).unwrap(); }
                    if ui.button("2. Record Action (3s)").clicked() { self.calib_act_max = 0.0; self.is_calibrating = true; self.calib_timer = 3.0; self.tx_cmd.send(GuiCommand::StartCalibration(true)).unwrap(); }
                    if self.is_calibrating { ui.label("Recording..."); }
                    ui.label(format!("Threshold: {:.1}", self.trigger_threshold));
                } else { ui.label("Connect & Stream first."); }
            }
        });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Language { English, Chinese }

impl Language {
    fn text(&self, key: UiText) -> &'static str {
        match (self, key) {
            (Language::English, UiText::Title) => "QNMDsol demo v0.1",
            (Language::English, UiText::Subtitle) => "Neural Interface",
            (Language::English, UiText::Sim) => "SIM",
            (Language::English, UiText::Real) => "REAL",
            (Language::English, UiText::Connect) => "CONNECT",
            (Language::English, UiText::Disconnect) => "DISCONNECT",
            (Language::English, UiText::StartStream) => "START STREAM",
            (Language::English, UiText::StopStream) => "STOP STREAM",
            (Language::English, UiText::ResetView) => "🔄 RESET VIEW",
            (Language::English, UiText::Controller) => "XBOX CONTROLLER VISUALIZER",
            (Language::English, UiText::Data) => "AI DATA COLLECTION",
            (Language::English, UiText::Recording) => "Recording...",
            (Language::English, UiText::HardwareRequired) => "Hardware required",
            (Language::English, UiText::KeyHint) => "Try Keys: WASD / Space / ZXC / QEUO / Arrows",
            (Language::English, UiText::ConnectFirst) => "Connect first.",
            (Language::English, UiText::Threshold) => "Trigger Threshold:",
            (Language::English, UiText::Calibration) => "Calibration",
            (Language::English, UiText::FollowOn) => "📡 Follow Latest: ON",
            (Language::English, UiText::FollowOff) => "📡 Follow Latest: OFF",
            (Language::English, UiText::Ready) => "QNMDsol Demo v0.1 Ready.",
            (Language::English, UiText::LanguagePrompt) => "Choose your language",
            (Language::English, UiText::StartSubtitle) => "Pick a language to start",
            (Language::English, UiText::StartRecording) => "🔴 RECORD",
            (Language::English, UiText::StopRecording) => "⏹ STOP",

            (Language::Chinese, UiText::Title) => "QNMDsol 演示 v0.1",
            (Language::Chinese, UiText::Subtitle) => "神经接口控制",
            (Language::Chinese, UiText::Sim) => "模拟模式",
            (Language::Chinese, UiText::Real) => "实机模式",
            (Language::Chinese, UiText::Connect) => "连接",
            (Language::Chinese, UiText::Disconnect) => "断开",
            (Language::Chinese, UiText::StartStream) => "开始采集",
            (Language::Chinese, UiText::StopStream) => "停止采集",
            (Language::Chinese, UiText::ResetView) => "🔄 重置视图",
            (Language::Chinese, UiText::Controller) => "XBOX 手柄可视化",
            (Language::Chinese, UiText::Data) => "AI 数据采集",
            (Language::Chinese, UiText::Recording) => "录制中...",
            (Language::Chinese, UiText::HardwareRequired) => "需要连接硬件设备",
            (Language::Chinese, UiText::KeyHint) => "模拟: WASD移动 / Space跳跃 / ZXC攻击 / QEUO肩键 / 方向键",
            (Language::Chinese, UiText::ConnectFirst) => "请先连接设备。",
            (Language::Chinese, UiText::Threshold) => "触发阈值：",
            (Language::Chinese, UiText::Calibration) => "校准",
            (Language::Chinese, UiText::FollowOn) => "📡 追踪最新波形：开",
            (Language::Chinese, UiText::FollowOff) => "📡 追踪最新波形：关",
            (Language::Chinese, UiText::Ready) => "QNMDsol 演示 v0.1 已就绪。",
            (Language::Chinese, UiText::LanguagePrompt) => "选择你的界面语言",
            (Language::Chinese, UiText::StartSubtitle) => "点击语言开始体验",
            (Language::Chinese, UiText::StartRecording) => "🔴 开始录制",
            (Language::Chinese, UiText::StopRecording) => "⏹ 停止录制",
        }
    }

    fn default_record_label(&self) -> &'static str {
        match self {
            Language::English => "Attack",
            Language::Chinese => "攻击",
        }
    }
}

#[derive(Clone, Copy)]
enum UiText {
    Title, Subtitle, Sim, Real, Connect, Disconnect, StartStream, StopStream, ResetView,
    Controller, Data, Recording, HardwareRequired, KeyHint, ConnectFirst, Threshold,
    Calibration, FollowOn, FollowOff, Ready, LanguagePrompt, StartSubtitle, StartRecording, StopRecording,
}