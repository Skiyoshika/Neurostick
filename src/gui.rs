// src/gui.rs
use crate::drivers::{
    render_spectrum_png, render_waveform_png, FrequencySpectrum, PlotStyle, SpectrumBuilder,
    TimeSeriesFrame,
};
use crate::assets::APP_ICON_PNG;
use crate::engine;
use crate::types::*;
use crate::visualizer;
use eframe::egui;
use egui::{Color32, ColorImage, TextureHandle, TextureOptions, Vec2};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints};
use std::collections::VecDeque;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::{fs, io::Write, path::PathBuf, time::Instant, time::SystemTime};
// 引入串口库
use serialport;

pub struct QnmdSolApp {
    is_connected: bool,
    is_vjoy_active: bool,
    is_streaming: bool,
    is_recording: bool,
    connection_mode: ConnectionMode,
    follow_latest: bool,
    time: f64,
    wave_buffers: Vec<VecDeque<[f64; 2]>>,
    last_frame: Option<TimeSeriesFrame>,
    last_spectrum: Option<FrequencySpectrum>,
    wave_png: Option<Vec<u8>>,
    spectrum_png: Option<Vec<u8>>,
    fft_size: usize,
    view_seconds: f64,
    display_gain: f64,
    vertical_spacing: f64,
    gamepad_target: GamepadState,
    gamepad_visual: GamepadState,
    calib_rest_max: f64,
    calib_act_max: f64,
    is_calibrating: bool,
    calib_timer: f32,
    trigger_threshold: f64,
    record_label: String,
    language: Language,
    has_started: bool,
    selected_tab: ViewTab,
    log_messages: Vec<String>,
    rx: Receiver<BciMessage>,
    tx_cmd: Sender<GuiCommand>,
    theme_dark: bool,
    icon_tex: Option<TextureHandle>,
    progress_label: Option<String>,
    progress_value: f32,
    signal_sensitivity: f64,
    smooth_alpha: f64,
    wave_smooth_state: Vec<f64>,
    wave_window_seconds: f64,
    stream_start: Option<Instant>,

    // === 新增：端口管理 ===
    available_ports: Vec<String>,
    selected_port: String,
}

impl Default for QnmdSolApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let (tx_cmd, rx_cmd) = channel();
        engine::spawn_thread(tx, rx_cmd);
        let buffers = vec![VecDeque::new(); 16];

        // === 自动扫描端口 ===
        let mut ports = Vec::new();
        if let Ok(available) = serialport::available_ports() {
            for p in available {
                ports.push(p.port_name);
            }
        }
        let default_port = if !ports.is_empty() {
            ports[0].clone()
        } else {
            "COM3".to_string()
        };

        let language = QnmdSolApp::load_language_from_disk().unwrap_or(Language::English);

        Self {
            is_connected: false,
            is_vjoy_active: false,
            is_streaming: false,
            is_recording: false,
            connection_mode: ConnectionMode::Hardware,
            follow_latest: true,
            time: 0.0,
            wave_buffers: buffers,
            last_frame: None,
            last_spectrum: None,
            wave_png: None,
            spectrum_png: None,
            fft_size: 256,
            view_seconds: 30.0,
            display_gain: 0.35,
            vertical_spacing: 420.0,
            gamepad_target: GamepadState::default(),
            gamepad_visual: GamepadState::default(),
            calib_rest_max: 0.0,
            calib_act_max: 0.0,
            is_calibrating: false,
            calib_timer: 0.0,
            selected_tab: ViewTab::Waveform,
            log_messages: vec![],
            trigger_threshold: 200.0,
            record_label: language.default_record_label().to_owned(),
            language,
            has_started: false,
            theme_dark: false,
            icon_tex: None,
            progress_label: None,
            progress_value: 0.0,
            signal_sensitivity: 0.35,
            smooth_alpha: 0.18,
            wave_smooth_state: Vec::new(),
            wave_window_seconds: 30.0,
            stream_start: None,
            rx,
            tx_cmd,
            // === 初始化端口字段 ===
            available_ports: ports,
            selected_port: default_port,
        }
    }
}

impl QnmdSolApp {
    fn apply_theme(&self, ctx: &egui::Context) {
        if self.theme_dark {
            let visuals = egui::Visuals::dark();
            ctx.set_visuals(visuals);
        } else {
            let mut visuals = egui::Visuals::light();
            visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(245, 245, 247);
            visuals.window_fill = Color32::from_rgb(250, 250, 252);
            visuals.override_text_color = Some(Color32::from_rgb(30, 30, 35));
            ctx.set_visuals(visuals);
        }
    }

    fn generate_report(&self) -> std::io::Result<String> {
        let dir = PathBuf::from("reports");
        fs::create_dir_all(&dir)?;
        let ts = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        let path = dir.join(format!("report_{ts}.log"));
        let mut f = fs::File::create(&path)?;
        writeln!(f, "QNMDsol Report")?;
        writeln!(f, "Timestamp: {ts}")?;
        writeln!(f, "Mode: {:?}", self.connection_mode)?;
        writeln!(f, "Connected: {}", self.is_connected)?;
        writeln!(f, "Streaming: {}", self.is_streaming)?;
        writeln!(f, "Recording: {}", self.is_recording)?;
        writeln!(f, "Selected Port: {}", self.selected_port)?;
        writeln!(f, "Last Logs:")?;
        for msg in &self.log_messages {
            writeln!(f, "  {msg}")?;
        }
        Ok(path.to_string_lossy().to_string())
    }

    fn text(&self, key: UiText) -> &'static str {
        self.language.text(key)
    }
    fn reset_localized_defaults(&mut self) {
        self.log_messages.clear();
        self.log(self.text(UiText::Ready));
        self.record_label = self.language.default_record_label().to_owned();
    }
    fn log(&mut self, msg: &str) {
        self.log_messages.push(format!("> {}", msg));
        if self.log_messages.len() > 8 {
            self.log_messages.remove(0);
        }
    }
    fn lerp(current: f32, target: f32, speed: f32) -> f32 {
        current + (target - current) * speed
    }

    fn language_store_path() -> PathBuf {
        PathBuf::from("data/last_language.txt")
    }

    fn load_language_from_disk() -> Option<Language> {
        let path = Self::language_store_path();
        if let Ok(raw) = fs::read_to_string(path) {
            match raw.trim() {
                "zh" | "cn" => Some(Language::Chinese),
                "en" => Some(Language::English),
                _ => None,
            }
        } else {
            None
        }
    }

    fn persist_language(&self) {
        let path = Self::language_store_path();
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        let code = match self.language {
            Language::English => "en",
            Language::Chinese => "zh",
        };
        let _ = fs::write(path, code);
    }

    fn set_language(&mut self, lang: Language) {
        if self.language != lang {
            self.language = lang;
            self.record_label = self.language.default_record_label().to_owned();
            self.persist_language();
        }
    }

    fn ensure_icon_texture(&mut self, ctx: &egui::Context) {
        if self.icon_tex.is_some() {
            return;
        }
        if let Ok(img) = image::load_from_memory(APP_ICON_PNG) {
            let rgba = img.to_rgba8();
            let size = [rgba.width() as usize, rgba.height() as usize];
            let color_image = ColorImage::from_rgba_unmultiplied(size, rgba.as_raw());
            self.icon_tex = Some(
                ctx.load_texture("qnmdsol_icon", color_image, TextureOptions::LINEAR),
            );
        }
    }

    fn set_progress(&mut self, label: impl Into<String>, value: f32) {
        self.progress_label = Some(label.into());
        self.progress_value = value.clamp(0.0, 1.0);
    }

    fn clear_progress(&mut self) {
        self.progress_label = None;
        self.progress_value = 0.0;
    }

    // 刷新端口列表
    fn refresh_ports(&mut self) {
        self.available_ports.clear();
        if let Ok(available) = serialport::available_ports() {
            for p in available {
                self.available_ports.push(p.port_name);
            }
        }
        if !self.available_ports.is_empty() && !self.available_ports.contains(&self.selected_port) {
            self.selected_port = self.available_ports[0].clone();
        }
        self.log(&format!(
            "{} {:?}",
            self.text(UiText::PortsScanned),
            self.available_ports
        ));
    }

    fn show_waveform(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if self.is_connected {
                if self.connection_mode == ConnectionMode::Simulation && self.is_streaming {
                    ui.label(
                        egui::RichText::new(self.text(UiText::KeyHint))
                            .strong()
                            .color(if self.theme_dark {
                                Color32::YELLOW
                            } else {
                                Color32::from_rgb(20, 60, 180)
                            }),
                    );
                }
            } else {
                ui.label(self.text(UiText::ConnectFirst));
            }
        });

        // 固定更小的可视区域，确保下方状态/手柄始终可见
        let plot_height = 220.0;

        ui.allocate_ui(Vec2::new(ui.available_width(), plot_height), |ui| {
            ui.horizontal(|ui| {
                ui.label(self.text(UiText::Sensitivity));
                ui.add(egui::Slider::new(&mut self.signal_sensitivity, 0.05..=2.0).logarithmic(true));
                ui.label(self.text(UiText::Smoothness));
                ui.add(egui::Slider::new(&mut self.smooth_alpha, 0.0..=0.8));
            });
            ui.horizontal(|ui| {
                ui.label(self.text(UiText::Window));
                for (label, seconds) in [
                    (self.text(UiText::Window30), 30.0),
                    (self.text(UiText::Window60), 60.0),
                ] {
                    let selected = (self.wave_window_seconds - seconds).abs() < f64::EPSILON;
                    if ui.selectable_label(selected, label).clicked() {
                        self.wave_window_seconds = seconds;
                        self.view_seconds = seconds;
                        self.time = 0.0;
                        for buf in &mut self.wave_buffers {
                            buf.clear();
                        }
                        self.wave_smooth_state.clear();
                    }
                }
            });

            Plot::new("main_plot")
                .view_aspect(2.0)
                .include_y(0.0)
                .auto_bounds_y()
                .allow_drag(false)
                .allow_zoom(false)
                .show(ui, |plot_ui| {
                    let colors = if self.theme_dark {
                        [
                            Color32::from_rgb(0, 255, 255),
                            Color32::from_rgb(0, 255, 255),
                            Color32::from_rgb(0, 200, 255),
                            Color32::from_rgb(0, 170, 255),
                            Color32::from_rgb(140, 140, 30),
                            Color32::from_rgb(150, 130, 30),
                            Color32::from_rgb(140, 120, 30),
                            Color32::from_rgb(130, 110, 25),
                            Color32::from_rgb(255, 0, 255),
                            Color32::from_rgb(230, 0, 200),
                            Color32::from_rgb(200, 0, 170),
                            Color32::from_rgb(170, 0, 140),
                            Color32::from_rgb(0, 255, 140),
                            Color32::from_rgb(0, 230, 120),
                            Color32::from_rgb(0, 200, 100),
                            Color32::from_rgb(0, 170, 80),
                        ]
                    } else {
                        [
                            Color32::from_rgb(0, 120, 160),
                            Color32::from_rgb(0, 130, 170),
                            Color32::from_rgb(0, 100, 160),
                            Color32::from_rgb(0, 90, 150),
                            Color32::from_rgb(90, 90, 0),
                            Color32::from_rgb(120, 110, 0),
                            Color32::from_rgb(140, 120, 10),
                            Color32::from_rgb(150, 100, 0),
                            Color32::from_rgb(120, 0, 120),
                            Color32::from_rgb(110, 0, 110),
                            Color32::from_rgb(100, 0, 100),
                            Color32::from_rgb(80, 0, 90),
                            Color32::from_rgb(0, 140, 90),
                            Color32::from_rgb(0, 130, 80),
                            Color32::from_rgb(0, 120, 70),
                            Color32::from_rgb(0, 110, 60),
                        ]
                    };
                    let mut min_y = f64::INFINITY;
                    let mut max_y = f64::NEG_INFINITY;
                    for (i, buf) in self.wave_buffers.iter().enumerate() {
                        if !buf.is_empty() {
                            let col = colors.get(i).unwrap_or(&Color32::WHITE);
                            let mut points: Vec<[f64; 2]> = Vec::with_capacity(buf.len() + 8);
                            let mut last_phase: Option<f64> = None;
                            for p in buf.iter() {
                                let phase = p[0] % self.wave_window_seconds;
                                if let Some(lp) = last_phase {
                                    if phase < lp {
                                        points.push([f64::NAN, f64::NAN]); // break line on wrap
                                    }
                                }
                                points.push([phase, p[1]]);
                                last_phase = Some(phase);
                            }
                            plot_ui.line(
                                Line::new(PlotPoints::new(points))
                                    .name(format!("Ch{}", i + 1))
                                    .color(*col),
                            );
                            if let Some((min, max)) = buf.iter().fold(None, |acc, p| match acc {
                                None => Some((p[1], p[1])),
                                Some((mi, ma)) => Some((mi.min(p[1]), ma.max(p[1]))),
                            }) {
                                min_y = min_y.min(min);
                                max_y = max_y.max(max);
                            }
                        }
                    }
                    if self.follow_latest && self.time > 0.0 {
                        let x_min = 0.0;
                        let x_max = self.wave_window_seconds;
                        let margin = if max_y > min_y {
                            (max_y - min_y) * 0.1
                        } else {
                            10.0
                        };
                        let y_min = if min_y.is_finite() {
                            min_y - margin
                        } else {
                            -1000.0
                        };
                        let y_max = if max_y.is_finite() {
                            max_y + margin
                        } else {
                            1000.0
                        };
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [x_min, y_min],
                            [x_max + 0.5, y_max],
                        ));
                    }
                });
        });
        ui.label(format!(
            "{} {:.1}",
            self.text(UiText::Threshold),
            self.trigger_threshold
        ));
    }

    fn show_spectrum(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label(self.text(UiText::FftSize));
            let choices = [32, 64, 128, 256, 512, 1024];
            for sz in choices.iter() {
                if ui
                    .selectable_value(&mut self.fft_size, *sz, format!("{sz}"))
                    .clicked()
                {
                    if let Some(frame) = self.last_frame.clone() {
                        let builder = SpectrumBuilder::with_size(*sz);
                        self.last_spectrum = Some(builder.compute(&frame));
                    }
                }
            }
            if ui.button(self.text(UiText::Update)).clicked() {
                if let Some(frame) = self.last_frame.clone() {
                    let builder = SpectrumBuilder::with_size(self.fft_size);
                    self.last_spectrum = Some(builder.compute(&frame));
                }
            }
        });
        if let Some(spec) = self.last_spectrum.as_ref() {
            Plot::new("spectrum_plot")
                .view_aspect(2.0)
                .allow_drag(true)
                .allow_zoom(true)
                .show(ui, |plot_ui| {
                    for (idx, mags) in spec.magnitudes.iter().enumerate() {
                        let points: PlotPoints = spec
                            .frequencies_hz
                            .iter()
                            .zip(mags.iter())
                            .map(|(f, m)| [*f as f64, *m as f64])
                            .collect();
                        plot_ui.line(
                            Line::new(points)
                                .name(
                                    spec.channel_labels
                                        .get(idx)
                                        .cloned()
                                        .unwrap_or_else(|| format!("Ch{}", idx + 1)),
                                )
                                .color(Color32::from_rgb(30 + (idx as u8 * 13), 200, 120)),
                        );
                    }
                });
        } else {
            ui.label(self.text(UiText::NoSpectrumYet));
        }
    }

    fn show_png(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            if ui.button(self.text(UiText::GenerateWaveformPng)).clicked() {
                if let Some(frame) = self.last_frame.clone() {
                    match render_waveform_png(&frame, PlotStyle::default()) {
                        Ok(png) => self.wave_png = Some(png),
                        Err(e) => {
                            let msg = match self.language {
                                Language::English => format!("Wave PNG failed: {e}"),
                                Language::Chinese => format!("波形导出失败: {e}"),
                            };
                            self.log(&msg);
                        }
                    }
                } else {
                    let msg = match self.language {
                        Language::English => "No frame to render.".to_owned(),
                        Language::Chinese => "没有可绘制的帧。".to_owned(),
                    };
                    self.log(&msg);
                }
            }
            if ui.button(self.text(UiText::GenerateSpectrumPng)).clicked() {
                let spec = if let Some(spec) = self.last_spectrum.clone() {
                    Some(spec)
                } else if let Some(frame) = self.last_frame.clone() {
                    let builder = SpectrumBuilder::with_size(self.fft_size);
                    Some(builder.compute(&frame))
                } else {
                    None
                };
                if let Some(spec) = spec {
                    match render_spectrum_png(&spec, PlotStyle::default()) {
                        Ok(png) => {
                            self.spectrum_png = Some(png);
                            self.last_spectrum = Some(spec);
                        }
                        Err(e) => {
                            let msg = match self.language {
                                Language::English => format!("Spectrum PNG failed: {e}"),
                                Language::Chinese => format!("频谱导出失败: {e}"),
                            };
                            self.log(&msg);
                        }
                    }
                } else {
                    let msg = match self.language {
                        Language::English => "No spectrum to render.".to_owned(),
                        Language::Chinese => "没有可绘制的频谱。".to_owned(),
                    };
                    self.log(&msg);
                }
            }
        });

        ui.separator();
        if let Some(png) = &self.wave_png {
            ui.label(self.text(UiText::WaveformPngLabel));
            ui.add(egui::Image::from_bytes("wave_png", png.clone()).max_width(600.0));
        }
        if let Some(png) = &self.spectrum_png {
            ui.label(self.text(UiText::SpectrumPngLabel));
            ui.add(egui::Image::from_bytes("spectrum_png", png.clone()).max_width(600.0));
        }
    }

    fn show_calibration(&mut self, ui: &mut egui::Ui) {
        ui.heading(self.text(UiText::Calibration));
        if self.is_connected && self.is_streaming {
            if ui.button(self.text(UiText::RecordRelax)).clicked() {
                self.calib_rest_max = 0.0;
                self.is_calibrating = true;
                self.calib_timer = 3.0;
                self.set_progress(self.text(UiText::Calibration), 0.0);
                self.tx_cmd
                    .send(GuiCommand::StartCalibration(false))
                    .unwrap();
            }
            if ui.button(self.text(UiText::RecordAction)).clicked() {
                self.calib_act_max = 0.0;
                self.is_calibrating = true;
                self.calib_timer = 3.0;
                self.set_progress(self.text(UiText::Calibration), 0.0);
                self.tx_cmd
                    .send(GuiCommand::StartCalibration(true))
                    .unwrap();
            }
            if self.is_calibrating {
                ui.label(self.text(UiText::Recording));
            }
            ui.label(format!("{} {:.1}", self.text(UiText::Threshold), self.trigger_threshold));
        } else {
            ui.label(self.text(UiText::ConnectStreamFirst));
        }
    }

    fn show_start_screen(&mut self, ctx: &egui::Context) {
        let mut visuals = egui::Visuals::light();
        visuals.widgets.noninteractive.bg_fill = Color32::from_rgb(242, 245, 250);
        visuals.window_fill = Color32::from_rgb(246, 248, 252);
        let window_fill = visuals.window_fill;
        ctx.set_visuals(visuals);

        let accent = Color32::from_rgb(40, 90, 200);
        let accent_soft = Color32::from_rgb(230, 236, 250);

        egui::CentralPanel::default()
            .frame(egui::Frame::none().fill(window_fill))
            .show(ctx, |ui| {
                ui.vertical_centered(|ui| {
                    ui.add_space(70.0);
                    if let Some(tex) = &self.icon_tex {
                        ui.add(
                            egui::Image::new(tex)
                                .fit_to_exact_size(Vec2::new(64.0, 64.0)),
                        );
                        ui.add_space(12.0);
                    }
                    ui.heading(
                        egui::RichText::new(self.text(UiText::StartHeading))
                            .size(36.0)
                            .strong()
                            .color(Color32::from_rgb(25, 30, 40)),
                    );
                    ui.add_space(12.0);
                    ui.label(
                        egui::RichText::new(self.text(UiText::StartSubtitle))
                            .size(18.0)
                            .color(Color32::from_rgb(90, 100, 120)),
                    );
                    ui.add_space(24.0);
                    egui::Frame::none()
                        .fill(accent_soft)
                        .stroke(egui::Stroke::new(1.2, accent))
                        .rounding(egui::Rounding::same(20.0))
                        .inner_margin(egui::style::Margin::symmetric(32.0, 28.0))
                        .show(ui, |ui| {
                            ui.vertical_centered(|ui| {
                                ui.label(
                                    egui::RichText::new(self.text(UiText::LanguagePrompt))
                                        .size(16.0)
                                        .color(Color32::from_rgb(70, 80, 100)),
                                );
                                ui.add_space(18.0);
                                ui.horizontal(|ui| {
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                egui::RichText::new("中文")
                                                    .size(17.0)
                                                    .strong()
                                                    .color(Color32::WHITE),
                                            )
                                            .min_size(Vec2::new(150.0, 46.0))
                                            .fill(accent)
                                            .rounding(egui::Rounding::same(14.0)),
                                        )
                                        .clicked()
                                    {
                                        self.set_language(Language::Chinese);
                                        self.has_started = true;
                                        self.reset_localized_defaults();
                                    }
                                    ui.add_space(18.0);
                                    if ui
                                        .add(
                                            egui::Button::new(
                                                egui::RichText::new("English")
                                                    .size(17.0)
                                                    .strong()
                                                    .color(accent),
                                            )
                                            .min_size(Vec2::new(150.0, 46.0))
                                            .rounding(egui::Rounding::same(14.0)),
                                        )
                                        .clicked()
                                    {
                                        self.set_language(Language::English);
                                        self.has_started = true;
                                        self.reset_localized_defaults();
                                    }
                                });
                            });
                        });
                });
            });
    }
}

impl eframe::App for QnmdSolApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.ensure_icon_texture(ctx);

        if !self.has_started {
            self.show_start_screen(ctx);
            return;
        }

        // 主题应用（苹果白默认，可切换黑夜）
        self.apply_theme(ctx);

        // 键盘输入 (Sim Mode) - 保持不变
        if self.connection_mode == ConnectionMode::Simulation {
            let mut input = SimInputIntent::default();
            if ctx.input(|i| i.key_down(egui::Key::W)) {
                input.w = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::S)) {
                input.s = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::A)) {
                input.a = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::D)) {
                input.d = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::Space)) {
                input.space = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::Z)) {
                input.key_z = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::X)) {
                input.key_x = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::C)) {
                input.key_c = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::I)) {
                input.up = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::K)) {
                input.down = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::J)) {
                input.left = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::L)) {
                input.right = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::Q)) {
                input.q = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::E)) {
                input.e = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::U)) {
                input.u = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::O)) {
                input.o = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::ArrowUp)) {
                input.arrow_up = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::ArrowDown)) {
                input.arrow_down = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::ArrowLeft)) {
                input.arrow_left = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::ArrowRight)) {
                input.arrow_right = true;
            }
            self.tx_cmd.send(GuiCommand::UpdateSimInput(input)).ok();
        }

        // 消息处理
        let mut msg_count = 0;
        while let Ok(msg) = self.rx.try_recv() {
            msg_count += 1;
            if msg_count > 20 {
                match msg {
                    BciMessage::GamepadUpdate(gp) => self.gamepad_target = gp,
                    _ => continue,
                }
            } else {
                match msg {
                    BciMessage::Log(s) => self.log(&s),
                    BciMessage::Status(b) => self.is_connected = b,
                    BciMessage::VJoyStatus(b) => self.is_vjoy_active = b,
                    BciMessage::GamepadUpdate(gp) => self.gamepad_target = gp,
                    BciMessage::RecordingStatus(b) => self.is_recording = b,
                    BciMessage::Spectrum(spec) => {
                        self.last_spectrum = Some(spec);
                    }
                    BciMessage::DataFrame(frame) => {
                        let sr = frame.sample_rate_hz as f64;
                        if sr <= 0.0 {
                            continue;
                        }

                        if self.wave_buffers.len() != frame.samples.len() {
                            self.wave_buffers = vec![VecDeque::new(); frame.samples.len()];
                        }
                        if self.wave_smooth_state.len() != frame.samples.len() {
                            self.wave_smooth_state = vec![0.0; frame.samples.len()];
                        }

                        self.last_frame = Some(frame.clone());

                        let samples_per_channel =
                            frame.samples.first().map(|c| c.len()).unwrap_or(0);
        for (i, channel) in frame.samples.iter().enumerate() {
            let offset = i as f64 * self.vertical_spacing;
            for (idx, sample) in channel.iter().enumerate() {
                let t = self.time + idx as f64 / sr;
                let alpha = self.smooth_alpha.clamp(0.0, 1.0);
                let scaled = *sample as f64 * self.display_gain * self.signal_sensitivity;
                let prev = self.wave_smooth_state[i];
                let smoothed = if alpha <= 0.0 {
                    scaled
                } else if alpha >= 1.0 {
                    scaled
                } else {
                    prev * (1.0 - alpha) + scaled * alpha
                };
                self.wave_smooth_state[i] = smoothed;
                self.wave_buffers[i]
                    .push_back([t, smoothed + offset]);
            }
        }
                        self.time += samples_per_channel as f64 / sr;

                        // 重建为实时扫屏：根据窗口长度映射到相位，避免累积历史导致闪烁
                        let now = Instant::now();
                        if self.stream_start.is_none() {
                            self.stream_start = Some(now);
                        }
                        let elapsed = self
                            .stream_start
                            .map(|t| now.saturating_duration_since(t).as_secs_f64())
                            .unwrap_or(0.0);
                        let frame_duration =
                            frame.samples.first().map(|c| c.len() as f64 / sr).unwrap_or(0.0);
                        let frame_start_time = (elapsed - frame_duration).max(0.0);
                        self.time = elapsed;

                        let window = self.wave_window_seconds;

                        for (i, channel) in frame.samples.iter().enumerate() {
                            let offset = i as f64 * self.vertical_spacing;
                            let buf = &mut self.wave_buffers[i];
                            buf.clear();
                            let mut last_phase: Option<f64> = None;
                            for (idx, sample) in channel.iter().enumerate() {
                                let abs_t = frame_start_time + idx as f64 / sr;
                                let phase = abs_t % window;
                                let scaled = *sample as f64 * self.display_gain * self.signal_sensitivity;
                                let prev = *self.wave_smooth_state.get(i).unwrap_or(&0.0);
                                let alpha = self.smooth_alpha.clamp(0.0, 1.0);
                                let smoothed = if alpha <= 0.0 {
                                    scaled
                                } else if alpha >= 1.0 {
                                    scaled
                                } else {
                                    prev * (1.0 - alpha) + scaled * alpha
                                };
                                if self.wave_smooth_state.len() > i {
                                    self.wave_smooth_state[i] = smoothed;
                                }

                                if let Some(lp) = last_phase {
                                    if phase < lp {
                                        buf.push_back([f64::NAN, f64::NAN]); // 断线，防止跨圈连线
                                    }
                                }
                                buf.push_back([phase, smoothed + offset]);
                                last_phase = Some(phase);
                            }
                        }
                    }
                    BciMessage::CalibrationResult(_, max) => {
                        self.is_calibrating = false;
                        self.clear_progress();
                        if self.calib_rest_max == 0.0 {
                            self.calib_rest_max = max;
                            let msg = match self.language {
                                Language::English => format!("Base: {:.1}", max),
                                Language::Chinese => format!("基线：{:.1}", max),
                            };
                            self.log(&msg);
                        } else {
                            self.calib_act_max = max;
                            let msg = match self.language {
                                Language::English => format!("Act: {:.1}", max),
                                Language::Chinese => format!("动作：{:.1}", max),
                            };
                            self.log(&msg);
                            let new = (self.calib_rest_max + self.calib_act_max) * 0.6;
                            self.trigger_threshold = new;
                            self.tx_cmd.send(GuiCommand::SetThreshold(new)).unwrap();
                            let thresh_msg = match self.language {
                                Language::English => format!("Threshold: {:.1}", new),
                                Language::Chinese => format!("阈值：{:.1}", new),
                            };
                            self.log(&thresh_msg);
                        }
                    }
                }
            }
        }

        // 动画插值
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

        if self.is_streaming {
            ctx.request_repaint();
        }
        if self.is_calibrating {
            self.calib_timer -= ctx.input(|i| i.stable_dt);
            let duration = 3.0;
            let progress = ((duration - self.calib_timer) / duration).clamp(0.0, 1.0);
            self.set_progress(self.text(UiText::Calibration), progress);
            if self.calib_timer < 0.0 {
                self.calib_timer = 0.0;
            }
            ctx.request_repaint();
        }

        egui::TopBottomPanel::top("topbar").show(ctx, |ui| {
            ui.vertical(|ui| {
                ui.horizontal_wrapped(|ui| {
                    let sim_label = self.text(UiText::Sim);
                    let real_label = self.text(UiText::Real);
                    if let Some(tex) = &self.icon_tex {
                        ui.add(
                            egui::Image::new(tex)
                                .fit_to_exact_size(Vec2::new(28.0, 28.0)),
                        );
                    }
                    ui.heading(self.text(UiText::Title));
                    ui.label(
                        egui::RichText::new(self.text(UiText::Subtitle))
                            .color(Color32::from_rgb(120, 120, 130)),
                    );
                    ui.separator();
                    ui.selectable_value(
                        &mut self.connection_mode,
                        ConnectionMode::Simulation,
                        sim_label,
                    );
                    ui.selectable_value(
                        &mut self.connection_mode,
                        ConnectionMode::Hardware,
                        real_label,
                    );
                    ui.separator();
                    if ui.button(self.text(UiText::ThemeLight)).clicked() {
                        self.theme_dark = false;
                        self.apply_theme(ctx);
                    }
                    if ui.button(self.text(UiText::ThemeDark)).clicked() {
                        self.theme_dark = true;
                        self.apply_theme(ctx);
                    }
                    ui.separator();
                    ui.label(self.text(UiText::LanguageSwitch));
                    let mut selected_language = self.language;
                    egui::ComboBox::from_id_source("language_switcher_top")
                        .selected_text(match self.language {
                            Language::English => "English",
                            Language::Chinese => "中文",
                        })
                        .show_ui(ui, |ui| {
                            ui.selectable_value(&mut selected_language, Language::English, "English");
                            ui.selectable_value(&mut selected_language, Language::Chinese, "中文");
                        });
                    if selected_language != self.language {
                        self.set_language(selected_language);
                    }
                    if ui.button(self.text(UiText::ReportFeedback)).clicked() {
                        match self.generate_report() {
                            Ok(path) => {
                                let msg = match self.language {
                                    Language::English => format!("Report saved: {path}"),
                                    Language::Chinese => format!("报告已保存: {path}"),
                                };
                                self.log(&msg);
                            }
                            Err(e) => {
                                let msg = match self.language {
                                    Language::English => format!("Report failed: {e}"),
                                    Language::Chinese => format!("报告生成失败: {e}"),
                                };
                                self.log(&msg);
                            }
                        }
                    }
                });

                ui.horizontal(|ui| {
                    if self.connection_mode == ConnectionMode::Hardware {
                        ui.label(self.text(UiText::PortLabel));
                        egui::ComboBox::from_id_source("port_selector_top")
                            .selected_text(&self.selected_port)
                            .show_ui(ui, |ui| {
                                for p in &self.available_ports {
                                    ui.selectable_value(&mut self.selected_port, p.clone(), p);
                                }
                            });
                        if ui.button(self.text(UiText::RefreshPorts)).clicked() {
                            self.refresh_ports();
                        }
                    }

                    let btn_txt = if self.is_connected {
                        self.text(UiText::Disconnect)
                    } else {
                        self.text(UiText::Connect)
                    };
                    if ui.button(btn_txt).clicked() {
                        if !self.is_connected {
                            self.tx_cmd
                                .send(GuiCommand::Connect(
                                    ConnectionMode::Hardware,
                                    self.selected_port.clone(),
                                ))
                                .unwrap();
                            self.connection_mode = ConnectionMode::Hardware;
                        } else {
                            self.tx_cmd.send(GuiCommand::Disconnect).unwrap();
                            self.stream_start = None;
                        }
                    }

                    if self.is_connected {
                        let stream_btn = if self.is_streaming {
                            self.text(UiText::StopStream)
                        } else {
                            self.text(UiText::StartStream)
                        };
                        if ui.button(stream_btn).clicked() {
                            if self.is_streaming {
                                self.tx_cmd.send(GuiCommand::StopStream).unwrap();
                                self.is_streaming = false;
                                self.stream_start = None;
                            } else {
                                self.tx_cmd.send(GuiCommand::StartStream).unwrap();
                                self.is_streaming = true;
                                self.stream_start = Some(Instant::now());
                            }
                        }
                        if ui.button(self.text(UiText::ResetView)).clicked() {
                            for buf in &mut self.wave_buffers {
                                buf.clear();
                            }
                            self.time = 0.0;
                        }
                        let follow_label = if self.follow_latest {
                            self.text(UiText::FollowOn)
                        } else {
                            self.text(UiText::FollowOff)
                        };
                        if ui.button(follow_label).clicked() {
                            self.follow_latest = !self.follow_latest;
                        }

                        if self.connection_mode == ConnectionMode::Simulation && self.is_streaming {
                            if ui.button(self.text(UiText::InjectArtifact)).clicked() {
                                self.tx_cmd.send(GuiCommand::InjectArtifact).unwrap();
                            }
                            ui.label(
                                egui::RichText::new(self.text(UiText::KeyHint))
                                    .small()
                                    .color(if self.theme_dark {
                                        Color32::YELLOW
                                    } else {
                                        Color32::from_rgb(20, 60, 180)
                                    }),
                            );
                        }
                    }
                });

                ui.separator();
                ui.horizontal(|ui| {
                    ui.label(self.text(UiText::Data));
                    ui.text_edit_singleline(&mut self.record_label);
                    let can_record = self.is_connected
                        && self.is_streaming
                        && self.connection_mode == ConnectionMode::Hardware;
                    let rec_btn_text = if self.is_recording {
                        self.text(UiText::StopRecording)
                    } else {
                        self.text(UiText::StartRecording)
                    };
                    let rec_btn_col = if self.is_recording {
                        Color32::RED
                    } else if can_record {
                        Color32::DARK_GRAY
                    } else {
                        Color32::from_rgb(30, 30, 30)
                    };
                    if ui
                        .add_enabled(
                            can_record,
                            egui::Button::new(
                                egui::RichText::new(rec_btn_text).color(Color32::WHITE),
                            )
                            .fill(rec_btn_col),
                        )
                        .clicked()
                    {
                        if self.is_recording {
                            self.tx_cmd.send(GuiCommand::StopRecording).unwrap();
                        } else {
                            self.tx_cmd
                                .send(GuiCommand::StartRecording(self.record_label.clone()))
                                .unwrap();
                        }
                    }

                    if self.is_connected && self.is_streaming {
                        if ui.button(self.text(UiText::RecordRelax)).clicked() {
                            self.calib_rest_max = 0.0;
                            self.is_calibrating = true;
                            self.calib_timer = 3.0;
                            self.tx_cmd
                                .send(GuiCommand::StartCalibration(false))
                                .unwrap();
                        }
                        if ui.button(self.text(UiText::RecordAction)).clicked() {
                            self.calib_act_max = 0.0;
                            self.is_calibrating = true;
                            self.calib_timer = 3.0;
                            self.tx_cmd
                                .send(GuiCommand::StartCalibration(true))
                                .unwrap();
                        }
                        ui.label(format!(
                            "{} {:.1}",
                            self.text(UiText::Threshold),
                            self.trigger_threshold
                        ));
                    } else if self.connection_mode == ConnectionMode::Simulation {
                        ui.label(
                            egui::RichText::new(self.text(UiText::HardwareRequired))
                                .small()
                                .color(if self.theme_dark {
                                    Color32::YELLOW
                                } else {
                                    Color32::from_rgb(20, 60, 180)
                                }),
                        );
                    }
                });
            });
        });

        egui::SidePanel::left("status_panel")
            .resizable(true)
            .min_width(260.0)
            .default_width(280.0)
            .show(ctx, |ui| {
                if let Some(label) = &self.progress_label {
                    ui.label(self.text(UiText::Loading));
                    ui.add(
                        egui::ProgressBar::new(self.progress_value)
                            .show_percentage()
                            .text(label.clone()),
                    );
                    ui.separator();
                }
                ui.label(self.text(UiText::Controller));
                visualizer::draw_xbox_controller(ui, &self.gamepad_visual);
                ui.separator();
                ui.label("Logs");
                egui::ScrollArea::vertical()
                    .auto_shrink([false; 2])
                    .show(ui, |ui| {
                        for m in &self.log_messages {
                            ui.monospace(m);
                        }
                    });
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                for (label, tab) in [
                    (self.text(UiText::TabWaveform), ViewTab::Waveform),
                    (self.text(UiText::TabSpectrum), ViewTab::Spectrum),
                    (self.text(UiText::TabPng), ViewTab::Png),
                    (self.text(UiText::TabCalibration), ViewTab::Calibration),
                ] {
                    let selected = self.selected_tab == tab;
                    if ui.selectable_label(selected, label).clicked() {
                        self.selected_tab = tab;
                    }
                }
            });
            ui.separator();

            match self.selected_tab {
                ViewTab::Waveform => self.show_waveform(ui),
                ViewTab::Spectrum => self.show_spectrum(ui),
                ViewTab::Png => self.show_png(ui),
                ViewTab::Calibration => self.show_calibration(ui),
            }
        });
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Language {
    English,
    Chinese,
}

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
            (Language::English, UiText::StartHeading) => "Welcome to QNMDsol",
            (Language::English, UiText::StartRecording) => "🔴 RECORD",
            (Language::English, UiText::StopRecording) => "⏹ STOP",
            (Language::English, UiText::FftSize) => "FFT Size:",
            (Language::English, UiText::Update) => "Update",
            (Language::English, UiText::GenerateWaveformPng) => "Generate Waveform PNG",
            (Language::English, UiText::GenerateSpectrumPng) => "Generate Spectrum PNG",
            (Language::English, UiText::WaveformPngLabel) => "Waveform PNG:",
            (Language::English, UiText::SpectrumPngLabel) => "Spectrum PNG:",
            (Language::English, UiText::NoSpectrumYet) => {
                "No spectrum yet. Start streaming to populate."
            }
            (Language::English, UiText::RecordRelax) => "1. Record Relax (3s)",
            (Language::English, UiText::RecordAction) => "2. Record Action (3s)",
            (Language::English, UiText::ConnectStreamFirst) => "Connect & Stream first.",
            (Language::English, UiText::Loading) => "Working...",
            (Language::English, UiText::Sensitivity) => "Sensitivity",
            (Language::English, UiText::Smoothness) => "Smoothing",
            (Language::English, UiText::Window) => "Window",
            (Language::English, UiText::Window30) => "30s",
            (Language::English, UiText::Window60) => "60s",
            (Language::English, UiText::TabWaveform) => "Waveform",
            (Language::English, UiText::TabSpectrum) => "Spectrum",
            (Language::English, UiText::TabPng) => "PNG Export",
            (Language::English, UiText::TabCalibration) => "Calibration",
            (Language::English, UiText::PortLabel) => "Port:",
            (Language::English, UiText::RefreshPorts) => "Refresh",
            (Language::English, UiText::PortsScanned) => "Ports scanned:",
            (Language::English, UiText::InjectArtifact) => "Inject Artifact",
            (Language::English, UiText::ReportFeedback) => "Report Feedback",
            (Language::English, UiText::ThemeLight) => "☀️",
            (Language::English, UiText::ThemeDark) => "🌙",
            (Language::English, UiText::LanguageSwitch) => "Language",

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
            (Language::Chinese, UiText::KeyHint) => {
                "模拟: WASD移动 / Space跳跃 / ZXC攻击 / QEUO肩键 / 方向键"
            }
            (Language::Chinese, UiText::ConnectFirst) => "请先连接设备。",
            (Language::Chinese, UiText::Threshold) => "触发阈值：",
            (Language::Chinese, UiText::Calibration) => "校准",
            (Language::Chinese, UiText::FollowOn) => "📡 追踪最新波形：开",
            (Language::Chinese, UiText::FollowOff) => "📡 追踪最新波形：关",
            (Language::Chinese, UiText::Ready) => "QNMDsol 演示 v0.1 已就绪。",
            (Language::Chinese, UiText::LanguagePrompt) => "选择你的界面语言",
            (Language::Chinese, UiText::StartSubtitle) => "点击语言开始体验",
            (Language::Chinese, UiText::StartHeading) => "欢迎来到 QNMDsol",
            (Language::Chinese, UiText::StartRecording) => "🔴 开始录制",
            (Language::Chinese, UiText::StopRecording) => "⏹ 停止录制",
            (Language::Chinese, UiText::FftSize) => "FFT 大小：",
            (Language::Chinese, UiText::Update) => "更新",
            (Language::Chinese, UiText::GenerateWaveformPng) => "导出波形 PNG",
            (Language::Chinese, UiText::GenerateSpectrumPng) => "导出频谱 PNG",
            (Language::Chinese, UiText::WaveformPngLabel) => "波形图：",
            (Language::Chinese, UiText::SpectrumPngLabel) => "频谱图：",
            (Language::Chinese, UiText::NoSpectrumYet) => "暂无频谱，请开始采集。",
            (Language::Chinese, UiText::RecordRelax) => "1. 记录放松状态（3秒）",
            (Language::Chinese, UiText::RecordAction) => "2. 记录动作状态（3秒）",
            (Language::Chinese, UiText::ConnectStreamFirst) => "请先连接设备并开始采集。",
            (Language::Chinese, UiText::Loading) => "处理中...",
            (Language::Chinese, UiText::Sensitivity) => "敏感度",
            (Language::Chinese, UiText::Smoothness) => "平滑度",
            (Language::Chinese, UiText::Window) => "窗口长度",
            (Language::Chinese, UiText::Window30) => "30秒",
            (Language::Chinese, UiText::Window60) => "60秒",
            (Language::Chinese, UiText::TabWaveform) => "波形",
            (Language::Chinese, UiText::TabSpectrum) => "频谱",
            (Language::Chinese, UiText::TabPng) => "导出 PNG",
            (Language::Chinese, UiText::TabCalibration) => "校准",
            (Language::Chinese, UiText::PortLabel) => "串口：",
            (Language::Chinese, UiText::RefreshPorts) => "刷新",
            (Language::Chinese, UiText::PortsScanned) => "已扫描端口：",
            (Language::Chinese, UiText::InjectArtifact) => "注入伪迹",
            (Language::Chinese, UiText::ReportFeedback) => "报告反馈",
            (Language::Chinese, UiText::ThemeLight) => "☀️",
            (Language::Chinese, UiText::ThemeDark) => "🌙",
            (Language::Chinese, UiText::LanguageSwitch) => "语言",
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
    Title,
    Subtitle,
    Sim,
    Real,
    Connect,
    Disconnect,
    StartStream,
    StopStream,
    ResetView,
    Controller,
    Data,
    Recording,
    HardwareRequired,
    KeyHint,
    ConnectFirst,
    Threshold,
    Calibration,
    FollowOn,
    FollowOff,
    Ready,
    LanguagePrompt,
    StartSubtitle,
    StartHeading,
    StartRecording,
    StopRecording,
    FftSize,
    Update,
    GenerateWaveformPng,
    GenerateSpectrumPng,
    WaveformPngLabel,
    SpectrumPngLabel,
    NoSpectrumYet,
    RecordRelax,
    RecordAction,
    ConnectStreamFirst,
    Loading,
    Sensitivity,
    Smoothness,
    Window,
    Window30,
    Window60,
    TabWaveform,
    TabSpectrum,
    TabPng,
    TabCalibration,
    PortLabel,
    RefreshPorts,
    PortsScanned,
    InjectArtifact,
    ReportFeedback,
    ThemeLight,
    ThemeDark,
    LanguageSwitch,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewTab {
    Waveform,
    Spectrum,
    Png,
    Calibration,
}
