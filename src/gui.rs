// src/gui.rs
use crate::assets::APP_ICON_PNG;
use crate::drivers::pipeline::make_batch;
use crate::drivers::{
    cyton_impedance_from_std, cyton_impedances_from_samples, ganglion_display_impedance_kohms,
    render_spectrum_png, render_waveform_png, FrequencySpectrum, ManualSource, PlotStyle,
    SignalPipeline, SignalSource, SpectrumBuilder, TimeSeriesFrame,
};
use crate::engine;
use crate::types::*;
use crate::visualizer;
use crate::waveform::{
    ChannelView, FilterKind, SamplePoint, TimeWindow, WaveformPipeline, WaveformView, YScale,
};
use eframe::egui;
use egui::{Color32, ColorImage, TextureHandle, TextureOptions, Vec2};
use egui_plot::{Line, Plot, PlotBounds, PlotPoints, Text};
use serde::Deserialize;
use std::sync::mpsc::{channel, Receiver, Sender};
use std::{fs, io::Write, path::PathBuf, time::Instant, time::SystemTime};
// 引入串口库
use serialport;

#[derive(Debug, Clone, Deserialize)]
struct BrainModel {
    version: Option<String>,
    n_channels: usize,
    csp_filters: Option<Vec<Vec<f64>>>,
    lda_coef: Option<Vec<Vec<f64>>>,
    lda_intercept: Option<Vec<f64>>,
    classes: Vec<String>,
}

#[derive(Debug, Clone)]
struct BrainModelStatus {
    path: String,
    loaded_at: SystemTime,
    info: BrainModel,
}

pub struct QnmdSolApp {
    is_connected: bool,
    is_vjoy_active: bool,
    is_streaming: bool,
    is_recording: bool,
    connection_mode: ConnectionMode,
    follow_latest: bool,
    waveform_pipeline: Option<WaveformPipeline>,
    waveform_view: Option<WaveformView>,
    waveform_sample_rate_hz: f32,
    waveform_clock: f32,
    waveform_last_len: usize,
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
    last_gamepad_update: Option<Instant>,
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
    wave_auto_scale: bool,
    wave_notch_50hz: bool,
    wave_fixed_range_uv: f32,
    wave_show_stats: bool,
    stream_start: Option<Instant>,
    total_samples_ingested: usize,
    last_data_at: Option<Instant>,
    resistance_values: Option<Vec<f32>>,
    resistance_labels: Vec<String>,
    resistance_window_seconds: Option<f32>,
    resistance_last_measured: Option<SystemTime>,
    impedance_highlight_idx: usize,
    impedance_last_cycle: Option<Instant>,
    // === 新增：端口管理 ===
    available_ports: Vec<String>,
    selected_port: String,
    // 控制面板开关与宽度
    control_panel_open: bool,
    control_panel_width: f32,
    // 模型状态
    model_path: String,
    model_status: Option<BrainModelStatus>,
    model_error: Option<String>,
    model_scores: Option<Vec<f32>>,
}
impl Default for QnmdSolApp {
    fn default() -> Self {
        let (tx, rx) = channel();
        let (tx_cmd, rx_cmd) = channel();
        engine::spawn_thread(tx, rx_cmd);
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
        let mut app = Self {
            is_connected: false,
            is_vjoy_active: false,
            is_streaming: false,
            is_recording: false,
            connection_mode: ConnectionMode::Hardware,
            follow_latest: true,
            waveform_pipeline: None,
            waveform_view: None,
            waveform_sample_rate_hz: 0.0,
            waveform_clock: 0.0,
            waveform_last_len: 0,
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
            last_gamepad_update: None,
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
            signal_sensitivity: 1.0,
            smooth_alpha: 0.18,
            wave_smooth_state: Vec::new(),
            wave_window_seconds: 30.0,
            wave_auto_scale: false,
            wave_notch_50hz: false,
            wave_fixed_range_uv: 200.0,
            wave_show_stats: true,
            stream_start: None,
            total_samples_ingested: 0,
            last_data_at: None,
            resistance_values: None,
            resistance_labels: Vec::new(),
            resistance_window_seconds: None,
            resistance_last_measured: None,
            impedance_highlight_idx: 0,
            impedance_last_cycle: None,
            rx,
            tx_cmd,
            // === 初始化端口字段 ===
            available_ports: ports,
            selected_port: default_port,
            control_panel_open: true,
            control_panel_width: 320.0,
            model_path: "brain_model.json".to_string(),
            model_status: None,
            model_error: None,
            model_scores: None,
        };
        app.autoload_model();
        app
    }
}
impl QnmdSolApp {
    fn impedance_status(value_ohms: f32, lang: Language) -> (Color32, &'static str) {
        let (c_good, c_ok, c_bad, c_railed) = (
            Color32::from_rgb(46, 204, 113),
            Color32::from_rgb(243, 156, 18),
            Color32::from_rgb(231, 76, 60),
            Color32::from_rgb(155, 89, 182),
        );
        if value_ohms.is_nan() || value_ohms.is_infinite() || value_ohms > 5_000_000.0 {
            return (
                c_railed,
                match lang {
                    Language::English => "Railed",
                    Language::Chinese => "未接触",
                },
            );
        }
        if value_ohms < 500_000.0 {
            return (
                c_good,
                match lang {
                    Language::English => "Good (<500k)",
                    Language::Chinese => "理想 (<500kΩ)",
                },
            );
        }
        if value_ohms < 2_500_000.0 {
            return (
                c_ok,
                match lang {
                    Language::English => "Acceptable (0.5-2.5M)",
                    Language::Chinese => "可用 (0.5-2.5MΩ)",
                },
            );
        }
        if value_ohms <= 5_000_000.0 {
            return (
                c_bad,
                match lang {
                    Language::English => "Poor (>2.5M)",
                    Language::Chinese => "不良 (>2.5MΩ)",
                },
            );
        }
        (c_railed, "Railed")
    }
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
        let (title, ts_label, mode_label, conn_label, stream_label, rec_label, port_label) =
            match self.language {
                Language::English => (
                    "QNMDsol Report",
                    "Timestamp",
                    "Mode",
                    "Connected",
                    "Streaming",
                    "Recording",
                    "Selected Port",
                ),
                Language::Chinese => (
                    "QNMDsol 报告",
                    "时间戳",
                    "模式",
                    "已连接",
                    "采集中",
                    "录制中",
                    "选择的串口",
                ),
            };
        let bool_text = |b: bool, lang: Language| -> String {
            match lang {
                Language::English => b.to_string(),
                Language::Chinese => {
                    if b {
                        "是".to_owned()
                    } else {
                        "否".to_owned()
                    }
                }
            }
        };
        let mode_text = match (self.language, self.connection_mode) {
            (Language::Chinese, ConnectionMode::Simulation) => "模拟",
            (Language::Chinese, ConnectionMode::Hardware) => "实机",
            (Language::English, ConnectionMode::Simulation) => "Simulation",
            (Language::English, ConnectionMode::Hardware) => "Hardware",
        };
        writeln!(f, "{title}")?;
        writeln!(f, "{ts_label}: {ts}")?;
        writeln!(f, "{mode_label}: {mode_text}")?;
        writeln!(
            f,
            "{conn_label}: {}",
            bool_text(self.is_connected, self.language)
        )?;
        writeln!(
            f,
            "{stream_label}: {}",
            bool_text(self.is_streaming, self.language)
        )?;
        writeln!(
            f,
            "{rec_label}: {}",
            bool_text(self.is_recording, self.language)
        )?;
        writeln!(f, "{port_label}: {}", self.selected_port)?;
        writeln!(f, "{}", self.text(UiText::ReportLogs))?;
        for msg in &self.log_messages {
            writeln!(f, "  {msg}")?;
        }
        Ok(path.to_string_lossy().to_string())
    }
    fn load_model_from_path(&mut self, path: &str) -> Result<(), String> {
        let trimmed = path.trim();
        if trimmed.is_empty() {
            self.model_status = None;
            self.model_error = Some("Empty model path".to_string());
            return Err("Empty model path".to_string());
        }
        let raw = fs::read_to_string(trimmed).map_err(|e| e.to_string())?;
        let info: BrainModel = serde_json::from_str(&raw).map_err(|e| e.to_string())?;
        let status = BrainModelStatus {
            path: trimmed.to_string(),
            loaded_at: SystemTime::now(),
            info,
        };
        let msg = format!(
            "Model loaded: {} ({} classes, {} channels)",
            status.path,
            status.info.classes.len(),
            status.info.n_channels
        );
        self.model_status = Some(status);
        self.model_error = None;
        self.model_scores = None;
        self.log(&msg);
        Ok(())
    }
    fn autoload_model(&mut self) {
        let path = self.model_path.clone();
        if PathBuf::from(&path).exists() {
            let _ = self.load_model_from_path(&path);
        } else {
            self.model_status = None;
            self.model_error = Some("Model file not found".to_string());
        }
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
            self.icon_tex =
                Some(ctx.load_texture("qnmdsol_icon", color_image, TextureOptions::LINEAR));
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
    fn apply_waveform_pipeline_config(&mut self) {
        if let Some(pipe) = &mut self.waveform_pipeline {
            let y_scale = if self.wave_auto_scale {
                YScale::Auto
            } else {
                YScale::FixedMicrovolts(self.wave_fixed_range_uv.max(10.0))
            };
            pipe.set_global_y_scale(y_scale);
            let filters = if self.wave_notch_50hz {
                vec![FilterKind::Notch {
                    freq_hz: 50.0,
                    q: 35.0,
                }]
            } else {
                Vec::new()
            };
            for idx in 0..pipe.channel_count() {
                pipe.set_channel_enabled(idx, true);
                pipe.set_channel_filters(idx, filters.clone());
            }
        }
    }
    fn show_waveform(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        // 顶部提示
        ui.horizontal_wrapped(|ui| {
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
        // 行1：灵敏度 / 平滑度 + 窗口长度
        ui.horizontal_wrapped(|ui| {
            ui.label(self.text(UiText::Sensitivity));
            ui.add(
                egui::Slider::new(&mut self.signal_sensitivity, 0.05..=8.0)
                    .logarithmic(true)
                    .show_value(false),
            );
            ui.monospace(format!("{:.2}", self.signal_sensitivity));
            ui.label(self.text(UiText::Smoothness));
            ui.add(egui::Slider::new(&mut self.smooth_alpha, 0.0..=0.8).show_value(false));
            ui.monospace(format!("{:.2}", self.smooth_alpha));
            ui.separator();
            ui.label(self.text(UiText::Window));
            for (label, seconds) in [
                (self.text(UiText::Window30), 30.0),
                (self.text(UiText::Window60), 60.0),
            ] {
                let selected = (self.wave_window_seconds - seconds).abs() < f64::EPSILON;
                if ui.selectable_label(selected, label).clicked() {
                    self.wave_window_seconds = seconds;
                    self.view_seconds = seconds;
                    self.wave_smooth_state.clear();
                    if let Some(pipe) = &mut self.waveform_pipeline {
                        pipe.set_time_window(TimeWindow::new(seconds as f32));
                        self.waveform_view = Some(pipe.view());
                    }
                }
            }
            ui.separator();
            ui.label(self.text(UiText::TimeAxis));
            let mut range = self.wave_window_seconds.clamp(5.0, 120.0);
            if ui
                .add(
                    egui::Slider::new(&mut range, 5.0..=120.0)
                        .logarithmic(false)
                        .show_value(true),
                )
                .changed()
            {
                self.wave_window_seconds = range;
                self.view_seconds = range;
                self.wave_smooth_state.clear();
                if let Some(pipe) = &mut self.waveform_pipeline {
                    pipe.set_time_window(TimeWindow::new(range as f32));
                    self.waveform_view = Some(pipe.view());
                }
            }
        });
        // 行2：分辨率 + 量程 / 滤波 + 阈值/丢包率
        ui.horizontal_wrapped(|ui| {
            ui.label(self.text(UiText::Resolution));
            let auto_y_label = self.text(UiText::AutoY);
            let fixed_uv_label = self.text(UiText::FixedUv);
            let notch_label = self.text(UiText::Notch50);
            let stats_label = self.text(UiText::Stats);
            for (label, size) in [
                ("960x540", [960.0, 540.0]),
                ("1280x720", [1280.0, 720.0]),
                ("1600x900", [1600.0, 900.0]),
            ] {
                if ui.button(label).clicked() {
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::Fullscreen(false));
                    ui.ctx()
                        .send_viewport_cmd(egui::ViewportCommand::InnerSize(size.into()));
                }
            }
            if ui.button(self.text(UiText::Maximize)).clicked() {
                ui.ctx()
                    .send_viewport_cmd(egui::ViewportCommand::Maximized(true));
            }
            ui.separator();
            let mut changed = false;
            changed |= ui
                .checkbox(&mut self.wave_auto_scale, auto_y_label)
                .changed();
            let resp = ui.add_enabled(
                !self.wave_auto_scale,
                egui::Slider::new(&mut self.wave_fixed_range_uv, 50.0..=800.0)
                    .show_value(false)
                    .text(fixed_uv_label),
            );
            changed |= resp.changed();
            changed |= ui
                .checkbox(&mut self.wave_notch_50hz, notch_label)
                .changed();
            changed |= ui
                .checkbox(&mut self.wave_show_stats, stats_label)
                .changed();
            if changed {
                self.apply_waveform_pipeline_config();
                if let Some(pipe) = &mut self.waveform_pipeline {
                    self.waveform_view = Some(pipe.view());
                }
            }
            ui.separator();
            ui.label(format!(
                "{} {:.1}",
                self.text(UiText::Threshold),
                self.trigger_threshold
            ));
            if let Some(start) = self.stream_start {
                let elapsed = start.elapsed().as_secs_f64();
                let expected = elapsed * self.waveform_sample_rate_hz as f64;
                ui.separator();
                if let Some(last) = self.last_data_at {
                    let since = last.elapsed().as_secs_f64();
                    if expected > 1.0 {
                        let actual = self.total_samples_ingested as f64;
                        let rate = (1.0 - actual / expected).clamp(0.0, 1.0) * 100.0;
                        ui.label(format!(
                            "{} {:.2}%",
                            if self.language == Language::Chinese {
                                "丢包率:"
                            } else {
                                "Drop:"
                            },
                            rate
                        ));
                        ui.label(format!(
                            "{} {:.1}s",
                            if self.language == Language::Chinese {
                                "最近一帧"
                            } else {
                                "Last frame"
                            },
                            since
                        ));
                    }
                } else {
                    ui.label(if self.language == Language::Chinese {
                        "未收到数据"
                    } else {
                        "No data received"
                    });
                }
            }
        });
        let available_h = ui.available_height();
        let mut _placeholder: Option<WaveformView> = None;
        let view: &WaveformView = if let Some(v) = self.waveform_view.as_ref() {
            v
        } else {
            _placeholder = Some(WaveformView {
                window_secs: self.wave_window_seconds as f32,
                channels: (0..16)
                    .map(|i| ChannelView {
                        index: i,
                        y_range: (-200.0, 200.0),
                        rms_u_v: 0.0,
                        min: 0.0,
                        max: 0.0,
                        samples: Vec::<SamplePoint>::new(),
                    })
                    .collect(),
            });
            _placeholder.as_ref().unwrap()
        };
        let channel_count = view.channels.len().max(16);
        if self.wave_smooth_state.len() != channel_count {
            self.wave_smooth_state = vec![0.0; channel_count];
        }
        let max_points_per_channel: usize = 1400;
        let colors = [
            Color32::from_rgb(118, 94, 186),
            Color32::from_rgb(83, 134, 203),
            Color32::from_rgb(67, 160, 71),
            Color32::from_rgb(0, 150, 136),
            Color32::from_rgb(255, 193, 7),
            Color32::from_rgb(230, 81, 0),
            Color32::from_rgb(244, 67, 54),
            Color32::from_rgb(255, 87, 34),
            Color32::from_rgb(171, 71, 188),
            Color32::from_rgb(79, 195, 247),
            Color32::from_rgb(76, 175, 80),
            Color32::from_rgb(205, 220, 57),
            Color32::from_rgb(121, 85, 72),
            Color32::from_rgb(96, 125, 139),
            Color32::from_rgb(33, 150, 243),
            Color32::from_rgb(255, 111, 0),
        ];
        let lane_height = (available_h / channel_count as f32).clamp(18.0, 42.0) as f64;
        let y_span = lane_height * 0.35;
        let x_min = -(view.window_secs as f64);
        let x_max = 0.0;
        let total_height = lane_height * channel_count as f64 + y_span * 2.0;
        let plot_height = total_height.max(available_h as f64) as f32;
        let y_min = -((channel_count as f64 - 1.0) * lane_height + y_span * 1.3);
        let y_max = y_span * 1.3;
        let smooth_alpha = self.smooth_alpha.clamp(0.0, 1.0);
        let empty: &[crate::waveform::view::SamplePoint] = &[];
        let uv_to_height = if y_span.abs() < f64::EPSILON {
            1.0
        } else {
            y_span / 160.0
        };
        egui::ScrollArea::vertical()
            .auto_shrink([false; 2])
            .show(ui, |ui| {
                Plot::new("waveform_plot")
                    .include_x(x_min)
                    .include_x(x_max)
                    .include_y(y_min)
                    .include_y(y_max)
                    .allow_drag(false)
                    .allow_zoom(false)
                    .show_axes([false, false])
                    .show_grid(false)
                    .height(plot_height)
                    .show(ui, |plot_ui| {
                        plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                            [x_min, y_min],
                            [x_max, y_max],
                        ));
                        for idx in 0..channel_count {
                            let ch_opt = view.channels.iter().find(|c| c.index == idx);
                            let samples = ch_opt.map(|c| c.samples.as_slice()).unwrap_or(empty);
                            let rms = ch_opt.map(|c| c.rms_u_v).unwrap_or(0.0);
                            let base = -(idx as f64) * lane_height;
                            let col = colors.get(idx).unwrap_or(&Color32::WHITE);
                            let step = samples
                                .len()
                                .checked_div(max_points_per_channel)
                                .unwrap_or(0)
                                .max(1);
                            let mut points: Vec<[f64; 2]> = Vec::new();
                            for sample in samples.iter().step_by(step) {
                                let scaled = sample.value as f64
                                    * self.display_gain as f64
                                    * self.signal_sensitivity as f64
                                    * uv_to_height;
                                let prev = self.wave_smooth_state.get(idx).copied().unwrap_or(0.0);
                                let smoothed = if smooth_alpha <= 0.0 || smooth_alpha >= 1.0 {
                                    scaled
                                } else {
                                    prev * (1.0 - smooth_alpha) + scaled * smooth_alpha
                                };
                                if let Some(state) = self.wave_smooth_state.get_mut(idx) {
                                    *state = smoothed;
                                }
                                let clamped = smoothed.clamp(-y_span, y_span);
                                points.push([sample.time as f64, base + clamped]);
                            }
                            if points.is_empty() {
                                if let Some(state) = self.wave_smooth_state.get_mut(idx) {
                                    *state = 0.0;
                                }
                                points.push([x_min, base]);
                                points.push([x_max, base]);
                            }
                            let boundary_color = Color32::from_gray(200);
                            plot_ui.line(
                                Line::new(PlotPoints::new(vec![
                                    [x_min, base + y_span],
                                    [x_max, base + y_span],
                                ]))
                                .color(boundary_color),
                            );
                            plot_ui.line(
                                Line::new(PlotPoints::new(vec![
                                    [x_min, base - y_span],
                                    [x_max, base - y_span],
                                ]))
                                .color(boundary_color),
                            );
                            plot_ui.line(
                                Line::new(PlotPoints::new(vec![[x_min, base], [x_max, base]]))
                                    .color(Color32::from_gray(140)),
                            );
                            plot_ui.line(
                                Line::new(PlotPoints::new(points))
                                    .color(*col)
                                    .name(format!("Ch{}", idx + 1)),
                            );
                            let label_x = x_min + view.window_secs as f64 * 0.02;
                            let rms_x = x_min + view.window_secs as f64 * 0.35;
                            plot_ui.text(
                                egui_plot::Text::new(
                                    [label_x, base + y_span * 0.6].into(),
                                    format!("{:02}", idx + 1),
                                )
                                .color(Color32::WHITE),
                            );
                            plot_ui.text(
                                egui_plot::Text::new(
                                    [rms_x, base + y_span * 0.2].into(),
                                    format!("{:.1} uVrms", rms),
                                )
                                .color(*col),
                            );
                            if self.wave_show_stats {
                                if let Some(ch) = ch_opt {
                                    let stats = format!(
                                        "min {:.0} / max {:.0} | y [{:.0}, {:.0}]",
                                        ch.min, ch.max, ch.y_range.0, ch.y_range.1
                                    );
                                    plot_ui.text(
                                        Text::new([label_x, base - y_span * 0.35].into(), stats)
                                            .color(Color32::from_gray(120)),
                                    );
                                }
                            }
                        }
                    });
            });
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
            let summary = match self.language {
                Language::English => format!(
                    "FFT @ {:.1} Hz, channels: {}",
                    spec.sample_rate_hz,
                    spec.channel_labels.len()
                ),
                Language::Chinese => format!(
                    "FFT {:.1} Hz，通道数: {}",
                    spec.sample_rate_hz,
                    spec.channel_labels.len()
                ),
            };
            ui.label(summary);
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
                    let batch = make_batch(
                        frame.sample_rate_hz,
                        frame.samples.clone(),
                        frame.channel_labels.clone(),
                    );
                    let manual_source = ManualSource::new(vec![batch]);
                    let mut pipeline =
                        SignalPipeline::new(manual_source, self.wave_window_seconds as f32);
                    match pipeline.pump_once() {
                        Ok(Some(wave_frame)) => {
                            match render_waveform_png(&wave_frame, PlotStyle::default()) {
                                Ok(png) => self.wave_png = Some(png),
                                Err(e) => {
                                    let msg = match self.language {
                                        Language::English => format!("Wave PNG failed: {e}"),
                                        Language::Chinese => format!("波形导出失败: {e}"),
                                    };
                                    self.log(&msg);
                                }
                            }
                        }
                        Ok(None) => {
                            let msg = match self.language {
                                Language::English => "Wave pipeline empty, no PNG.".to_owned(),
                                Language::Chinese => "波形流水线为空，无法导出PNG。".to_owned(),
                            };
                            self.log(&msg);
                        }
                        Err(e) => {
                            let msg = match self.language {
                                Language::English => format!("Wave pipeline failed: {e}"),
                                Language::Chinese => format!("波形流水线失败: {e}"),
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
                let spec = if let Some(frame) = self.last_frame.clone() {
                    let batch = make_batch(
                        frame.sample_rate_hz,
                        frame.samples.clone(),
                        frame.channel_labels.clone(),
                    );
                    let manual_source = ManualSource::new(vec![batch]);
                    let _trait_ref: &dyn SignalSource = &manual_source;
                    let mut pipeline =
                        SignalPipeline::new(manual_source, self.wave_window_seconds as f32);
                    match pipeline.pump_once() {
                        Ok(_) => match pipeline.latest_spectrum(self.fft_size) {
                            Ok(spec) => Some(spec),
                            Err(e) => {
                                let msg = match self.language {
                                    Language::English => format!("Spectrum calc failed: {e}"),
                                    Language::Chinese => format!("频谱计算失败: {e}"),
                                };
                                self.log(&msg);
                                None
                            }
                        },
                        Err(e) => {
                            let msg = match self.language {
                                Language::English => format!("Spectrum pipeline failed: {e}"),
                                Language::Chinese => format!("频谱流水线失败: {e}"),
                            };
                            self.log(&msg);
                            None
                        }
                    }
                } else {
                    self.last_spectrum.clone()
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
            ui.label(format!("Rest µ-power: {:.3}", self.calib_rest_max));
            ui.label(format!("Imagery µ-power: {:.3}", self.calib_act_max));
        } else {
            ui.label(self.text(UiText::ConnectStreamFirst));
        }
    }
    fn run_resistance_check(&mut self) {
        if !self.is_connected || !self.is_streaming {
            self.log(self.text(UiText::ConnectStreamFirst));
            return;
        }
        let Some(frame) = self.last_frame.as_ref() else {
            self.log(self.text(UiText::ImpedanceNoData));
            return;
        };
        if frame.samples.is_empty() {
            self.log(self.text(UiText::ImpedanceNoData));
            return;
        }
        let channels: Vec<&[f32]> = frame.samples.iter().map(|c| c.as_slice()).collect();
        let values = cyton_impedances_from_samples(&channels);
        self.resistance_labels = frame.channel_labels.clone();
        self.resistance_window_seconds = Some(frame.duration_seconds());
        self.resistance_last_measured = Some(SystemTime::now());
        self.resistance_values = Some(values);
        self.log(self.text(UiText::ImpedanceUpdated));
    }
    fn show_impedance(&mut self, ui: &mut egui::Ui) {
        ui.heading(self.text(UiText::TabImpedance));
        ui.label(self.text(UiText::ImpedanceDesc));
        ui.add_space(8.0);
        let can_measure = self.is_connected && self.is_streaming;
        let button = egui::Button::new(
            egui::RichText::new(self.text(UiText::ImpedanceAction)).color(Color32::WHITE),
        )
        .min_size(Vec2::new(160.0, 28.0))
        .fill(if self.theme_dark {
            Color32::from_rgb(60, 100, 170)
        } else {
            Color32::from_rgb(60, 120, 210)
        });
        if ui.add_enabled(can_measure, button).clicked() {
            self.run_resistance_check();
        }
        if !can_measure {
            ui.label(self.text(UiText::ConnectStreamFirst));
        }
        ui.separator();
        if let Some(values) = self.resistance_values.as_ref() {
            let labels: Vec<String> = if self.resistance_labels.is_empty() {
                (1..=values.len()).map(|i| format!("Ch{i}")).collect()
            } else {
                self.resistance_labels.clone()
            };
            // 循环高亮每个通道，便于“轮询”查看
            if values.len() > 0 {
                let now = Instant::now();
                let advance = match self.impedance_last_cycle {
                    Some(t) => t.elapsed().as_millis() > 600,
                    None => true,
                };
                if advance {
                    self.impedance_highlight_idx =
                        (self.impedance_highlight_idx + 1) % values.len();
                    self.impedance_last_cycle = Some(now);
                }
            }
            egui::Grid::new("resistance_grid")
                .striped(true)
                .show(ui, |ui| {
                    ui.label(self.text(UiText::ImpedanceChannelHeader));
                    ui.label(self.text(UiText::ImpedanceValueHeader));
                    ui.end_row();
                    for (row, (label, value)) in labels.iter().zip(values.iter()).enumerate() {
                        let ohms = *value;
                        let (color, status) = Self::impedance_status(ohms, self.language);
                        let marker = egui::RichText::new("⬤").color(color);
                        ui.horizontal(|ui| {
                            if row == self.impedance_highlight_idx {
                                ui.visuals_mut().extreme_bg_color =
                                    Color32::from_rgba_unmultiplied(80, 120, 200, 30);
                            }
                            ui.label(marker);
                            ui.label(label);
                        });
                        ui.label(format!("{:.2} kΩ ({status})", ohms / 1000.0));
                        ui.end_row();
                    }
                });
            if let Some(window) = self.resistance_window_seconds {
                ui.label(format!("{} {:.1}s", self.text(UiText::Window), window));
            }
            if let Some(first) = values.first() {
                let ganglion_k = ganglion_display_impedance_kohms((*first as f32) / 1000.0);
                ui.label(format!("Ganglion 显示(kΩ)：{:.2}", ganglion_k));
            }
            if let Some(frame) = self.last_frame.as_ref() {
                if let Some(ch) = frame.samples.get(0) {
                    let mean: f32 = ch.iter().copied().sum::<f32>() / ch.len().max(1) as f32;
                    let variance: f32 = ch
                        .iter()
                        .map(|v| {
                            let d = *v - mean;
                            d * d
                        })
                        .sum::<f32>()
                        / ch.len().max(1) as f32;
                    let std = variance.sqrt();
                    let imp = cyton_impedance_from_std(std);
                    ui.label(format!("Ch1 即时估算(Ω)：{:.0}", imp));
                }
            }
            if let Some(measured_at) = self.resistance_last_measured {
                if let Ok(elapsed) = measured_at.elapsed() {
                    ui.label(format!(
                        "{} {:.0}s",
                        self.text(UiText::ImpedanceUpdated),
                        elapsed.as_secs_f32()
                    ));
                } else {
                    ui.label(self.text(UiText::ImpedanceUpdated));
                }
            }
        } else {
            ui.label(self.text(UiText::ImpedanceNoData));
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
                        ui.add(egui::Image::new(tex).fit_to_exact_size(Vec2::new(64.0, 64.0)));
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
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
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
            if ctx.input(|i| i.key_down(egui::Key::Num1)) {
                input.key_1 = true;
            }
            if ctx.input(|i| i.key_down(egui::Key::Num2)) {
                input.key_2 = true;
            }
            self.tx_cmd.send(GuiCommand::UpdateSimInput(input)).ok();
        }
        // 消息处理
        let mut msg_count = 0;
        while let Ok(msg) = self.rx.try_recv() {
            msg_count += 1;
            if msg_count > 2000 {
                match msg {
                    BciMessage::GamepadUpdate(gp) => {
                        self.gamepad_target = gp;
                        self.last_gamepad_update = Some(Instant::now());
                    }
                    BciMessage::ModelPrediction(scores) => {
                        self.model_scores = Some(scores);
                    }
                    _ => continue,
                }
            } else {
                match msg {
                    BciMessage::Log(s) => self.log(&s),
                    BciMessage::Status(b) => {
                        self.is_connected = b;
                        if !b {
                            self.resistance_values = None;
                            self.resistance_window_seconds = None;
                            self.resistance_last_measured = None;
                            self.resistance_labels.clear();
                        }
                    }
                    BciMessage::VJoyStatus(b) => self.is_vjoy_active = b,
                    BciMessage::GamepadUpdate(gp) => {
                        self.gamepad_target = gp;
                        self.last_gamepad_update = Some(Instant::now());
                    }
                    BciMessage::ModelPrediction(scores) => {
                        self.model_scores = Some(scores);
                    }
                    BciMessage::RecordingStatus(b) => self.is_recording = b,
                    BciMessage::Spectrum(spec) => {
                        self.last_spectrum = Some(spec);
                    }
                    BciMessage::DataFrame(frame) => {
                        let sr = frame.sample_rate_hz;
                        if sr <= 0.0 {
                            continue;
                        }
                        self.last_frame = Some(frame.clone());
                        let channel_count = frame.samples.len();
                        let needs_new_pipeline = self
                            .waveform_pipeline
                            .as_ref()
                            .map(|p| p.channel_count() != channel_count)
                            .unwrap_or(true)
                            || (self.waveform_sample_rate_hz - sr).abs() > f32::EPSILON;
                        if needs_new_pipeline {
                            self.waveform_pipeline = Some(WaveformPipeline::new(channel_count, sr));
                            self.wave_smooth_state = vec![0.0; channel_count];
                            self.waveform_view = None;
                            self.stream_start = None;
                            self.waveform_clock = 0.0;
                            self.total_samples_ingested = 0;
                            self.waveform_last_len = 0;
                            self.vertical_spacing = 240.0_f64.max(self.vertical_spacing);
                            self.stream_start = Some(Instant::now());
                            self.apply_waveform_pipeline_config();
                            if let Some(pipe) = &mut self.waveform_pipeline {
                                let zeros = vec![0.0; channel_count];
                                pipe.ingest_frame(0.0, &zeros);
                            }
                        }
                        if let Some(pipe) = &mut self.waveform_pipeline {
                            pipe.set_time_window(TimeWindow::new(self.wave_window_seconds as f32));
                            let total_samples = frame.samples.first().map(|c| c.len()).unwrap_or(0);
                            if total_samples == 0 {
                                continue;
                            }
                            // 初次填充：填满当前窗口长度的尾巴
                            let window_cap = (self.wave_window_seconds * sr as f64).ceil() as usize;
                            let chunk_size =
                                if self.waveform_clock == 0.0 && self.waveform_last_len == 0 {
                                    total_samples.min(window_cap)
                                } else {
                                    // 后续每帧仅摄入约 1/8 秒的新数据，确保持续刷新又不积压
                                    let target = (sr / 8.0).ceil() as usize;
                                    target.clamp(1, total_samples.min(window_cap))
                                };
                            let start_idx = total_samples.saturating_sub(chunk_size);
                            let mut tails: Vec<Vec<f32>> = Vec::with_capacity(frame.samples.len());
                            for ch in &frame.samples {
                                tails.push(ch.iter().skip(start_idx).cloned().collect());
                            }
                            let start_time = self.waveform_clock;
                            pipe.ingest_block(start_time, &tails);
                            self.waveform_clock += chunk_size as f32 / sr;
                            self.waveform_last_len = total_samples;
                            self.total_samples_ingested =
                                self.total_samples_ingested.saturating_add(chunk_size);
                            self.waveform_view = Some(pipe.view());
                            self.waveform_sample_rate_hz = sr;
                            self.last_data_at = Some(Instant::now());
                        }
                    }
                    BciMessage::CalibrationResult(_, max) => {
                        self.is_calibrating = false;
                        self.clear_progress();
                        if self.calib_rest_max == 0.0 {
                            self.calib_rest_max = max;
                            let msg = match self.language {
                                Language::English => format!("Rest µ-power: {:.3}", max),
                                Language::Chinese => format!("基线：{:.1}", max),
                            };
                            self.log(&msg);
                        } else {
                            self.calib_act_max = max;
                            let msg = match self.language {
                                Language::English => format!("Imagery µ-power: {:.3}", max),
                                Language::Chinese => format!("动作：{:.1}", max),
                            };
                            self.log(&msg);
                            // Hardware mode now uses pure EEG µ-band power mapping for forward axis.
                        }
                    }
                }
            }
        }
        // 动画插值
        // 没有新按键消息一段时间则复位手柄状态，避免常亮
        if !self.is_streaming
            || self
                .last_gamepad_update
                .map(|t| t.elapsed().as_secs_f32() > 0.5)
                .unwrap_or(true)
        {
            self.gamepad_target = GamepadState::default();
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
        egui::TopBottomPanel::top("topbar_min").show(ctx, |ui| {
            ui.horizontal(|ui| {
                let toggle_label = if self.control_panel_open {
                    self.text(UiText::HidePanel)
                } else {
                    self.text(UiText::ShowPanel)
                };
                if ui.button(toggle_label).clicked() {
                    self.control_panel_open = !self.control_panel_open;
                }
                ui.separator();
                ui.label(self.text(UiText::Title));
                ui.label(
                    egui::RichText::new(self.text(UiText::Subtitle))
                        .color(Color32::from_rgb(120, 120, 130)),
                );
            });
        });
        if self.control_panel_open {
            egui::SidePanel::left("control_panel")
                .resizable(true)
                .default_width(self.control_panel_width)
                .width_range(220.0..=480.0)
                .show(ctx, |ui| {
                    self.control_panel_width = ui.available_width();
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            let sim_label = self.text(UiText::Sim);
                            let real_label = self.text(UiText::Real);
                            if let Some(tex) = &self.icon_tex {
                                ui.add(
                                    egui::Image::new(tex).fit_to_exact_size(Vec2::new(24.0, 24.0)),
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
                        });
                        ui.separator();
                        ui.horizontal_wrapped(|ui| {
                            if ui.button(self.text(UiText::ThemeLight)).clicked() {
                                self.theme_dark = false;
                                self.apply_theme(ctx);
                            }
                            if ui.button(self.text(UiText::ThemeDark)).clicked() {
                                self.theme_dark = true;
                                self.apply_theme(ctx);
                            }
                        });
                        ui.separator();
                        ui.label(self.text(UiText::LanguageSwitch));
                        let mut selected_language = self.language;
                        egui::ComboBox::from_id_source("language_switcher_side")
                            .selected_text(match self.language {
                                Language::English => "English",
                                Language::Chinese => "中文",
                            })
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut selected_language,
                                    Language::English,
                                    "English",
                                );
                                ui.selectable_value(
                                    &mut selected_language,
                                    Language::Chinese,
                                    "中文",
                                );
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
                        ui.separator();
                        if self.connection_mode == ConnectionMode::Hardware {
                            ui.label(self.text(UiText::PortLabel));
                            egui::ComboBox::from_id_source("port_selector_side")
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
                        ui.separator();
                        ui.heading(self.text(UiText::ModelSection));
                        ui.horizontal(|ui| {
                            ui.label(self.text(UiText::ModelPath));
                            ui.text_edit_singleline(&mut self.model_path);
                        });
                        let path_to_load = self.model_path.clone();
                        if ui.button(self.text(UiText::ModelReload)).clicked() {
                            if let Err(e) = self.load_model_from_path(&path_to_load) {
                                self.model_error = Some(e.clone());
                                let msg = match self.language {
                                    Language::English => format!("Model load failed: {e}"),
                                    Language::Chinese => format!("模型加载失败: {e}"),
                                };
                                self.log(&msg);
                            }
                        }
                        if let Some(status) = &self.model_status {
                            ui.label(format!(
                                "{}: {}",
                                self.text(UiText::ModelLoaded),
                                status.path
                            ));
                            ui.label(format!(
                                "{}: {}",
                                self.text(UiText::ModelChannels),
                                status.info.n_channels
                            ));
                            ui.label(format!(
                                "{}: {}",
                                self.text(UiText::ModelClasses),
                                status.info.classes.join(", ")
                            ));
                        } else if let Some(err) = &self.model_error {
                            ui.colored_label(
                                Color32::from_rgb(200, 60, 60),
                                format!("{}: {err}", self.text(UiText::ModelError)),
                            );
                        } else {
                            ui.label(self.text(UiText::ModelNone));
                        }
                        let connect_label = if self.is_connected {
                            self.text(UiText::Disconnect)
                        } else {
                            self.text(UiText::Connect)
                        };
                        if ui.button(connect_label).clicked() {
                            if self.is_connected {
                                self.tx_cmd.send(GuiCommand::Disconnect).ok();
                                self.stream_start = None;
                                self.gamepad_target = GamepadState::default();
                                self.last_gamepad_update = None;
                            } else {
                                self.tx_cmd
                                    .send(GuiCommand::Connect(
                                        self.connection_mode,
                                        self.selected_port.clone(),
                                    ))
                                    .ok();
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
                                    self.tx_cmd.send(GuiCommand::StopStream).ok();
                                    self.is_streaming = false;
                                    self.stream_start = None;
                                } else {
                                    self.tx_cmd.send(GuiCommand::StartStream).ok();
                                    self.is_streaming = true;
                                    self.stream_start = Some(Instant::now());
                                }
                            }
                            if ui.button(self.text(UiText::ResetView)).clicked() {
                                self.waveform_pipeline = None;
                                self.waveform_view = None;
                                self.waveform_last_len = 0;
                                self.waveform_clock = 0.0;
                                self.wave_smooth_state.clear();
                                self.stream_start = None;
                                self.gamepad_target = GamepadState::default();
                                self.last_gamepad_update = None;
                            }
                            let follow_label = if self.follow_latest {
                                self.text(UiText::FollowOn)
                            } else {
                                self.text(UiText::FollowOff)
                            };
                            if ui.button(follow_label).clicked() {
                                self.follow_latest = !self.follow_latest;
                            }
                            if self.connection_mode == ConnectionMode::Simulation
                                && self.is_streaming
                            {
                                if ui.button(self.text(UiText::InjectArtifact)).clicked() {
                                    self.tx_cmd.send(GuiCommand::InjectArtifact).ok();
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
                        ui.separator();
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
                                self.tx_cmd.send(GuiCommand::StopRecording).ok();
                            } else {
                                self.tx_cmd
                                    .send(GuiCommand::StartRecording(self.record_label.clone()))
                                    .ok();
                            }
                        }
                        if self.is_connected && self.is_streaming {
                            if ui.button(self.text(UiText::RecordRelax)).clicked() {
                                self.calib_rest_max = 0.0;
                                self.is_calibrating = true;
                                self.calib_timer = 3.0;
                                self.tx_cmd.send(GuiCommand::StartCalibration(false)).ok();
                            }
                            if ui.button(self.text(UiText::RecordAction)).clicked() {
                                self.calib_act_max = 0.0;
                                self.is_calibrating = true;
                                self.calib_timer = 3.0;
                                self.tx_cmd.send(GuiCommand::StartCalibration(true)).ok();
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
        }
        egui::SidePanel::right("status_panel")
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
                ui.label(self.text(UiText::ModelOutput));
                if let Some(status) = &self.model_status {
                    let classes = &status.info.classes;
                    let scores = self
                        .model_scores
                        .clone()
                        .unwrap_or_else(|| vec![0.0; classes.len()]);
                    for (idx, name) in classes.iter().enumerate() {
                        let v = scores.get(idx).copied().unwrap_or(0.0).clamp(0.0, 1.0);
                        ui.horizontal(|ui| {
                            ui.label(name);
                            ui.add(
                                egui::ProgressBar::new(v)
                                    .show_percentage()
                                    .desired_width(140.0),
                            );
                        });
                    }
                } else if let Some(err) = &self.model_error {
                    ui.colored_label(
                        Color32::from_rgb(200, 60, 60),
                        format!("{}: {err}", self.text(UiText::ModelError)),
                    );
                } else {
                    ui.label(self.text(UiText::ModelNone));
                }
                ui.separator();
                ui.label(self.text(UiText::Logs));
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
                    (self.text(UiText::TabImpedance), ViewTab::Impedance),
                ] {
                    let selected = self.selected_tab == tab;
                    if ui.selectable_label(selected, label).clicked() {
                        self.selected_tab = tab;
                    }
                }
            });
            ui.separator();
            match self.selected_tab {
                ViewTab::Waveform => self.show_waveform(ui, frame),
                ViewTab::Spectrum => self.show_spectrum(ui),
                ViewTab::Png => self.show_png(ui),
                ViewTab::Calibration => self.show_calibration(ui),
                ViewTab::Impedance => self.show_impedance(ui),
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
            (Language::English, UiText::Subtitle) => "Neural Interface Control",
            (Language::English, UiText::Sim) => "Simulation",
            (Language::English, UiText::Real) => "Hardware",
            (Language::English, UiText::Connect) => "Connect",
            (Language::English, UiText::Disconnect) => "Disconnect",
            (Language::English, UiText::StartStream) => "Start Stream",
            (Language::English, UiText::StopStream) => "Stop Stream",
            (Language::English, UiText::ResetView) => "Reset View",
            (Language::English, UiText::Controller) => "Xbox Controller Visualizer",
            (Language::English, UiText::Data) => "AI Data Collection",
            (Language::English, UiText::Recording) => "Recording...",
            (Language::English, UiText::HardwareRequired) => "Hardware required",
            (Language::English, UiText::KeyHint) => "Try Keys: WASD / Space / ZXC / QEUO / Arrows",
            (Language::English, UiText::ConnectFirst) => "Connect first.",
            (Language::English, UiText::Threshold) => "Trigger Threshold:",
            (Language::English, UiText::Calibration) => "Calibration",
            (Language::English, UiText::FollowOn) => "Follow Latest: ON",
            (Language::English, UiText::FollowOff) => "Follow Latest: OFF",
            (Language::English, UiText::Ready) => "QNMDsol Demo v0.1 Ready.",
            (Language::English, UiText::LanguagePrompt) => "Choose your language",
            (Language::English, UiText::StartSubtitle) => "Pick a language to start",
            (Language::English, UiText::StartHeading) => "Welcome to QNMDsol",
            (Language::English, UiText::StartRecording) => "Record",
            (Language::English, UiText::StopRecording) => "Stop",
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
            (Language::English, UiText::TabImpedance) => "Resistance Check",
            (Language::English, UiText::ImpedanceDesc) => {
                "Estimate electrode impedance from the latest buffer (Cyton math)."
            }
            (Language::English, UiText::ImpedanceAction) => "Run check",
            (Language::English, UiText::ImpedanceNoData) => "No impedance result yet.",
            (Language::English, UiText::ImpedanceUpdated) => "Impedance results updated.",
            (Language::English, UiText::ImpedanceChannelHeader) => "Channel",
            (Language::English, UiText::ImpedanceValueHeader) => "Impedance (kOhm)",
            (Language::English, UiText::PortLabel) => "Port:",
            (Language::English, UiText::RefreshPorts) => "Refresh",
            (Language::English, UiText::PortsScanned) => "Ports scanned:",
            (Language::English, UiText::InjectArtifact) => "Inject Artifact",
            (Language::English, UiText::ReportFeedback) => "Report Feedback",
            (Language::English, UiText::ThemeLight) => "Light",
            (Language::English, UiText::ThemeDark) => "Dark",
            (Language::English, UiText::LanguageSwitch) => "Language",
            (Language::English, UiText::Logs) => "Logs",
            (Language::English, UiText::ReportLogs) => "Last Logs:",
            (Language::English, UiText::Resolution) => "Resolution",
            (Language::English, UiText::Maximize) => "Maximize",
            (Language::English, UiText::AutoY) => "Auto Y",
            (Language::English, UiText::FixedUv) => "Fixed uV",
            (Language::English, UiText::Notch50) => "50Hz Notch",
            (Language::English, UiText::Stats) => "Stats",
            (Language::English, UiText::TimeAxis) => "Time span (s)",
            (Language::English, UiText::ShowPanel) => "Show Panel",
            (Language::English, UiText::HidePanel) => "Hide Panel",
            (Language::English, UiText::ImpedanceLegend) => {
                "Good <500k | Acceptable 0.5-2.5M | Poor >2.5M | Railed = no contact"
            }
            (Language::English, UiText::ModelSection) => "AI Model",
            (Language::English, UiText::ModelPath) => "Path",
            (Language::English, UiText::ModelReload) => "Load / Reload",
            (Language::English, UiText::ModelLoaded) => "Model",
            (Language::English, UiText::ModelNone) => "No model loaded.",
            (Language::English, UiText::ModelError) => "Model error",
            (Language::English, UiText::ModelClasses) => "Classes",
            (Language::English, UiText::ModelChannels) => "Channels",
            (Language::English, UiText::ModelOutput) => "Model Output",
            (Language::Chinese, UiText::Title) => "QNMDsol 演示 v0.1",
            (Language::Chinese, UiText::Subtitle) => "神经接口控制",
            (Language::Chinese, UiText::Sim) => "模拟模式",
            (Language::Chinese, UiText::Real) => "实机模式",
            (Language::Chinese, UiText::Connect) => "连接",
            (Language::Chinese, UiText::Disconnect) => "断开",
            (Language::Chinese, UiText::StartStream) => "开始采集",
            (Language::Chinese, UiText::StopStream) => "停止采集",
            (Language::Chinese, UiText::ResetView) => "重置视图",
            (Language::Chinese, UiText::Controller) => "手柄可视化",
            (Language::Chinese, UiText::Data) => "AI数据采集",
            (Language::Chinese, UiText::Recording) => "录制中...",
            (Language::Chinese, UiText::HardwareRequired) => "需要硬件设备",
            (Language::Chinese, UiText::KeyHint) => "键盘提示：WASD / 空格 / ZXC / QEUO / 方向键",
            (Language::Chinese, UiText::ConnectFirst) => "请先连接设备。",
            (Language::Chinese, UiText::Threshold) => "触发阈值：",
            (Language::Chinese, UiText::Calibration) => "校准",
            (Language::Chinese, UiText::FollowOn) => "跟随最新：开",
            (Language::Chinese, UiText::FollowOff) => "跟随最新：关",
            (Language::Chinese, UiText::Ready) => "QNMDsol 演示 v0.1 就绪。",
            (Language::Chinese, UiText::LanguagePrompt) => "选择语言",
            (Language::Chinese, UiText::StartSubtitle) => "选择语言开始",
            (Language::Chinese, UiText::StartHeading) => "欢迎使用 QNMDsol",
            (Language::Chinese, UiText::StartRecording) => "开始录制",
            (Language::Chinese, UiText::StopRecording) => "停止录制",
            (Language::Chinese, UiText::FftSize) => "FFT 大小:",
            (Language::Chinese, UiText::Update) => "更新",
            (Language::Chinese, UiText::GenerateWaveformPng) => "导出波形PNG",
            (Language::Chinese, UiText::GenerateSpectrumPng) => "导出频谱PNG",
            (Language::Chinese, UiText::WaveformPngLabel) => "波形PNG:",
            (Language::Chinese, UiText::SpectrumPngLabel) => "频谱PNG:",
            (Language::Chinese, UiText::NoSpectrumYet) => "暂无频谱，开始采集后生成。",
            (Language::Chinese, UiText::RecordRelax) => "1. 录制静息 (3s)",
            (Language::Chinese, UiText::RecordAction) => "2. 录制动作 (3s)",
            (Language::Chinese, UiText::ConnectStreamFirst) => "请先连接并开始采集。",
            (Language::Chinese, UiText::Loading) => "处理中...",
            (Language::Chinese, UiText::Sensitivity) => "敏感度",
            (Language::Chinese, UiText::Smoothness) => "平滑度",
            (Language::Chinese, UiText::Window) => "窗口长度",
            (Language::Chinese, UiText::Window30) => "30秒",
            (Language::Chinese, UiText::Window60) => "60秒",
            (Language::Chinese, UiText::TabWaveform) => "波形",
            (Language::Chinese, UiText::TabSpectrum) => "频谱",
            (Language::Chinese, UiText::TabPng) => "导出PNG",
            (Language::Chinese, UiText::TabCalibration) => "校准",
            (Language::Chinese, UiText::TabImpedance) => "阻抗检测",
            (Language::Chinese, UiText::ImpedanceDesc) => {
                "基于最新缓冲区估算电极阻抗（Cyton 计算）。"
            }
            (Language::Chinese, UiText::ImpedanceAction) => "执行检测",
            (Language::Chinese, UiText::ImpedanceNoData) => "暂无阻抗结果。",
            (Language::Chinese, UiText::ImpedanceUpdated) => "阻抗结果已更新。",
            (Language::Chinese, UiText::ImpedanceChannelHeader) => "通道",
            (Language::Chinese, UiText::ImpedanceValueHeader) => "阻抗 (kOhm)",
            (Language::Chinese, UiText::PortLabel) => "串口:",
            (Language::Chinese, UiText::RefreshPorts) => "刷新",
            (Language::Chinese, UiText::PortsScanned) => "已扫描串口:",
            (Language::Chinese, UiText::InjectArtifact) => "注入伪迹",
            (Language::Chinese, UiText::ReportFeedback) => "报告反馈",
            (Language::Chinese, UiText::ThemeLight) => "浅色",
            (Language::Chinese, UiText::ThemeDark) => "深色",
            (Language::Chinese, UiText::LanguageSwitch) => "语言",
            (Language::Chinese, UiText::Logs) => "日志",
            (Language::Chinese, UiText::ReportLogs) => "最近日志：",
            (Language::Chinese, UiText::Resolution) => "分辨率",
            (Language::Chinese, UiText::Maximize) => "最大化",
            (Language::Chinese, UiText::AutoY) => "自动Y轴",
            (Language::Chinese, UiText::FixedUv) => "固定范围(uV)",
            (Language::Chinese, UiText::Notch50) => "50Hz 陷波",
            (Language::Chinese, UiText::Stats) => "统计",
            (Language::Chinese, UiText::TimeAxis) => "时间轴长度(秒)",
            (Language::Chinese, UiText::ShowPanel) => "展开面板",
            (Language::Chinese, UiText::HidePanel) => "收起面板",
            (Language::Chinese, UiText::ImpedanceLegend) => {
                "良好 <500k | 可用 0.5-2.5M | 较差 >2.5M | Railed = 无接触"
            }
            (Language::Chinese, UiText::ModelSection) => "AI 模型",
            (Language::Chinese, UiText::ModelPath) => "路径",
            (Language::Chinese, UiText::ModelReload) => "加载 / 重载",
            (Language::Chinese, UiText::ModelLoaded) => "模型",
            (Language::Chinese, UiText::ModelNone) => "未加载模型。",
            (Language::Chinese, UiText::ModelError) => "模型错误",
            (Language::Chinese, UiText::ModelClasses) => "类别",
            (Language::Chinese, UiText::ModelChannels) => "通道数",
            (Language::Chinese, UiText::ModelOutput) => "模型输出",
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
    TabImpedance,
    ImpedanceDesc,
    ImpedanceAction,
    ImpedanceNoData,
    ImpedanceUpdated,
    ImpedanceChannelHeader,
    ImpedanceValueHeader,
    PortLabel,
    RefreshPorts,
    PortsScanned,
    InjectArtifact,
    ReportFeedback,
    ThemeLight,
    ThemeDark,
    LanguageSwitch,
    Logs,
    ReportLogs,
    Resolution,
    Maximize,
    AutoY,
    FixedUv,
    Notch50,
    Stats,
    TimeAxis,
    ShowPanel,
    HidePanel,
    ImpedanceLegend,
    ModelSection,
    ModelPath,
    ModelReload,
    ModelLoaded,
    ModelNone,
    ModelError,
    ModelClasses,
    ModelChannels,
    ModelOutput,
}
#[derive(Clone, Copy, PartialEq, Eq)]
enum ViewTab {
    Waveform,
    Spectrum,
    Png,
    Calibration,
    Impedance,
}
