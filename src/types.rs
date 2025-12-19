use crate::drivers::{FrequencySpectrum, TimeSeriesFrame};
// src/types.rs
#[derive(PartialEq, Clone, Copy, Debug)]
pub enum ConnectionMode {
    Simulation,
    Hardware,
}
#[derive(Clone, Debug)]
pub enum GuiCommand {
    // === 修改：Connect 现在接收 (模式, 端口名) ===
    Connect(ConnectionMode, String),
    Disconnect,
    StartStream,
    StopStream,
    SetThreshold(f64),
    /// Set FFT size used by the engine when emitting `BciMessage::Spectrum`.
    SetFftSize(usize),
    StartCalibration(bool),
    UpdateSimInput(SimInputIntent),
    StartRecording(String),
    StopRecording,
    InjectArtifact,
    /// Helper to generate vJoy input for Steam mapping without keyboard focus.
    SetMappingHelper(MappingHelperCommand),
    /// Update NeuroGPT adaptive trigger gate parameters.
    SetNeuroGptGate(NeuroGptGateParams),
    /// Run a quick NeuroGPT inference self-test and log the output (no hardware required).
    NeuroGptSelfTest,
    /// Start an auto-calibration window for the NeuroGPT adaptive gate (requires streaming).
    NeuroGptCalibrateStart { seconds: u32, target_triggers_per_min: f32 },
}

#[derive(Clone, Copy, Debug)]
pub struct NeuroGptGateParams {
    pub warmup: u32,
    pub cooldown_ms: u64,
    pub min_prob: f32,
    pub k_sigma: f32,
}

#[derive(Clone, Debug)]
pub struct NeuroGptRuntimeStatus {
    pub onnx_loaded: bool,
    pub onnx_path: Option<String>,
    pub last_error: Option<String>,
    pub last_infer_ms_ago: Option<u64>,
    pub gate: NeuroGptGateParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MappingHelperCommand {
    Off,
    PulseA,
    PulseB,
    PulseX,
    PulseY,
    PulseLB,
    PulseRB,
    PulseLT,
    PulseRT,
    PulseBack,
    PulseStart,
    PulseLeftStickClick,
    PulseRightStickClick,
    PulseDpadUp,
    PulseDpadDown,
    PulseDpadLeft,
    PulseDpadRight,
    PulseLeftStickUp,
    PulseLeftStickDown,
    PulseLeftStickLeft,
    PulseLeftStickRight,
    PulseRightStickUp,
    PulseRightStickDown,
    PulseRightStickLeft,
    PulseRightStickRight,
    AutoCycle,
}
#[derive(Clone, Debug)]
pub enum BciMessage {
    Log(String),
    Status(bool),
    VJoyStatus(bool),
    DataFrame(TimeSeriesFrame),
    Spectrum(FrequencySpectrum),
    GamepadUpdate(GamepadState),
    RecordingStatus(bool),
    CalibrationResult(CalibrationTarget, f64),
    ModelPrediction(Vec<f32>),
    NeuroGptStatus(NeuroGptRuntimeStatus),
    NeuroGptTrigger(usize),
    NeuroGptCalibrationProgress { progress01: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CalibrationTarget {
    Relax,
    Action,
}
#[derive(Clone, Copy, Debug, Default)]
pub struct GamepadState {
    pub lx: f32,
    pub ly: f32,
    pub rx: f32,
    pub ry: f32,
    pub a: bool,
    pub b: bool,
    pub x: bool,
    pub y: bool,
    pub lb: bool,
    pub rb: bool,
    pub lt: bool,
    pub rt: bool,
    pub back: bool,
    pub start: bool,
    pub ls: bool,
    pub rs: bool,
    pub dpad_up: bool,
    pub dpad_down: bool,
    pub dpad_left: bool,
    pub dpad_right: bool,
}
#[derive(Default, Clone, Copy, Debug)]
pub struct SimInputIntent {
    pub w: bool,
    pub a: bool,
    pub s: bool,
    pub d: bool,
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub space: bool,
    pub key_z: bool,
    pub key_x: bool,
    pub key_c: bool,
    pub key_1: bool,
    pub key_2: bool,
    pub q: bool,
    pub e: bool,
    pub u: bool,
    pub o: bool,
    pub arrow_up: bool,
    pub arrow_down: bool,
    pub arrow_left: bool,
    pub arrow_right: bool,
}
