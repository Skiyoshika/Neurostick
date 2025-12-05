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
    StartCalibration(bool),
    UpdateSimInput(SimInputIntent),
    StartRecording(String),
    StopRecording,
    InjectArtifact,
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
    CalibrationResult((), f64),
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
