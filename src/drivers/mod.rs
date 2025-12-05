// src/drivers/mod.rs

// 声明同级目录下的子模块文件
pub mod buffer;
pub mod error;
pub mod fft;
pub mod pipeline;
pub mod plot;
pub mod source;

// 公开导出这些模块里的结构体，方便外部调用
pub use buffer::{SignalBuffer, TimeSeriesFrame};
pub use error::ModelizeError;
pub use fft::{FrequencySpectrum, SpectrumBuilder};
pub use pipeline::SignalPipeline;
pub use plot::{render_spectrum_png, render_waveform_png, PlotStyle};
pub use source::{ManualSource, SignalBatch, SignalSource};
