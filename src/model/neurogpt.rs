use anyhow::{anyhow, Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::path::{Path, PathBuf};

use crate::drivers::TimeSeriesFrame;
use crate::types::MappingHelperCommand;

pub struct NeuroGPTSession {
    session: Session,
    input_rank: usize,
    input_name: String,
}

#[derive(Debug, Clone)]
pub struct AdaptiveGate {
    warmup: usize,
    cooldown_ms: u64,
    min_prob: f32,
    k_sigma: f32,
    ema_alpha: f32,
    count: usize,
    ema: f32,
    var: f32,
    last_fire: std::time::Instant,
}

impl AdaptiveGate {
    pub fn new() -> Self {
        Self {
            warmup: 30,
            cooldown_ms: 400,
            min_prob: 0.55,
            k_sigma: 2.5,
            ema_alpha: 0.05,
            count: 0,
            ema: 0.0,
            var: 0.0,
            last_fire: std::time::Instant::now() - std::time::Duration::from_secs(10),
        }
    }

    pub fn params(&self) -> crate::types::NeuroGptGateParams {
        crate::types::NeuroGptGateParams {
            warmup: self.warmup as u32,
            cooldown_ms: self.cooldown_ms,
            min_prob: self.min_prob,
            k_sigma: self.k_sigma,
        }
    }

    pub fn set_params(&mut self, p: crate::types::NeuroGptGateParams) {
        self.warmup = p.warmup as usize;
        self.cooldown_ms = p.cooldown_ms;
        self.min_prob = p.min_prob;
        self.k_sigma = p.k_sigma;
        // Keep existing EMA baseline to remain "adaptive"; don't reset count by default.
    }

    pub fn reset_baseline(&mut self, mean: f32, var: f32) {
        self.ema = mean;
        self.var = var.max(0.0);
        self.count = self.warmup.max(1);
        self.last_fire = std::time::Instant::now() - std::time::Duration::from_secs(10);
    }

    pub fn decide(&mut self, probs: &[f32], cmd: MappingHelperCommand) -> Option<MappingHelperCommand> {
        let (top1, top2) = top2_probs(probs)?;
        let margin = (top1 - top2).max(0.0);

        // Update baseline stats on margin; keeps scale stable even if logits shift.
        self.update_stats(margin);

        // Warmup: don't trigger while baseline is still forming.
        if self.count < self.warmup {
            return None;
        }

        // Cooldown.
        if self.last_fire.elapsed() < std::time::Duration::from_millis(self.cooldown_ms) {
            return None;
        }

        // Adaptive threshold: baseline(mean) + k*std, with a floor based on absolute prob.
        let std = self.var.max(0.0).sqrt();
        let adaptive = self.ema + self.k_sigma * std;
        let pass = top1 >= self.min_prob && margin >= adaptive;
        if pass {
            self.last_fire = std::time::Instant::now();
            Some(cmd)
        } else {
            None
        }
    }

    fn update_stats(&mut self, x: f32) {
        self.count += 1;
        if self.count == 1 {
            self.ema = x;
            self.var = 0.0;
            return;
        }
        let prev = self.ema;
        let a = self.ema_alpha.clamp(0.001, 0.5);
        self.ema = (1.0 - a) * self.ema + a * x;
        // Exponential moving variance (approx), keeps var >= 0.
        let diff = x - prev;
        self.var = (1.0 - a) * (self.var + a * diff * diff);
    }
}

/// Channel reorder map from device channel index -> model input channel index.
///
/// `CHANNEL_MAP[model_channel] = device_channel`
///
/// Default is identity (0..15). Adjust this to match your Cyton+Daisy montage and your
/// NeuroGPT training channel order (10-20 mapping), e.g. put the device index for C3
/// at the model index you expect C3 to be.
///
/// Your current hardware montage (device order Ch0..Ch15) is:
/// 0=Fp1, 1=Fp2, 2=C3, 3=C4, 4=P7(T5), 5=P8(T6), 6=O1, 7=O2,
/// 8=F7, 9=F8, 10=F3, 11=F4, 12=T3, 13=T4, 14=P3, 15=P4.
pub const CHANNEL_LABELS_10_20: [&str; 16] = [
    "Fp1", "Fp2", "C3", "C4", "P7", "P8", "O1", "O2", "F7", "F8", "F3", "F4", "T3", "T4", "P3", "P4",
];
pub const CHANNEL_MAP: [usize; 16] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

impl NeuroGPTSession {
    pub fn new() -> Result<Self> {
        let model_path = find_model_path()?;

        // Prefer dynamically loading the bundled runtime if present (repo root ships onnxruntime_x64.dll).
        let env_builder = if Path::new("onnxruntime_x64.dll").exists() {
            ort::init_from("onnxruntime_x64.dll")
        } else if Path::new("onnxruntime.dll").exists() {
            ort::init_from("onnxruntime.dll")
        } else {
            ort::init()
        };
        // If already initialized, commit() returns Ok and is effectively a no-op.
        let _ = env_builder
            .with_name("qnmdsol-neurogpt")
            .with_execution_providers([ort::execution_providers::CPUExecutionProvider::default()
                .build()])
            .commit();

        let session = Session::builder()?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model: {}", model_path.display()))?;

        let input_name = session
            .inputs
            .first()
            .map(|i| i.name.clone())
            .unwrap_or_else(|| "input".to_owned());

        let input_rank = session
            .inputs
            .first()
            .and_then(|i| match &i.input_type {
                ort::value::ValueType::Tensor { shape, .. } => Some(shape.len()),
                _ => None,
            })
            .unwrap_or(3);

        Ok(Self {
            session,
            input_rank,
            input_name,
        })
    }

    pub fn predict_command(
        &mut self,
        frame: &TimeSeriesFrame,
    ) -> Result<(usize, Vec<f32>, MappingHelperCommand)> {
        let scores = self.run(frame)?;
        let probs = softmax(&scores);
        let (idx, _p) = probs
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .ok_or_else(|| anyhow!("Empty model output"))?;
        let cmd = map_class_to_mapping_helper(idx)?;
        Ok((idx, probs, cmd))
    }

    fn run(&mut self, frame: &TimeSeriesFrame) -> Result<Vec<f32>> {
        if frame.samples.len() < 16 {
            return Err(anyhow!(
                "Expected 16 EEG channels (Cyton+Daisy), got {}",
                frame.samples.len()
            ));
        }
        if CHANNEL_MAP.iter().any(|&idx| idx >= 16) {
            return Err(anyhow!("CHANNEL_MAP contains an out-of-range channel index"));
        }

        // The model expects 250 timesteps. Cyton+Daisy is typically 250 Hz, but some setups may run at 125 Hz.
        let src_sr = frame.sample_rate_hz.round() as i32;
        let need = src_sr.max(1) as usize;
        let need = match src_sr {
            125 => 125usize,
            250 => 250usize,
            _ => {
                // Generic path: take 1 second if possible, resample to 250.
                need.min(250).max(1)
            }
        };

        let mut input_data = Vec::<f32>::with_capacity(1 * 16 * 250);
        for model_ch in 0..16 {
            let device_ch = CHANNEL_MAP[model_ch];
            let chan = &frame.samples[device_ch];
            if chan.len() < need {
                return Err(anyhow!(
                    "Not enough samples for NeuroGPT: ch{} has {}, need {}",
                    device_ch,
                    chan.len(),
                    need
                ));
            }
            let start = chan.len() - need;
            let mut x: Vec<f32> = chan[start..].iter().copied().collect();

            let fs = frame.sample_rate_hz;
            bandpass_biquad_inplace(&mut x, fs, 8.0, 30.0)?;

            let y_250 = match src_sr {
                125 => upsample_125_to_250(&x),
                250 => x,
                _ => resample_linear_to_250(&x, fs),
            };
            if y_250.len() != 250 {
                return Err(anyhow!(
                    "Internal resampling error: expected 250 samples, got {}",
                    y_250.len()
                ));
            }
            input_data.extend(y_250);
        }

        // Some exports may expect rank-4 tensors; keep a compatible path.
        let outputs = if self.input_rank == 4 {
            let input_tensor = Tensor::from_array(([1i64, 16, 250, 1], input_data))?;
            self.session
                .run(ort::inputs![self.input_name.as_str() => input_tensor])?
        } else {
            let input_tensor = Tensor::from_array(([1i64, 16, 250], input_data))?;
            self.session
                .run(ort::inputs![self.input_name.as_str() => input_tensor])?
        };

        let first = outputs
            .values()
            .next()
            .ok_or_else(|| anyhow!("ONNX session returned no outputs"))?;
        let (shape, data) = first
            .try_extract_tensor::<f32>()
            .context("Failed to extract output tensor data")?;
        if shape.is_empty() {
            return Ok(data.to_vec());
        }
        // Expected: (batch, classes) or (batch, ..., classes). Take the first batch slice if present.
        let batch = shape[0].max(1) as usize;
        if batch == 0 {
            return Err(anyhow!("Invalid output shape: {shape}"));
        }
        let per_batch = data.len() / batch;
        Ok(data[..per_batch].to_vec())
    }
}

fn find_model_path() -> Result<PathBuf> {
    let candidates = [
        PathBuf::from("models").join("neurogpt.onnx"),
        PathBuf::from("model").join("neurogpt.onnx"),
        PathBuf::from("neurogpt.onnx"),
    ];
    for p in candidates {
        if p.exists() {
            return Ok(p);
        }
    }
    Err(anyhow!(
        "neurogpt.onnx not found; expected at models/neurogpt.onnx or model/neurogpt.onnx"
    ))
}

fn map_class_to_mapping_helper(class_idx: usize) -> Result<MappingHelperCommand> {
    match class_idx {
        0 => Ok(MappingHelperCommand::PulseLeftStickLeft),
        1 => Ok(MappingHelperCommand::PulseLeftStickRight),
        2 => Ok(MappingHelperCommand::PulseLeftStickUp), // Forward
        _ => Err(anyhow!("Unexpected NeuroGPT class idx: {}", class_idx)),
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &v in logits {
        let e = (v - max).exp();
        sum += e;
        exps.push(e);
    }
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|e| e / sum).collect()
}

pub(crate) fn top2_probs(probs: &[f32]) -> Option<(f32, f32)> {
    if probs.is_empty() {
        return None;
    }
    let mut top1 = f32::NEG_INFINITY;
    let mut top2 = f32::NEG_INFINITY;
    for &p in probs {
        if p > top1 {
            top2 = top1;
            top1 = p;
        } else if p > top2 {
            top2 = p;
        }
    }
    if !top1.is_finite() {
        return None;
    }
    if !top2.is_finite() {
        top2 = 0.0;
    }
    Some((top1, top2))
}

fn upsample_125_to_250(x: &[f32]) -> Vec<f32> {
    let n = x.len();
    if n == 0 {
        return vec![];
    }
    let mut y = vec![0.0f32; n * 2];
    for i in 0..n {
        y[2 * i] = x[i];
        y[2 * i + 1] = if i + 1 < n {
            0.5 * (x[i] + x[i + 1])
        } else {
            x[i]
        };
    }
    y
}

fn resample_linear_to_250(x: &[f32], fs_hz: f32) -> Vec<f32> {
    // Resample a ~1 second window (x) to exactly 250 samples.
    // For Cyton+Daisy we mainly care about 125/250; this is a safe fallback.
    let n = x.len();
    if n == 0 || fs_hz <= 0.0 {
        return vec![0.0; 250];
    }
    let duration_s = n as f32 / fs_hz;
    if duration_s <= 0.0 {
        return vec![0.0; 250];
    }

    let mut y = vec![0.0f32; 250];
    for i in 0..250usize {
        let t = (i as f32) * (duration_s / 250.0);
        let src = t * fs_hz;
        let idx0 = src.floor() as isize;
        let frac = src - idx0 as f32;
        let idx0u = idx0.clamp(0, (n - 1) as isize) as usize;
        let idx1u = (idx0u + 1).min(n - 1);
        let v0 = x[idx0u];
        let v1 = x[idx1u];
        y[i] = v0 + frac * (v1 - v0);
    }
    y
}

fn bandpass_biquad_inplace(x: &mut [f32], fs_hz: f32, low_hz: f32, high_hz: f32) -> Result<()> {
    if !(low_hz > 0.0 && high_hz > low_hz && high_hz < fs_hz * 0.5) {
        return Err(anyhow!(
            "Invalid bandpass params: fs={}, low={}, high={}",
            fs_hz,
            low_hz,
            high_hz
        ));
    }

    // Simple 2nd-order bandpass biquad via RBJ cookbook (constant skirt gain, peak gain = Q).
    let f0 = (low_hz * high_hz).sqrt();
    let bw = high_hz - low_hz;
    let q = (f0 / bw).max(0.1);

    let w0 = 2.0 * std::f32::consts::PI * f0 / fs_hz;
    let alpha = w0.sin() / (2.0 * q);
    let cos_w0 = w0.cos();

    let b0 = alpha;
    let b1 = 0.0;
    let b2 = -alpha;
    let a0 = 1.0 + alpha;
    let a1 = -2.0 * cos_w0;
    let a2 = 1.0 - alpha;

    // Direct Form I state
    let mut x1 = 0.0f32;
    let mut x2 = 0.0f32;
    let mut y1 = 0.0f32;
    let mut y2 = 0.0f32;

    for v in x.iter_mut() {
        let x0 = *v;
        let y0 = (b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2) / a0;
        x2 = x1;
        x1 = x0;
        y2 = y1;
        y1 = y0;
        *v = y0;
    }
    Ok(())
}
