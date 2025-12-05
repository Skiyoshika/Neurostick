use rustfft::{num_complex::Complex32, FftPlanner};

use crate::drivers::TimeSeriesFrame;

/// Magnitude spectrum for each channel.
#[derive(Clone, Debug)]
pub struct FrequencySpectrum {
    pub sample_rate_hz: f32,
    pub frequencies_hz: Vec<f32>,
    pub magnitudes: Vec<Vec<f32>>, // channel -> bins
    pub channel_labels: Vec<String>,
}

/// Helper that computes FFTs for a given window size.
pub struct SpectrumBuilder {
    fft_size: usize,
}

impl SpectrumBuilder {
    pub fn with_size(fft_size: usize) -> Self {
        Self { fft_size }
    }

    pub fn compute(&self, frame: &TimeSeriesFrame) -> FrequencySpectrum {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(self.fft_size);
        let mut frequencies = Vec::with_capacity(self.fft_size / 2);
        for k in 0..self.fft_size / 2 {
            let freq = k as f32 * (frame.sample_rate_hz / self.fft_size as f32);
            frequencies.push(freq);
        }

        let magnitudes: Vec<Vec<f32>> = frame
            .samples
            .iter()
            .map(|channel| {
                let mut buffer: Vec<Complex32> = channel
                    .iter()
                    .copied()
                    .take(self.fft_size)
                    .map(|v| Complex32::new(v, 0.0))
                    .collect();
                buffer.resize(self.fft_size, Complex32::ZERO);
                fft.process(&mut buffer);
                buffer
                    .iter()
                    .take(self.fft_size / 2)
                    .map(|c| c.norm() / self.fft_size as f32)
                    .collect()
            })
            .collect();

        FrequencySpectrum {
            sample_rate_hz: frame.sample_rate_hz,
            frequencies_hz: frequencies,
            magnitudes,
            channel_labels: frame.channel_labels.clone(),
        }
    }
}
