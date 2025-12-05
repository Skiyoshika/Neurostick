use std::time::SystemTime;

use crate::drivers::error::ModelizeError;
use crate::drivers::fft::{FrequencySpectrum, SpectrumBuilder};
use crate::drivers::source::{SignalBatch, SignalSource};
use crate::drivers::{SignalBuffer, TimeSeriesFrame};

/// High level pipeline that receives batches and exposes ready-to-plot frames.
pub struct SignalPipeline<S: SignalSource> {
    source: S,
    buffer: Option<SignalBuffer>,
    history_seconds: f32,
}

impl<S: SignalSource> SignalPipeline<S> {
    pub fn new(source: S, history_seconds: f32) -> Self {
        Self {
            source,
            buffer: None,
            history_seconds,
        }
    }

    pub fn pump_once(&mut self) -> Result<Option<TimeSeriesFrame>, ModelizeError> {
        let Some(batch) = self.source.next_batch()? else {
            return Ok(None);
        };
        let frame = self.push_and_snapshot(batch)?;
        Ok(Some(frame))
    }

    pub fn push_and_snapshot(
        &mut self,
        batch: SignalBatch,
    ) -> Result<TimeSeriesFrame, ModelizeError> {
        let history_seconds = self.history_seconds;
        let buffer = self.ensure_buffer(&batch)?;
        buffer.push_batch(&batch)?;
        Ok(buffer.snapshot(history_seconds))
    }

    pub fn latest_frame(&self) -> Result<TimeSeriesFrame, ModelizeError> {
        let buffer = self
            .buffer
            .as_ref()
            .ok_or(ModelizeError::BufferUninitialized)?;
        Ok(buffer.snapshot(self.history_seconds))
    }

    pub fn latest_spectrum(&self, fft_size: usize) -> Result<FrequencySpectrum, ModelizeError> {
        let frame = self.latest_frame()?;
        let builder = SpectrumBuilder::with_size(fft_size);
        Ok(builder.compute(&frame))
    }

    fn ensure_buffer(&mut self, batch: &SignalBatch) -> Result<&mut SignalBuffer, ModelizeError> {
        if self.buffer.is_none() {
            batch.validate()?;
            self.buffer = Some(SignalBuffer::with_history_seconds(
                batch.channel_labels.clone(),
                batch.sample_rate_hz,
                self.history_seconds,
            )?);
        }
        self.buffer
            .as_mut()
            .ok_or(ModelizeError::BufferUninitialized)
    }
}

/// Lightweight helper to produce a batch from owned sample data.
pub fn make_batch(
    sample_rate_hz: f32,
    samples: Vec<Vec<f32>>,
    channel_labels: Vec<String>,
) -> SignalBatch {
    SignalBatch {
        started_at: SystemTime::now(),
        sample_rate_hz,
        samples,
        channel_labels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::drivers::plot::{render_spectrum_png, render_waveform_png, PlotStyle};
    use crate::drivers::source::ManualSource;

    #[test]
    fn pipeline_buffers_and_computes_fft() {
        let batch = make_batch(
            250.0,
            vec![vec![0.0; 128], vec![1.0; 128]],
            vec!["C1".into(), "C2".into()],
        );
        let source = ManualSource::new(vec![batch]);
        let mut pipeline = SignalPipeline::new(source, 1.0);

        let frame = pipeline.pump_once().unwrap().unwrap();
        assert_eq!(frame.samples.len(), 2);
        assert_eq!(frame.samples[0].len(), 128);

        let spectrum = pipeline.latest_spectrum(64).unwrap();
        assert_eq!(spectrum.magnitudes.len(), 2);
        assert_eq!(spectrum.frequencies_hz.len(), 32);
    }

    #[test]
    fn plotting_helpers_return_png() {
        let batch = make_batch(250.0, vec![vec![0.0; 32]], vec!["C1".into()]);
        let source = ManualSource::new(vec![batch]);
        let mut pipeline = SignalPipeline::new(source, 1.0);
        let frame = pipeline.pump_once().unwrap().unwrap();
        let spectrum = pipeline.latest_spectrum(32).unwrap();

        let png_wave = render_waveform_png(&frame, PlotStyle::default()).unwrap();
        let png_fft = render_spectrum_png(&spectrum, PlotStyle::default()).unwrap();
        assert!(!png_wave.is_empty());
        assert!(!png_fft.is_empty());
    }
}
