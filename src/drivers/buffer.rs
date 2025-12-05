use std::collections::VecDeque;

use crate::drivers::ModelizeError;
use crate::drivers::SignalBatch;

/// Flattened view of the current time-domain buffer.
#[derive(Clone, Debug)]
pub struct TimeSeriesFrame {
    pub sample_rate_hz: f32,
    pub channel_labels: Vec<String>,
    pub samples: Vec<Vec<f32>>, // channels x samples
}

impl TimeSeriesFrame {
    pub fn duration_seconds(&self) -> f32 {
        self.samples
            .first()
            .map(|c| c.len() as f32 / self.sample_rate_hz)
            .unwrap_or(0.0)
    }
}

/// Rolling buffer that stores recent samples per channel.
pub struct SignalBuffer {
    per_channel: Vec<VecDeque<f32>>, // channel -> samples
    channel_labels: Vec<String>,
    sample_rate_hz: f32,
    capacity: usize,
}

impl SignalBuffer {
    pub fn with_history_seconds(
        channel_labels: Vec<String>,
        sample_rate_hz: f32,
        history_seconds: f32,
    ) -> Result<Self, ModelizeError> {
        if sample_rate_hz <= 0.0 {
            return Err(ModelizeError::InvalidSampleRate);
        }
        let capacity = (sample_rate_hz * history_seconds).ceil() as usize;
        let per_channel = channel_labels
            .iter()
            .map(|_| VecDeque::with_capacity(capacity))
            .collect();
        Ok(Self {
            per_channel,
            channel_labels,
            sample_rate_hz,
            capacity,
        })
    }

    pub fn sample_rate_hz(&self) -> f32 {
        self.sample_rate_hz
    }

    pub fn channel_labels(&self) -> &[String] {
        &self.channel_labels
    }

    pub fn push_batch(&mut self, batch: &SignalBatch) -> Result<(), ModelizeError> {
        batch.validate()?;
        if batch.sample_rate_hz != self.sample_rate_hz {
            return Err(ModelizeError::SampleRateMismatch {
                expected: self.sample_rate_hz,
                actual: batch.sample_rate_hz,
            });
        }
        if batch.num_channels() != self.per_channel.len() {
            return Err(ModelizeError::ChannelMismatch {
                expected: self.per_channel.len(),
                actual: batch.num_channels(),
            });
        }

        for (channel_queue, new_samples) in self.per_channel.iter_mut().zip(&batch.samples) {
            for &sample in new_samples {
                if channel_queue.len() == self.capacity {
                    channel_queue.pop_front();
                }
                channel_queue.push_back(sample);
            }
        }
        Ok(())
    }

    pub fn snapshot(&self, seconds: f32) -> TimeSeriesFrame {
        let take = (self.sample_rate_hz * seconds).ceil() as usize;
        let samples: Vec<Vec<f32>> = self
            .per_channel
            .iter()
            .map(|channel| channel.iter().rev().take(take).rev().cloned().collect())
            .collect();

        TimeSeriesFrame {
            sample_rate_hz: self.sample_rate_hz,
            channel_labels: self.channel_labels.clone(),
            samples,
        }
    }

    pub fn full_frame(&self) -> TimeSeriesFrame {
        self.snapshot(self.capacity as f32 / self.sample_rate_hz)
    }
}
