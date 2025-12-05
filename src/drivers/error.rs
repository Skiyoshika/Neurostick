use thiserror::Error;

#[derive(Debug, Error)]
pub enum ModelizeError {
    #[error("sample rate must be greater than zero")]
    InvalidSampleRate,
    #[error("sample rate mismatch: expected {expected}, got {actual}")]
    SampleRateMismatch { expected: f32, actual: f32 },
    #[error("channel count mismatch: expected {expected}, got {actual}")]
    ChannelMismatch { expected: usize, actual: usize },
    #[error("buffer not initialized yet; feed at least one batch first")]
    BufferUninitialized,
    #[error("failed to render plot: {0}")]
    Plot(String),
}

impl<E: std::error::Error + Send + Sync + 'static> From<plotters::drawing::DrawingAreaErrorKind<E>>
    for ModelizeError
{
    fn from(value: plotters::drawing::DrawingAreaErrorKind<E>) -> Self {
        ModelizeError::Plot(format!("{value:?}"))
    }
}

impl From<image::ImageError> for ModelizeError {
    fn from(value: image::ImageError) -> Self {
        ModelizeError::Plot(value.to_string())
    }
}
