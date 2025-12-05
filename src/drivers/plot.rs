use std::io::Cursor;

use image::{DynamicImage, ImageBuffer, ImageFormat, Rgb};
use plotters::prelude::LineSeries;
use plotters::prelude::*;

use crate::drivers::error::ModelizeError;
use crate::drivers::fft::FrequencySpectrum;
use crate::drivers::TimeSeriesFrame;

#[derive(Clone, Debug)]
pub struct PlotStyle {
    pub width: u32,
    pub height: u32,
    pub background: RGBColor,
    pub palette: Vec<RGBColor>,
}

impl Default for PlotStyle {
    fn default() -> Self {
        Self {
            width: 900,
            height: 400,
            background: RGBColor(10, 10, 10),
            palette: vec![BLUE, RED, GREEN, CYAN, MAGENTA, YELLOW, WHITE],
        }
    }
}

pub fn render_waveform_png(
    frame: &TimeSeriesFrame,
    style: PlotStyle,
) -> Result<Vec<u8>, ModelizeError> {
    if frame.samples.is_empty() {
        return Err(ModelizeError::Plot(
            "time-series frame has no samples".into(),
        ));
    }

    let mut buffer = vec![0u8; (style.width * style.height * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut buffer, (style.width, style.height))
            .into_drawing_area();
        root.fill(&style.background)?;

        let y_min = frame
            .samples
            .iter()
            .flat_map(|c| c.iter().copied())
            .fold(0.0f32, |acc, v| acc.min(v));
        let y_max = frame
            .samples
            .iter()
            .flat_map(|c| c.iter().copied())
            .fold(0.0f32, |acc, v| acc.max(v));
        let y_bounds = if (y_max - y_min).abs() < f32::EPSILON {
            (-50.0, 50.0)
        } else {
            (y_min, y_max)
        };

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption("Time Series", ("sans-serif", 20).into_font().color(&WHITE))
            .set_label_area_size(LabelAreaPosition::Left, 45)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(0f32..frame.samples[0].len() as f32, y_bounds.0..y_bounds.1)?;

        chart
            .configure_mesh()
            .light_line_style(&WHITE.mix(0.1))
            .draw()?;

        for (idx, channel) in frame.samples.iter().enumerate() {
            let color = style.palette[idx % style.palette.len()];
            let series = channel.iter().enumerate().map(|(i, v)| (i as f32, *v));
            chart
                .draw_series(LineSeries::new(series, &color))?
                .label(
                    frame
                        .channel_labels
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("Ch {idx}")),
                )
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
        }

        chart
            .configure_series_labels()
            .border_style(&WHITE.mix(0.2))
            .background_style(&style.background)
            .draw()?;

        root.present()?;
    }

    encode_png(&buffer, style.width, style.height)
}

pub fn render_spectrum_png(
    spectrum: &FrequencySpectrum,
    style: PlotStyle,
) -> Result<Vec<u8>, ModelizeError> {
    if spectrum.magnitudes.is_empty() {
        return Err(ModelizeError::Plot("spectrum has no magnitudes".into()));
    }

    let mut buffer = vec![0u8; (style.width * style.height * 3) as usize];
    {
        let root = BitMapBackend::with_buffer(&mut buffer, (style.width, style.height))
            .into_drawing_area();
        root.fill(&style.background)?;

        let mut chart = ChartBuilder::on(&root)
            .margin(10)
            .caption(
                "FFT Magnitude",
                ("sans-serif", 20).into_font().color(&WHITE),
            )
            .set_label_area_size(LabelAreaPosition::Left, 45)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .build_cartesian_2d(
                0f32..spectrum.frequencies_hz.last().copied().unwrap_or(0.0),
                0f32..spectrum
                    .magnitudes
                    .iter()
                    .flat_map(|c| c.iter().copied())
                    .fold(0.0f32, |acc, v| acc.max(v))
                    .max(1e-3),
            )?;

        chart
            .configure_mesh()
            .light_line_style(&WHITE.mix(0.1))
            .draw()?;

        for (idx, mags) in spectrum.magnitudes.iter().enumerate() {
            let color = style.palette[idx % style.palette.len()];
            let series = spectrum
                .frequencies_hz
                .iter()
                .cloned()
                .zip(mags.iter().cloned());
            chart
                .draw_series(LineSeries::new(series, &color))?
                .label(
                    spectrum
                        .channel_labels
                        .get(idx)
                        .cloned()
                        .unwrap_or_else(|| format!("Ch {idx}")),
                )
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &color));
        }

        chart
            .configure_series_labels()
            .border_style(&WHITE.mix(0.2))
            .background_style(&style.background)
            .draw()?;

        root.present()?;
    }

    encode_png(&buffer, style.width, style.height)
}

fn encode_png(buffer: &[u8], width: u32, height: u32) -> Result<Vec<u8>, ModelizeError> {
    let image = ImageBuffer::<Rgb<u8>, _>::from_raw(width, height, buffer.to_vec())
        .ok_or_else(|| ModelizeError::Plot("failed to allocate image buffer".into()))?;
    let mut output = Vec::new();
    let dynamic = DynamicImage::ImageRgb8(image);
    dynamic.write_to(&mut Cursor::new(&mut output), ImageFormat::Png)?;
    Ok(output)
}
