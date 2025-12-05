use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::SystemTime;

pub struct DataRecorder {
    writer: Option<BufWriter<File>>,
    start_time: SystemTime,
}

impl DataRecorder {
    pub fn new() -> Self {
        Self {
            writer: None,
            start_time: SystemTime::now(),
        }
    }

    pub fn start(&mut self, label: &str) {
        // 文件名带时间戳和标签，方便后续 AI 识别
        let timestamp = SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let filename = format!("training_data_{}_{}.csv", label, timestamp);

        if let Ok(file) = File::create(&filename) {
            let mut w = BufWriter::new(file);
            // 写入 CSV 表头: Timestamp, Ch0 ... Ch15
            writeln!(
                w,
                "Timestamp,Ch0,Ch1,Ch2,Ch3,Ch4,Ch5,Ch6,Ch7,Ch8,Ch9,Ch10,Ch11,Ch12,Ch13,Ch14,Ch15"
            )
            .ok();
            self.writer = Some(w);
            println!("💾 Recording started: {}", filename);
        }
    }

    pub fn stop(&mut self) {
        if let Some(mut w) = self.writer.take() {
            w.flush().ok();
            println!("💾 Recording saved.");
        }
    }

    pub fn write_record(&mut self, data: &[f64]) {
        if let Some(w) = &mut self.writer {
            // 写入一行数据
            let t = self.start_time.elapsed().unwrap_or_default().as_secs_f64();
            write!(w, "{:.4}", t).ok();
            for val in data.iter().take(16) {
                write!(w, ",{:.2}", val).ok();
            }
            writeln!(w).ok();
        }
    }

    pub fn is_recording(&self) -> bool {
        self.writer.is_some()
    }
}
