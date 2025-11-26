use anyhow::Result;
use serialport::SerialPort;
use std::io::{BufRead, BufReader, Write};
use std::time::Duration;

/// Minimal OpenBCI (Cyton/Cyton+Daisy) serial session wrapper.
///
/// This module intentionally mirrors the start/stop semantics used in the
/// official OpenBCI GUI so that "start stream" simply sends the `b` command
/// and "stop" sends the `s` command. Data packets are parsed line-by-line and
/// converted into `Vec<f64>` samples for the engine loop to consume.
pub struct OpenBciSession {
    port_name: String,
    reader: BufReader<Box<dyn SerialPort>>,
    is_streaming: bool,
}

impl OpenBciSession {
    /// Connects to the given serial port using the default OpenBCI baud rate.
    pub fn connect(port_name: &str) -> Result<Self> {
        let port = serialport::new(port_name, 115200)
            .timeout(Duration::from_millis(200))
            .open()?;

        Ok(Self {
            port_name: port_name.to_string(),
            reader: BufReader::new(port),
            is_streaming: false,
        })
    }

    /// Returns the port currently in use (for logging/debugging).
    pub fn port_name(&self) -> &str {
        &self.port_name
    }

    /// Sends the OpenBCI start-stream command ("b").
    pub fn start_stream(&mut self) -> Result<()> {
        self.send_command(b"b")?;
        self.is_streaming = true;
        Ok(())
    }

    /// Sends the OpenBCI stop-stream command ("s").
    pub fn stop_stream(&mut self) -> Result<()> {
        if self.is_streaming {
            self.send_command(b"s")?;
            self.is_streaming = false;
        }
        Ok(())
    }

    /// Attempts to read the next line of data from the board, parsing numeric channels.
    ///
    /// The OpenBCI GUI expects ASCII packets; this parser is tolerant to brackets and
    /// whitespace so that it works with both raw numeric dumps and CSV-formatted streams.
    pub fn next_sample(&mut self) -> Option<Vec<f64>> {
        let mut line = String::new();
        match self.reader.read_line(&mut line) {
            Ok(0) => None,
            Ok(_) => {
                let cleaned =
                    line.trim_matches(|c: char| c.is_whitespace() || c == '[' || c == ']');
                let numbers: Vec<f64> = cleaned
                    .split(|c| c == ',' || c == ' ' || c == '\t')
                    .filter_map(|s| if s.is_empty() { None } else { s.parse().ok() })
                    .collect();
                if numbers.is_empty() {
                    None
                } else {
                    Some(numbers)
                }
            }
            Err(_) => None,
        }
    }

    fn send_command(&mut self, cmd: &[u8]) -> Result<()> {
        let port = self.reader.get_mut();
        port.write_all(cmd)?;
        port.write_all(b"\n")?;
        port.flush()?;
        Ok(())
    }
}

impl Drop for OpenBciSession {
    fn drop(&mut self) {
        let _ = self.stop_stream();
    }
}
