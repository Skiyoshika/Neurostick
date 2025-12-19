use anyhow::{anyhow, Context, Result};
use libloading::Library;
use once_cell::sync::OnceCell;
use serde::Serialize;
use std::ffi::CString;
use std::os::raw::{c_char, c_double, c_int};
use std::path::PathBuf;
const BOARD_ID_CYTON_DAISY: c_int = 2; // matches python trainer script
const PRESET_DEFAULT: c_int = 0;
const STREAM_RINGBUF_PACKETS: c_int = 450_000;
#[derive(Serialize)]
struct BrainFlowInputParams {
    serial_port: String,
    mac_address: String,
    ip_address: String,
    ip_address_aux: String,
    ip_address_anc: String,
    ip_port: i32,
    ip_port_aux: i32,
    ip_port_anc: i32,
    ip_protocol: i32,
    other_info: String,
    timeout: i32,
    serial_number: String,
    file: String,
    file_aux: String,
    file_anc: String,
    master_board: i32,
}
impl BrainFlowInputParams {
    fn for_serial(port: &str) -> Self {
        Self {
            serial_port: port.to_string(),
            mac_address: String::new(),
            ip_address: String::new(),
            ip_address_aux: String::new(),
            ip_address_anc: String::new(),
            ip_port: 0,
            ip_port_aux: 0,
            ip_port_anc: 0,
            ip_protocol: 0,
            other_info: String::new(),
            timeout: 0,
            serial_number: String::new(),
            file: String::new(),
            file_aux: String::new(),
            file_anc: String::new(),
            master_board: -100, // NO_BOARD
        }
    }
}
struct BrainFlowApi {
    #[allow(dead_code)]
    lib: Library,
    set_log_level: Option<unsafe extern "C" fn(c_int) -> c_int>,
    set_log_file: Option<unsafe extern "C" fn(*const c_char) -> c_int>,
    get_error_msg: Option<unsafe extern "C" fn(c_int, *mut c_char, c_int) -> c_int>,
    prepare_session: unsafe extern "C" fn(c_int, *const c_char) -> c_int,
    start_stream: unsafe extern "C" fn(c_int, *const c_char, c_int, *const c_char) -> c_int,
    stop_stream: unsafe extern "C" fn(c_int, *const c_char) -> c_int,
    release_session: unsafe extern "C" fn(c_int, *const c_char) -> c_int,
    get_sampling_rate: unsafe extern "C" fn(c_int, c_int, *mut c_int) -> c_int,
    get_num_rows: unsafe extern "C" fn(c_int, c_int, *mut c_int) -> c_int,
    get_eeg_channels: unsafe extern "C" fn(c_int, c_int, *mut c_int, *mut c_int) -> c_int,
    get_current_board_data: unsafe extern "C" fn(
        c_int,
        c_int,
        *mut c_double,
        *mut c_int,
        c_int,
        *const c_char,
    ) -> c_int,
}
impl BrainFlowApi {
    fn load() -> Result<Self> {
        // BoardController.dll must be next to the executable (already shipped in repo root).
        let lib = unsafe { Library::new("BoardController.dll") }
            .context("BoardController.dll not found in working directory")?;
        // Safety: we assume BrainFlow C API signatures from the official package.
        unsafe {
            let set_log_level = lib
                .get(b"set_log_level_board_controller\0")
                .ok()
                .map(|s: libloading::Symbol<unsafe extern "C" fn(c_int) -> c_int>| *s);
            let set_log_file = lib
                .get(b"set_log_file_board_controller\0")
                .ok()
                .map(|s: libloading::Symbol<unsafe extern "C" fn(*const c_char) -> c_int>| *s);
            let get_error_msg = lib
                .get(b"get_error_msg\0")
                .ok()
                .map(
                    |s: libloading::Symbol<unsafe extern "C" fn(c_int, *mut c_char, c_int) -> c_int>| {
                        *s
                    },
                );

            // Reduce noisy BoardController logs in the terminal; write them to a file instead.
            // Log levels (BrainFlow): 0=TRACE,1=DEBUG,2=INFO,3=WARN,4=ERROR,5=CRITICAL,6=OFF.
            if let Some(f) = set_log_file {
                let _ = std::fs::create_dir_all("logs");
                let path: PathBuf = ["logs", "board_controller.log"].iter().collect();
                // Ensure the file exists so it's easy to find even if BoardController doesn't write.
                let _ = std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(&path);
                if let Ok(path) = CString::new(path.to_string_lossy().as_bytes()) {
                    let _ = f(path.as_ptr());
                }
            }
            if let Some(f) = set_log_level {
                let _ = f(3); // WARN (still quiet in terminal, but keeps warnings in file)
            }

            Ok(Self {
                set_log_level,
                set_log_file,
                get_error_msg,
                prepare_session: *lib.get(b"prepare_session\0")?,
                start_stream: *lib.get(b"start_stream\0")?,
                stop_stream: *lib.get(b"stop_stream\0")?,
                release_session: *lib.get(b"release_session\0")?,
                get_sampling_rate: *lib.get(b"get_sampling_rate\0")?,
                get_num_rows: *lib.get(b"get_num_rows\0")?,
                get_eeg_channels: *lib.get(b"get_eeg_channels\0")?,
                get_current_board_data: *lib.get(b"get_current_board_data\0")?,
                lib,
            })
        }
    }
    fn instance() -> Result<&'static BrainFlowApi> {
        static API: OnceCell<BrainFlowApi> = OnceCell::new();
        API.get_or_try_init(Self::load)
    }
    fn error_text(&self, code: c_int) -> Option<String> {
        let f = self.get_error_msg?;
        let mut buf = vec![0u8; 512];
        let rc = unsafe { f(code, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int) };
        if rc != 0 {
            return None;
        }
        let nul = buf.iter().position(|b| *b == 0).unwrap_or(buf.len());
        Some(String::from_utf8_lossy(&buf[..nul]).trim().to_string())
    }
    fn check(&self, code: c_int, ctx: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            let extra = self.error_text(code).unwrap_or_default();
            if extra.is_empty() {
                Err(anyhow!("{ctx} failed (BrainFlow code {code})"))
            } else {
                Err(anyhow!("{ctx} failed (BrainFlow code {code}): {extra}"))
            }
        }
    }
    fn prepare(&self, board_id: c_int, input: &CString) -> Result<()> {
        self.check(
            unsafe { (self.prepare_session)(board_id, input.as_ptr()) },
            "prepare_session",
        )
    }
    fn start_stream(&self, board_id: c_int, input: &CString) -> Result<()> {
        self.check(
            unsafe {
                (self.start_stream)(
                    STREAM_RINGBUF_PACKETS,
                    std::ptr::null(),
                    board_id,
                    input.as_ptr(),
                )
            },
            "start_stream",
        )
    }
    fn stop_stream(&self, board_id: c_int, input: &CString) -> Result<()> {
        self.check(
            unsafe { (self.stop_stream)(board_id, input.as_ptr()) },
            "stop_stream",
        )
    }
    fn release(&self, board_id: c_int, input: &CString) -> Result<()> {
        self.check(
            unsafe { (self.release_session)(board_id, input.as_ptr()) },
            "release_session",
        )
    }
    fn sampling_rate(&self, board_id: c_int) -> Result<c_int> {
        let mut rate: c_int = 0;
        self.check(
            unsafe { (self.get_sampling_rate)(board_id, PRESET_DEFAULT, &mut rate as *mut c_int) },
            "get_sampling_rate",
        )?;
        Ok(rate)
    }
    fn num_rows(&self, board_id: c_int) -> Result<c_int> {
        let mut rows: c_int = 0;
        self.check(
            unsafe { (self.get_num_rows)(board_id, PRESET_DEFAULT, &mut rows as *mut c_int) },
            "get_num_rows",
        )?;
        Ok(rows)
    }
    fn eeg_channels(&self, board_id: c_int, max_channels: usize) -> Result<Vec<c_int>> {
        let mut out_len: c_int = 0;
        let mut buf = vec![0 as c_int; max_channels.max(32)];
        self.check(
            unsafe {
                (self.get_eeg_channels)(
                    board_id,
                    PRESET_DEFAULT,
                    buf.as_mut_ptr(),
                    &mut out_len as *mut c_int,
                )
            },
            "get_eeg_channels",
        )?;
        buf.truncate(out_len as usize);
        Ok(buf)
    }
    fn current_board_data(
        &self,
        board_id: c_int,
        num_rows: usize,
        input: &CString,
        num_samples: usize,
        buffer: &mut [f64],
    ) -> Result<usize> {
        let mut current_size: c_int = 0;
        self.check(
            unsafe {
                (self.get_current_board_data)(
                    num_samples as c_int,
                    PRESET_DEFAULT,
                    buffer.as_mut_ptr(),
                    &mut current_size as *mut c_int,
                    board_id,
                    input.as_ptr(),
                )
            },
            "get_current_board_data",
        )?;
        let samples = current_size.max(0) as usize;
        let expected = num_rows * num_samples;
        if buffer.len() < expected {
            return Err(anyhow::anyhow!(
                "buffer too small: {} < {}",
                buffer.len(),
                expected
            ));
        }
        Ok(samples)
    }
}
/// BrainFlow-backed session for OpenBCI Cyton + Daisy via USB dongle.
///
/// Compared to the previous raw-serial approach, this uses BrainFlow's
/// `BoardController.dll` so we decode the binary dongle stream reliably and
/// get properly scaled EEG samples.
pub struct OpenBciSession {
    port_name: String,
    api: &'static BrainFlowApi,
    input_json: CString,
    eeg_channels: Vec<c_int>,
    num_rows: usize,
    sample_rate_hz: f32,
    is_streaming: bool,
    released: bool,
}
impl OpenBciSession {
    /// Connects and prepares a BrainFlow session for Cyton+Daisy (board id 2).
    pub fn connect(port_name: &str) -> Result<Self> {
        let api = BrainFlowApi::instance()?;
        let params = BrainFlowInputParams::for_serial(port_name);
        let json = serde_json::to_string(&params)?;
        let input_json =
            CString::new(json).context("failed to encode BrainFlow input params to C string")?;
        api.prepare(BOARD_ID_CYTON_DAISY, &input_json)?;
        let sample_rate_hz = api.sampling_rate(BOARD_ID_CYTON_DAISY)? as f32;
        let num_rows = api.num_rows(BOARD_ID_CYTON_DAISY)? as usize;
        let eeg_channels = api.eeg_channels(BOARD_ID_CYTON_DAISY, num_rows)?;
        Ok(Self {
            port_name: port_name.to_string(),
            api,
            input_json,
            eeg_channels,
            num_rows,
            sample_rate_hz,
            is_streaming: false,
            released: false,
        })
    }
    pub fn port_name(&self) -> &str {
        &self.port_name
    }
    pub fn sample_rate_hz(&self) -> f32 {
        self.sample_rate_hz
    }
    pub fn eeg_channel_count(&self) -> usize {
        self.eeg_channels.len()
    }
    pub fn start_stream(&mut self) -> Result<()> {
        if self.released {
            return Err(anyhow!("session already released; reconnect required"));
        }
        if !self.is_streaming {
            self.api
                .start_stream(BOARD_ID_CYTON_DAISY, &self.input_json)?;
            self.is_streaming = true;
        }
        Ok(())
    }
    pub fn stop_stream(&mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }
        if self.is_streaming {
            self.api
                .stop_stream(BOARD_ID_CYTON_DAISY, &self.input_json)?;
            self.is_streaming = false;
        }
        Ok(())
    }

    /// Releases the BrainFlow session. After calling this, the session cannot be started again.
    pub fn release(&mut self) -> Result<()> {
        if self.released {
            return Ok(());
        }
        if self.is_streaming {
            let _ = self.stop_stream();
        }
        self.api.release(BOARD_ID_CYTON_DAISY, &self.input_json)?;
        self.released = true;
        Ok(())
    }
    /// Pulls the most recent sample for all EEG channels (if any).
    pub fn next_sample(&mut self) -> Result<Option<Vec<f64>>> {
        // We request up to 5 samples to reduce FFI overhead; only the latest is used.
        let max_samples = 5;
        let mut buf = vec![0.0f64; self.num_rows * max_samples];
        let available = self.api.current_board_data(
            BOARD_ID_CYTON_DAISY,
            self.num_rows,
            &self.input_json,
            max_samples,
            &mut buf,
        )?;
        if available == 0 {
            return Ok(None);
        }
        let last_idx = available - 1;
        let mut sample = Vec::with_capacity(self.eeg_channels.len());
        for ch in &self.eeg_channels {
            let ch_idx = *ch as usize;
            if ch_idx < self.num_rows {
                // BrainFlow writes a (num_rows x num_samples_requested) row-major matrix into `buf`.
                // Only the first `available` columns are valid, but the row stride remains `max_samples`.
                let offset = ch_idx * max_samples + last_idx;
                if offset < buf.len() {
                    sample.push(buf[offset]);
                }
            }
        }
        if sample.is_empty() {
            Ok(None)
        } else {
            Ok(Some(sample))
        }
    }
}
impl Drop for OpenBciSession {
    fn drop(&mut self) {
        let _ = self.stop_stream();
        let _ = self.release();
    }
}
