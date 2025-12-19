// src/vjoy.rs
use anyhow::{anyhow, Result};
use libloading::{Library, Symbol};
use std::sync::Arc;
// 定义函数签名
// vJoyInterface.dll uses WINAPI (__stdcall) on Windows; use `extern "system"`.
type FnAcquire = unsafe extern "system" fn(u32) -> i32;
type FnRelinquish = unsafe extern "system" fn(u32) -> i32;
type FnSetBtn = unsafe extern "system" fn(i32, u32, u8) -> i32;
type FnSetAxis = unsafe extern "system" fn(i32, u32, u32) -> i32;
type FnSetContPov = unsafe extern "system" fn(i32, u32, u8) -> i32;
type FnGetVJDButtonNumber = unsafe extern "system" fn(u32) -> i32;
type FnGetVJDContPovNumber = unsafe extern "system" fn(u32) -> i32;
type FnGetVJDAxisExist = unsafe extern "system" fn(u32, u32) -> i32;
type FnGetVJDStatus = unsafe extern "system" fn(u32) -> i32;
type FnvJoyEnabled = unsafe extern "system" fn() -> i32;
type FnGetOwnerPid = unsafe extern "system" fn(u32) -> u32;
type FnisVJDExists = unsafe extern "system" fn(u32) -> i32;
type FnReset = unsafe extern "system" fn(u32) -> i32;
pub struct VJoyClient {
    lib: Arc<Library>,
    device_id: u32,
}
impl VJoyClient {
    pub fn new(device_id: u32) -> Result<Self> {
        unsafe {
            // Prefer the installed vJoyInterface.dll to avoid mismatched DLL/driver versions.
            // (A stale copy in the project root can load successfully but fail to drive the device.)
            let candidates = [
                "C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll",
                "C:\\Program Files (x86)\\vJoy\\x64\\vJoyInterface.dll",
                "vJoyInterface.dll",
            ];
            let mut last_err: Option<anyhow::Error> = None;
            let mut loaded: Option<Library> = None;
            for path in candidates {
                match Library::new(path) {
                    Ok(lib) => {
                        loaded = Some(lib);
                        break;
                    }
                    Err(e) => last_err = Some(anyhow!(e)),
                }
            }
            let lib = loaded.ok_or_else(|| {
                last_err.unwrap_or_else(|| anyhow!("Failed to load vJoy DLL"))
            })?;
            let client = Self {
                lib: Arc::new(lib),
                device_id,
            };
            // Fail fast if driver is not enabled.
            if client.vjoy_enabled() == Some(false) {
                return Err(anyhow!("vJoy driver not enabled"));
            }
            if client.vjd_exists() == Some(false) {
                return Err(anyhow!("vJoy device does not exist (id={})", client.device_id));
            }
            client.acquire()?;
            // AcquireVJD can succeed but the device may still not be owned; validate with GetVJDStatus when available.
            if let Some(status) = client.vjd_status() {
                // vJoyInterface.h: VJD_STAT_OWN == 0
                if status != 0 {
                    let owner = client.owner_pid().unwrap_or(0);
                    return Err(anyhow!(
                        "vJoy device not owned after acquire (id={}, status={}, owner_pid={})",
                        client.device_id,
                        status,
                        owner
                    ));
                }
            }
            client.reset();
            Ok(client)
        }
    }
    fn acquire(&self) -> Result<()> {
        unsafe {
            let func: Symbol<FnAcquire> = self.lib.get(b"AcquireVJD")?;
            if func(self.device_id) == 0 {
                return Err(anyhow!("Acquire Failed"));
            }
            Ok(())
        }
    }

    pub fn device_id(&self) -> u32 {
        self.device_id
    }
    pub fn reset(&self) {
        unsafe {
            if let Ok(f) = self.lib.get::<FnReset>(b"ResetVJD") {
                f(self.device_id);
            }
        }
    }

    pub fn vjoy_enabled(&self) -> Option<bool> {
        unsafe {
            let f: Symbol<FnvJoyEnabled> = self.lib.get(b"vJoyEnabled").ok()?;
            Some(f() != 0)
        }
    }

    pub fn vjd_exists(&self) -> Option<bool> {
        unsafe {
            let f: Symbol<FnisVJDExists> = self.lib.get(b"isVJDExists").ok()?;
            Some(f(self.device_id) != 0)
        }
    }

    /// vJoy device status code (vJoyInterface.h: VjdStat).
    pub fn vjd_status(&self) -> Option<i32> {
        unsafe {
            let f: Symbol<FnGetVJDStatus> = self.lib.get(b"GetVJDStatus").ok()?;
            Some(f(self.device_id))
        }
    }

    pub fn owner_pid(&self) -> Option<u32> {
        unsafe {
            let f: Symbol<FnGetOwnerPid> = self.lib.get(b"GetOwnerPid").ok()?;
            Some(f(self.device_id))
        }
    }

    pub fn set_button(&self, btn_id: u8, down: bool) -> bool {
        unsafe {
            if let Ok(f) = self.lib.get::<FnSetBtn>(b"SetBtn") {
                return f(if down { 1 } else { 0 }, self.device_id, btn_id) != 0;
            }
        }
        false
    }

    pub fn set_axis(&self, axis_id: u32, value: i32) -> bool {
        unsafe {
            if let Ok(f) = self.lib.get::<FnSetAxis>(b"SetAxis") {
                return f(value, self.device_id, axis_id) != 0;
            }
        }
        false
    }

    /// Set continuous POV (hat) angle in hundredths of degrees (0..35999), or -1 for neutral.
    pub fn set_cont_pov(&self, pov_id: u8, value: i32) -> bool {
        unsafe {
            if let Ok(f) = self.lib.get::<FnSetContPov>(b"SetContPov") {
                return f(value, self.device_id, pov_id) != 0;
            }
        }
        false
    }

    pub fn button_count(&self) -> Option<u32> {
        unsafe {
            let f: Symbol<FnGetVJDButtonNumber> = self.lib.get(b"GetVJDButtonNumber").ok()?;
            let n = f(self.device_id);
            if n <= 0 { None } else { Some(n as u32) }
        }
    }

    pub fn cont_pov_count(&self) -> Option<u32> {
        unsafe {
            let f: Symbol<FnGetVJDContPovNumber> = self.lib.get(b"GetVJDContPovNumber").ok()?;
            let n = f(self.device_id);
            if n < 0 { None } else { Some(n as u32) }
        }
    }

    pub fn axis_exists(&self, axis_id: u32) -> Option<bool> {
        unsafe {
            let f: Symbol<FnGetVJDAxisExist> = self.lib.get(b"GetVJDAxisExist").ok()?;
            Some(f(self.device_id, axis_id) != 0)
        }
    }
}
impl Drop for VJoyClient {
    fn drop(&mut self) {
        unsafe {
            if let Ok(f) = self.lib.get::<FnRelinquish>(b"RelinquishVJD") {
                f(self.device_id);
            }
        }
    }
}
