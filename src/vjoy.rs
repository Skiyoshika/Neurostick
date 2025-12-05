// src/vjoy.rs
use anyhow::{anyhow, Result};
use libloading::{Library, Symbol};
use std::sync::Arc;

// 定义函数签名
type FnAcquire = unsafe extern "C" fn(u32) -> i32;
type FnRelinquish = unsafe extern "C" fn(u32) -> i32;
type FnSetBtn = unsafe extern "C" fn(i32, u32, u8) -> i32;
type FnSetAxis = unsafe extern "C" fn(i32, u32, u32) -> i32;
type FnReset = unsafe extern "C" fn(u32) -> i32;

pub struct VJoyClient {
    lib: Arc<Library>,
    device_id: u32,
}

impl VJoyClient {
    pub fn new(device_id: u32) -> Result<Self> {
        unsafe {
            let lib_name = "vJoyInterface.dll";
            let lib = Library::new(lib_name)
                .or_else(|_| Library::new("C:\\Program Files\\vJoy\\x64\\vJoyInterface.dll"))
                .map_err(|_| anyhow!("Failed to load vJoy DLL"))?;

            let client = Self {
                lib: Arc::new(lib),
                device_id,
            };
            client.acquire()?;
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

    pub fn reset(&self) {
        unsafe {
            if let Ok(f) = self.lib.get::<FnReset>(b"ResetVJD") {
                f(self.device_id);
            }
        }
    }

    pub fn set_button(&self, btn_id: u8, down: bool) {
        unsafe {
            if let Ok(f) = self.lib.get::<FnSetBtn>(b"SetBtn") {
                f(if down { 1 } else { 0 }, self.device_id, btn_id);
            }
        }
    }

    pub fn set_axis(&self, axis_id: u32, value: i32) {
        unsafe {
            if let Ok(f) = self.lib.get::<FnSetAxis>(b"SetAxis") {
                f(value, self.device_id, axis_id);
            }
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
