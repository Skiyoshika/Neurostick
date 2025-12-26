# QNMDsol Setup & Deployment (Windows)

This guide matches the current `main` branch behavior and focuses on a minimal, reproducible setup for:
- OpenBCI Cyton + Daisy (16ch) via BrainFlow (`BoardController.dll`)
- vJoy virtual gamepad output
- Steam Input mapping (for XInput-only games)

## 0. What You Get (Current Main)
- **SIM mode**: keyboard drives the virtual gamepad (for testing). Keyboard shortcuts only work when the QNMDsol window is focused.
- **REAL mode**: OpenBCI data → simple threshold demo → vJoy output.
- **Waveform/Spectrum views**: basic real-time visualization.
- **Calibration tab**: records 2×3s windows and computes a demo threshold.
- **Impedance tab**: estimates impedance quality from samples.
- **Recording**: saves EEG samples to CSV for offline training (see `trainer/`).
- **AI Model UI**: loads `brain_model.json` and displays a placeholder output (model inference is not wired in `main` yet).

## 1. Requirements
- Windows 10/11 (64-bit)
- Rust stable toolchain: https://rustup.rs
- OpenBCI Cyton + Daisy + USB Dongle
- vJoy **v2.2.2.0**: https://github.com/BrunnerInnovation/vJoy/releases/tag/v2.2.2.0

## 2. Install & Configure vJoy (required)
1. Install vJoy v2.2.2.0.
2. Run `vJoyConf.exe` (usually `C:\Program Files\vJoy\vJoyConf.exe`).
3. Configure **Device 1**:
   - Axes: enable at least `X`, `Y`, `Rx`, `Ry`
   - Buttons: set to `12` (or more)
   - (Optional) POV: enable 1 POV if you want a D-Pad
   - Click `Apply`
4. Verify the device exists:
   - Press `Win + R`, run `joy.cpl`
   - Select `vJoy Device` → `Properties`

## 3. Ensure Runtime DLLs (required)
QNMDsol loads these DLLs at runtime. They must be present in the **working directory**:
- When running `cargo run`: the repo root
- When running a built `.exe`: next to that `.exe`

Required:
- `BoardController.dll` (BrainFlow)
- `DataHandler.dll` (BrainFlow)
- `vJoyInterface.dll` (vJoy)

This repository includes these DLLs in the repo root for Windows x64.

If you removed them:
- `BoardController.dll` / `DataHandler.dll`: restore from BrainFlow Windows x64 package.
- `vJoyInterface.dll`: copy from `C:\Program Files\vJoy\x64\vJoyInterface.dll` or keep it in repo root.

## 4. Build & Run
```bash
git clone https://github.com/Skiyoshika/QNMDsol.git
cd QNMDsol
cargo run
```

## 5. SIM Mode (no hardware)
1. Select **SIM**
2. Click **CONNECT** → **START STREAM**
3. Use keyboard:
   - `W/A/S/D`: Left stick
   - `I/J/K/L`: Right stick
   - `Space`: A
   - `Z/X/C`: B/X/Y

Note: these keys are read only when the QNMDsol window has focus.

## 6. REAL Mode (hardware)
1. Close OpenBCI GUI (it may occupy the COM port).
2. Plug in the dongle, power on the board.
3. Select **REAL**
4. Choose the correct **COM port** in the dropdown.
5. Click **CONNECT** → **START STREAM**
6. Use `joy.cpl` to verify vJoy reacts.

### 6.1 Calibration (demo threshold)
1. Open the **Calibration** tab.
2. Click **Record Relax (3s)** (baseline).
3. Click **Record Action (3s)** (your intended mental action).
4. The app computes a demo `Threshold` used for gamepad output.

## 6.2 Feature Reference (panels & knobs)
- **Connection controls**: CONNECT/START STREAM/STOP. CONNECT opens the serial port (REAL) or keyboard simulator (SIM); START STREAM begins acquisition; STOP cleanly closes streams.
- **Waveform view**: scrolls raw channel data. The **Reset View** button clears the plotted history if you want a fresh look.
- **Spectrum view**: shows FFT magnitudes. You can change the FFT window (Hann/Hamming/Blackman) and size for quicker alpha/beta inspection.
- **Calibration tab**: guided buttons **Record Relax (3s)** / **Record Action (3s)** compute a demo threshold that drives vJoy output. The Threshold slider remains adjustable after recording.
- **Impedance tab**: estimates each channel’s impedance band and labels them (Good / Acceptable / Poor / Railed) to spot loose/railed electrodes.
- **AI Data Collection**: enter any label (e.g., `Attack`, `Rest`) and click 🔴 RECORD / ⏹ STOP to write CSV files under the repo root for offline training (`trainer/`).
- **AI Model UI**: set a `brain_model.json` path, click **LOAD**/**RELOAD**, and view placeholder per-class probabilities in the status bar (runtime inference is not wired yet in `main`).
- **vJoy output preview**: the left panel mirrors sticks/triggers/buttons so you can confirm that SIM key presses or REAL threshold triggers are reaching the virtual gamepad before opening a game.

## 7. Steam Input Mapping (XInput translation)
Many games are XInput-only (Xbox controller). vJoy is DirectInput, so you need Steam Input to translate:
1. Steam → **Settings** → **Controller** → enable **Generic Gamepad Configuration Support**
2. Game → **Properties** → **Controller** → enable **Steam Input**
3. Game → **Controller Layout**:
   - Bind vJoy axes/buttons to Xbox controls
4. Restart the game if it does not re-detect controllers.

Important: Steam captures **vJoy device input**. It does not capture QNMDsol’s keyboard shortcuts if QNMDsol is not focused.

## 8. Troubleshooting
- **`joy.cpl` shows vJoy moving but the game says “no controller”**: enable Steam Input for XInput translation.
- **Connect fails / port open failed**: wrong COM port or COM port is occupied. Close OpenBCI GUI and try again.
- **App exits immediately / DLL error**: missing `BoardController.dll` / `DataHandler.dll` / `vJoyInterface.dll` in the working directory.
- **Waveform looks like square waves or huge values**: check signal quality (electrode contact) and ensure you are not saturating; use the Impedance tab as a first-pass check.

