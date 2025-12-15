# QNMDsol
Quick Neural Mind-Driven Souls-like Controller

![Rust](https://img.shields.io/badge/Built_with-Rust-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-blue?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-OpenBCI-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

QNMDsol is a Rust app that reads EEG data from **OpenBCI Cyton + Daisy (16ch)** (via BrainFlow) and outputs a **vJoy virtual gamepad** for game control. The UI also provides waveform/spectrum visualization, impedance estimation, and CSV recording.

- English setup: `USAGE.md`
- 中文说明: `使用说明.md`

## What This Repo Supports (current `main`)
- **SIM mode**: keyboard → vJoy (for testing). Keyboard only works when QNMDsol is focused.
- **REAL mode**: Cyton+Daisy (BrainFlow) → simple threshold demo → vJoy output.
- **Waveform/Spectrum**: basic real-time visualization.
- **Calibration tab**: “Record Relax (3s)” + “Record Action (3s)” computes a demo threshold.
- **Impedance tab**: estimates impedance (rough quality check).
- **Recording**: saves EEG to CSV for offline training (`trainer/`).
- **AI Model UI**: can load `brain_model.json` and display placeholder outputs; real-time inference is not wired into `engine` yet.

## Requirements (Windows)
### Hardware
- OpenBCI Cyton + Daisy (16 channels) + USB dongle

### Software
1. Windows 10/11 (64-bit)
2. Rust stable (install: https://rustup.rs)
3. vJoy **v2.2.2.0** (required): https://github.com/BrunnerInnovation/vJoy/releases/tag/v2.2.2.0
4. Runtime DLLs (required; must be in working directory / next to `.exe`):
   - `BoardController.dll` (BrainFlow)
   - `DataHandler.dll` (BrainFlow)
   - `vJoyInterface.dll` (vJoy SDK)

This repository includes these DLLs in the repo root for Windows x64. If you removed them, see `USAGE.md` / `使用说明.md`.

## Quick Start
```bash
git clone https://github.com/Skiyoshika/QNMDsol.git
cd QNMDsol
cargo run
```

## Steam Input (XInput translation)
Most modern games only recognize Xbox controllers (XInput). vJoy is a DirectInput device, so you usually need Steam Input to translate:
1. Steam → Settings → Controller → enable “Generic Gamepad Configuration Support”
2. Game → Properties → Controller → enable “Steam Input”
3. Game → “Controller Layout” → bind vJoy axes/buttons to Xbox controls

Tip: Steam listens to **vJoy device input**, not your keyboard. SIM keyboard shortcuts only work while QNMDsol is focused. Use `joy.cpl` or REAL mode for reliable mapping.

## AI Pipeline (demo/offline)
- Offline scripts live under `trainer/`.
- `trainer/run_all.bat` produces a demo `brain_model.json` in the project root.

## License
MIT License (see `LICENSE`).
