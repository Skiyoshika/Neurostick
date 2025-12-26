# QNMDsol
Quick Neural Mind-Driven Souls-like Controller

![Rust](https://img.shields.io/badge/Built_with-Rust-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-blue?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-OpenBCI-purple?style=flat-square)
![License](https://img.shields.io/badge/License-AGPLv3-blue?style=flat-square)

QNMDsol is a Rust app that reads EEG data from **OpenBCI Cyton + Daisy (16ch)** (via BrainFlow) and outputs a **vJoy virtual gamepad** for game control. The UI also provides waveform/spectrum visualization, impedance estimation, and CSV recording.

- English setup: `USAGE.md`
- 中文说明: `使用说明.md`

## Feature Overview (current `main`)
- **Mode selector (SIM / REAL)**:
  - **SIM**: keyboard shortcuts drive the vJoy device. Only works when the QNMDsol window has focus.
  - **REAL**: OpenBCI Cyton+Daisy (BrainFlow) streams EEG into QNMDsol, which applies a simple threshold demo and drives vJoy.
- **Connection flow**: CONNECT opens the data source (keyboard simulator or serial port), START STREAM begins data acquisition, STOP stream cleanly closes it.
- **Waveform view**: real-time plot of the incoming channels with a reset button to clear history.
- **Spectrum view**: FFT-based magnitude plot with selectable window/size (for quick alpha/beta checks).
- **Calibration tab**:
  - Guided buttons **Record Relax (3s)** and **Record Action (3s)** capture two windows and compute a demo threshold.
  - The resulting Threshold slider can be adjusted manually if the trigger is too sensitive or too hard.
- **Impedance tab**: estimates per-channel impedance quality and labels them (Good / Acceptable / Poor / Railed) for rapid electrode checks.
- **AI Data Collection**: enters a free-form label and records CSV EEG samples for offline training (saved under the repo root; see `trainer/`).
- **AI Model UI**: loads a `brain_model.json` path, reloads on demand, and shows placeholder per-class probabilities in the status bar (inference pipeline is not yet wired into `engine`).
- **vJoy output preview**: left panel mirrors stick/trigger/button states so you can confirm mappings before opening a game.

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

1.  **Preparation:** Launch Steam and ensure Elden Ring is in your Steam library (add it as a non-Steam game if necessary).
2.  **Generic Support:** Go to Steam -> **Settings** -> **Controller** -> Check **"Enable Generic Gamepad Configuration Support"**.
3.  **Enable Steam Input:** In the Steam Library, right-click Elden Ring -> **Properties** -> **Controller** -> Select **"Enable Steam Input"**.
4.  **Button Mapping:** Click **"Controller Layout"**. You must manually map the signals output by QNMDsol (e.g., **Button 1**, **Axis X/Y**) to the corresponding standard **Xbox 360 Buttons** (e.g., A button, Left Stick).
    * **Tip:** Steam detects the **vJoy device input**, not your keyboard. SIM keyboard shortcuts only work when the QNMDsol window is focused. For mapping, prefer REAL mode (EEG drives vJoy in the background) or use `joy.cpl` to confirm axes/buttons first.

After completing these steps, the game will be able to recognize your mind-controlled gamepad.
    
### 3. Run Hardware Mode
Plug in the OpenBCI Dongle (COM port is selectable in the UI).

Turn on the Cyton Board.

Select REAL mode in the GUI.

Click CONNECT. Once connected, click START STREAM.

Data Recording: Enter a label (e.g., "Attack") and click 🔴 RECORD to save training data.

---

## 🗺️ Roadmap
[x] v0.1 Demo: Core architecture, Simulation, vJoy integration, Data recording.

[ ] v0.2 AI Integration: Integrate ONNX Runtime to load Python-trained CSP+LDA models.

[ ] v0.3 Macro System: automated macros for complex in-game actions (e.g., healing, dodging).

[ ] v1.0 Release: Closed-loop bidirectional VR sensory modulation integration.

---

## ⚠️ Disclaimer
This project is in early Alpha.

Do not use BCI devices while operating heavy machinery.

The developer is not responsible for emotional damage caused by in-game deaths ("YOU DIED").

---

## 📄 License
GNU AGPLv3 (see `LICENSE`).

Made with ❤️ and 🧠 by Independent Developer.

## AI Pipeline (demo/offline)
- Offline scripts live under `trainer/`.
- `trainer/run_all.bat` produces a demo `brain_model.json` in the project root.

## License
MIT License (see `LICENSE`).
