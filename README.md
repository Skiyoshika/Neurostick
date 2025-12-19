# 🧠 QNMDsol
### Quick Neural Mind-Driven Souls-like Controller

![Rust](https://img.shields.io/badge/Built_with-Rust-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-blue?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-OpenBCI-purple?style=flat-square)
![License](https://img.shields.io/badge/License-AGPLv3-blue?style=flat-square)

**QNMDsol** is a high-performance Brain-Computer Interface (BCI) game control system built with **Rust**. It is designed to control demanding "Souls-like" action games (e.g., *Elden Ring*, *Code Vein*) using real-time EEG/EMG signals.

By interfacing with **OpenBCI** hardware and mapping biological signals to a **vJoy** virtual gamepad, QNMDsol allows players to attack, move, and interact with the game world using their mind and facial muscle signals.

---

## ✅ Features

* **🚀 Blazing Fast Core:** Multi-threaded architecture separating acquisition, processing, and UI rendering.
* **🎮 Virtual Gamepad Integration:** Simulates a controller via **vJoy** (dual sticks, triggers, ABXY).
* **📈 Real-time Visualization:** `egui` GUI with waveform + spectrum + controller feedback.
* **🧪 Dual Operation Modes:**
  * **Simulation Mode:** Keyboard-driven simulator for mapping/debug without hardware.
  * **Hardware Mode:** OpenBCI Cyton + Daisy (16-channel) via USB dongle.
* **💾 Data Collection:** Record raw 16-channel EEG data to CSV for training.

---

## 🛠️ Architecture

* `src/main.rs`: Entry point and app lifecycle.
* `src/engine.rs`: Acquisition + signal processing + vJoy output.
* `src/gui.rs`: `egui` UI, plotting, user interaction.
* `src/vjoy.rs`: Windows vJoy interface wrapper.
* `src/openbci.rs`: BrainFlow/OpenBCI session.
* `trainer/`: Python scripts for training/export.

---

## ⚙️ Prerequisites

### Hardware
* OpenBCI Cyton + Daisy (16 channels)
* OpenBCI USB dongle

### Software
1. Windows 10 / 11 (64-bit)
2. Rust toolchain (stable)
3. vJoy driver: https://github.com/shauleiz/vJoy
4. BrainFlow dynamic libraries (place in project root):
   * `BoardController.dll`
   * `DataHandler.dll`
   * `vJoyInterface.dll`

---

## 🚀 Quick Start

### 1. Run Simulation (no hardware required)

```bash
cargo run
```

1. Select **SIM** mode in the UI.
2. Click **CONNECT** → **START STREAM**.
3. Keyboard mapping:
   * **W / A / S / D**: Left stick (movement)
   * **I / J / K / L**: Right stick (camera)
   * **Space**: Button A
   * **Z / X / C**: Buttons B / X / Y

### 2.1 🔴 【CRITICAL】Game Recognition Setup (XInput Translation)

Many games only recognize XInput controllers. QNMDsol outputs via vJoy (DirectInput), so you need a translator.

Recommended: **Steam Input**

1. Steam → **Settings** → **Controller** → enable **Generic Gamepad Configuration Support**
2. Game → **Properties** → **Controller** → **Enable Steam Input**
3. **Controller Layout**: map QNMDsol outputs (Button / Axis / POV) to Xbox buttons/sticks.
   * 💡 Tip: use the built-in SIM mode and the Mapping Helper window to generate reliable pulses for binding.

### 3. Run Hardware Mode

1. Plug in the OpenBCI dongle and power on the board.
2. Select **REAL** mode.
3. Pick the correct COM port in the UI (port list is auto-scanned).
4. Click **CONNECT** → **START STREAM**.

---

## AI Pipeline (demo/offline)

- Train CSP+LDA from existing `training_data_*.csv` and export `brain_model.json`:

```bat
trainer\run_all.bat
```

- GUI model path supports `Load / Reload`.

---

## 🗺️ Roadmap

- [x] v0.1 Demo: Core architecture, simulation, vJoy, data recording.
- [ ] v0.2 AI Integration: make `brain_model.json` and/or ONNX inference fully wired by default.
- [ ] v0.3 Macro system: automated in-game macros (healing, dodging, etc.).

---

## ⚠️ Disclaimer

This project is in early alpha. Use at your own risk.

---

## 📄 License

GNU Affero General Public License v3.0 (AGPL-3.0)
