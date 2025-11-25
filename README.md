# 🧠 QNMDsol
### Quick Neural Mind-Driven Souls-like Controller

![Rust](https://img.shields.io/badge/Built_with-Rust-orange?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows-blue?style=flat-square)
![Hardware](https://img.shields.io/badge/Hardware-OpenBCI-purple?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

**QNMDsol** is a high-performance Brain-Computer Interface (BCI) game control system built with **Rust**. It is designed to control demanding "Souls-like" action games (e.g., *Elden Ring*, *Code Vein*) using real-time EEG/EMG signals.

By interfacing with **OpenBCI** hardware and mapping biological signals to a **vJoy** virtual gamepad, QNMDsol allows players to attack, move, and interact with the game world using their mind and facial muscle signals.

---

## ✨ Features

* **🚀 Blazing Fast Core:** Powered by Rust's multi-threaded architecture, separating data acquisition, signal processing, and UI rendering for sub-millisecond latency.
* **🎮 Virtual Gamepad Integration:** Fully simulates an Xbox 360 controller via **vJoy**, supporting dual analog sticks, triggers, and ABXY buttons.
* **📈 Real-time Visualization:** A cyberpunk-styled GUI based on `egui`, offering 60FPS waveform rendering and visual controller feedback.
* **🧪 Dual Operation Modes:**
    * **Simulation Mode:** Built-in software signal generator. Use your keyboard to simulate brainwaves and test game mappings without hardware.
    * **Hardware Mode:** Direct connection to OpenBCI Cyton + Daisy (16-channel) via USB Dongle.
* **💾 AI-Ready Data Collection:** Integrated recorder to save raw 16-channel EEG data to CSV for training CSP/LDA or Deep Learning models.

---

## 🛠️ Architecture

The project follows a modular MVC-like structure:

* `src/main.rs`: Entry point and application lifecycle management.
* `src/engine.rs`: **The Brain**. Handles BrainFlow driver interaction, signal processing logic, and threshold detection.
* `src/gui.rs`: **The View**. Handles the egui interface, waveform plotting, and user interaction.
* `src/vjoy.rs`: **The Hand**. Wraps the Windows vJoy C interface for virtual controller output.
* `src/recorder.rs`: **The Memory**. Handles data logging to CSV files.

---

## ⚙️ Prerequisites

### Hardware
* **OpenBCI Cyton + Daisy Board** (16 Channels)
* OpenBCI Programmable USB Dongle

### Software
1.  **Windows 10 / 11** (64-bit)
2.  **Rust Toolchain** (Latest Stable)
3.  **[vJoy Driver](https://github.com/shauleiz/vJoy)** (Must be installed and configured)
4.  **BrainFlow Dynamic Libraries**:
    * `BoardController.dll`
    * `DataHandler.dll`
    * `vJoyInterface.dll`
    * *(Place these in the project root directory)*

---

## 🚀 Quick Start

### 1. Setup
Clone the repository and ensure all DLLs are in the root folder.

```bash
git clone [https://github.com/YourUsername/QNMDsol.git](https://github.com/YourUsername/QNMDsol.git)
cd QNMDsol
# Copy BoardController.dll, DataHandler.dll, vJoyInterface.dll here!
```

### 2. Run Simulation (No Hardware Required)
Test the logic and game mapping immediately using the built-in simulator.

```bash
cargo run
```

1.  Select **SIM** mode in the top-left corner.
2.  Click **CONNECT** -> **START STREAM**.
3.  Use your **Keyboard** to simulate brain signals:
    * **W / A / S / D**: Left Stick (Movement)
    * **I / J / K / L**: Right Stick (Camera)
    * **Space**: Button A (Jump/Confirm)
    * **Z / X / C**: Buttons B / X / Y

### **2.1 🔴 【CRITICAL】Game Recognition Setup (XInput Translation)**

Modern games like Elden Ring only recognize the Xbox (XInput) standard controller, while QNMDsol simulates a generic controller (DirectInput). Therefore, a "translator" is required to bridge this gap.

**We strongly recommend using Steam's built-in "Steam Input" feature for this conversion. It is the cleanest and most stable solution as it requires no file injection.**

#### **Steam Input One-Time Setup Process:**

1.  **Preparation:** Launch Steam and ensure Elden Ring is in your Steam library (add it as a non-Steam game if necessary).
2.  **Generic Support:** Go to Steam -> **Settings** -> **Controller** -> Check **"Enable Generic Gamepad Configuration Support"**.
3.  **Enable Steam Input:** In the Steam Library, right-click Elden Ring -> **Properties** -> **Controller** -> Select **"Enable Steam Input"**.
4.  **Button Mapping:** Click **"Controller Layout"**. You must manually map the signals output by QNMDsol (e.g., **Button 1**, **Axis X/Y**) to the corresponding standard **Xbox 360 Buttons** (e.g., A button, Left Stick).
    * **💡 Tip:** To find the correct binding, run QNMDsol in **SIM mode** first, press the keyboard keys (`W/A/S/D`), and observe which axis is moving in the Steam mapping interface.

After completing these steps, the game will be able to recognize your mind-controlled gamepad.
    
### 3. Run Hardware Mode
Plug in the OpenBCI Dongle (Default: COM4).

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
MIT License

Made with ❤️ and 🧠 by Independent Developer.
