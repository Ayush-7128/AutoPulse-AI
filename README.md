# 🚗 AutoPulse AI (Maruti SonicTwin) — Vibro-Acoustic Edge Diagnostics
**Maruti Suzuki Hackathon (IIMCIP) | Domain 2: Automotive AI & Vehicle Intelligence**

### 🎯 Problem Statement
Standard OBD-II ports are blind to **70% of mechanical breakdowns** in Indian vehicles (slipping alternator belts, worn brake pads, starter solenoid clash). **AutoPulse AI** transforms a smartphone into a **Zero-BOM Vibro-Acoustic Virtual Mechanic**.

### 🧠 Key Technical Differentiators
1. **State-Gated Diagnostic Windows:** *Cold Start*, *Neutral Idle*, and *Braking*.
2. **Vibro-Acoustic Sensor Fusion:** Fuses **128-bin Log-Mel Spectrograms** with **100 Hz 3-Axis IMU Telemetry**. Rejects external traffic horns via Z-axis Kurtosis.
3. **Edge-Quantized TFLite:** Compressed to **`1.8 MB`** with `<25 ms` latency.
4. **Maruti Service-Bay Triage:** Maps faults to **Maruti Genuine Parts** and pre-books workshop bays.