# Audio Filter Design Studio

An interactive Python desktop application for designing digital audio filters with real-time visual feedback, interactive graph dragging, signal verification testbench, data export, and zero-dependency ANSI C code generation.

---

## Features

1. **Comprehensive Filter Topologies & Responses**:
   - **Filter Types**: Low-Pass (LPF), High-Pass (HPF), Band-Pass (BPF), Band-Stop / Notch (BSF).
   - **Response Approximations**:
     - **Butterworth**: Maximally flat passband response.
     - **Chebyshev Type I**: Steep roll-off with adjustable passband ripple ($dB$).
     - **Chebyshev Type II (Inverse Chebyshev)**: Flat passband with adjustable stopband attenuation ($dB$) and transmission zeros.
     - **Bessel**: Maximally flat group delay and linear phase response for minimal audio transient distortion.
   - **Filter Orders**: 1st through 10th order (cascaded 2nd-order Biquad sections for maximum numerical stability).
   - **Sample Rates**: 44.1 kHz, 48 kHz, 88.2 kHz, 96 kHz, 192 kHz, etc.

2. **User-Friendly Drag-and-Drop Graphical Interaction**:
   - Draggable vertical cutoff cursors directly on the logarithmic frequency response canvas (10 Hz – 24 kHz).
   - Real-time recalculation of coefficients and instantaneous frequency response updates while dragging.
   - Dual cutoff dragging for BPF/BSF (Low and High cutoffs).
   - Alternative logarithmic precision sliders and direct numerical input.
   - Dynamic passband / stopband color-coded background zones.
   - Optional Phase Response display.

3. **Built-in Signal Verification & Test Mode**:
   - Built-in signal generator supporting:
     - **Sine wave** (single tone)
     - **Multi-tone** (harmonic components)
     - **Logarithmic Chirp** (exponential frequency sweep)
     - **Linear Chirp** (linear frequency sweep)
     - **White Noise** and **Pink Noise** (1/f acoustic noise)
     - **Square, Triangle, and Sawtooth waveforms**
     - **Impulse response** and **Step response**
   - **Real-Time Oscilloscope**: Dual time-domain waveform display comparing Input vs Filtered Output.
   - **FFT Spectrum Analyzer**: Dual spectrum plot showing input vs output magnitude spectrum across frequencies.
   - Real-time quantitative metrics: Input/Output RMS levels, peak levels, and attenuation ($dB$).

4. **Multi-Format Export**:
   - **Zero-Dependency ANSI C99 Code**: Generates complete C structs, initialization functions, and sample/block processing functions using Direct Form II Transposed Biquad cascades. Fully self-contained, buildable with any C/C++ compiler (`gcc`, `clang`, `cl`, embedded compilers) without external dependencies.
   - **CSV Data Export**: High-precision time-domain samples (Time, Input, Output).
   - **JSON Export**: Complete filter specification metadata and time/signal arrays.
   - **16-bit PCM WAV Export**: Filtered output signal exported directly to playable audio WAV file.
   - **Coefficients Inspection**: Biquad second-order section parameters ($b_0, b_1, b_2, a_1, a_2$) and expanded Direct Form I/II polynomials ($B(z), A(z)$).

---

## Architecture & Zero External Dependencies

The application is built entirely using Python's standard library (`math`, `cmath`, `random`, `wave`, `struct`, `json`, `csv`, `tkinter`):
- `filter_engine.py`: Core DSP engine implementing analog prototype synthesis, bilinear transform with prewarping, and biquad cascade grouping.
- `filter_canvas.py`: Custom Tkinter canvas supporting logarithmic grids, real-time mouse drag events, and anti-aliased response plotting.
- `signal_testbench.py`: Signal synthesis, Cooley-Tukey radix-2 FFT analyzer, and data exporters.
- `app.py`: Desktop user interface integrating design, test, C code generation, and export workflows.

---

## Running the Application

Launch the desktop UI with:

```powershell
python app.py
```

### Running the Automated Test Suite

```powershell
python -m unittest test_filter_design.py
```
