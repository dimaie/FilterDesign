# Audio Filter Design Studio

An interactive Python desktop application for designing digital audio filters with real-time visual feedback, interactive graph dragging, signal verification testbench, data export, and zero-dependency ANSI C code generation.

---

## Features

1. **Dual Filter Architectures (IIR & FIR)**:
   - **Filter Architectures**:
     - **IIR (Infinite Impulse Response)**: Analog prototype bilinear transform biquad cascades (Butterworth, Chebyshev Type I, Chebyshev Type II, Bessel) up to 10th order (20th order for band filters).
     - **FIR (Finite Impulse Response)**: Linear-phase windowed-sinc Type I symmetric FIR filters (7 to 255 taps) featuring exact linear phase and constant group delay for zero phase distortion.
   - **Filter Topologies**: Low-Pass (LPF), High-Pass (HPF), Band-Pass (BPF), Band-Stop / Notch (BSF).
   - **Comprehensive FIR Window Functions**: 14 standard window functions with dedicated selection browser dialog:
     - **Hamming**: Balanced general audio standard (~53 dB attenuation)
     - **Hann (Hanning)**: Smooth cosine bell with zero endpoints (~44 dB attenuation)
     - **Blackman**: Classic 3-term window (~74 dB stopband attenuation)
     - **Blackman-Harris**: 4-term minimum side-lobe window (~92 dB attenuation)
     - **Nuttall**: Continuous 1st-derivative 4-term window (~93 dB attenuation)
     - **Flat-Top**: Ultra-flat passband amplitude response (calibration standard)
     - **Bohman**: Convolved half-wave cosines with 24 dB/octave decay
     - **Parzen**: 4th-order B-spline with 24 dB/octave decay
     - **Bartlett**: Triangular window with zero endpoints (~26 dB attenuation)
     - **Bartlett-Hann**: Modified combination of Bartlett and Hann
     - **Tukey (Tapered Cosine)**: Cosine-tapered boxcar with adjustable taper fraction ($\alpha$)
     - **Kaiser**: Flexible DPSS approximation with adjustable shape parameter ($\beta$)
     - **Gaussian**: Bell-curve window with adjustable standard deviation ($\sigma$)
     - **Rectangular**: Dirichlet boxcar with narrowest transition band
   - **Sample Rates**: 44.1 kHz, 48 kHz, 88.2 kHz, 96 kHz, 192 kHz, etc.

2. **User-Friendly Drag-and-Drop Graphical Interaction**:
   - Draggable vertical cutoff cursors directly on the logarithmic frequency response canvas (10 Hz – 24 kHz).
   - Real-time recalculation of coefficients and instantaneous frequency response updates while dragging.
   - Dual cutoff dragging for BPF/BSF (Low and High cutoffs).
   - Alternative logarithmic precision sliders and direct numerical input.
   - Dynamic passband / stopband color-coded background zones.
   - Optional Phase Response display.

3. **Real-Time Continuous Audio Synthesizer & Verification Mode** (inspired by Paul Falstad's DFilter):
   - Continuous audio streaming engine running in a dedicated background thread with zero external dependencies (using native Windows `winmm` audio hardware streaming).
   - Instant tone and timbre updates: as you drag cutoffs on the frequency response canvas, change filter order, adjust ripple/attenuation, or switch filter types, the playing sound adapts in real time.
   - Live signal synthesis supporting:
     - **Sine wave** (single tone)
     - **Multi-tone** (harmonic components)
     - **Logarithmic Chirp** (continuous looping exponential frequency sweep)
     - **Linear Chirp** (continuous looping linear frequency sweep)
     - **White Noise** and **Pink Noise** (1/f acoustic noise)
     - **Square, Triangle, and Sawtooth waveforms**
     - **Periodic Noise** (repeating random timbre cycle)
     - **Impulse train**
   - Real-time continuous **Input Frequency slider**: sweep input tones smoothly across the filter spectrum to hear passband and stopband attenuation live.
   - **Sound On/Off toggle** and real-time **Volume slider** (0% to 100%).
   - **Continuous Real-Time Oscilloscope**: Live time-domain dual waveform displaying Input vs Filtered Output.
   - **Continuous Real-Time FFT Spectrum Analyzer**: Live spectrum plot displaying input vs output magnitude spectrum across frequencies at ~30 FPS.
   - Real-time quantitative metrics: Input/Output RMS levels, peak levels, gain/attenuation ($dB$), and instability detection.

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
