"""
Signal Generator and Verification Testbench for Audio Filter Design.
Supports generating input signals:
- Sine wave (single or multi-tone)
- Frequency Sweep (Chirp: linear or logarithmic)
- White Noise, Pink Noise
- Square wave, Triangle wave, Sawtooth wave
- Unit Impulse, Step function

Processes signals through the designed DigitalFilter, compares input/output,
computes FFT spectrum, SNR/attenuation, and exports data to CSV / JSON / WAV.
Pure Python standard library (math, random, struct, wave, csv, json).
"""

import math
import random
import cmath
import struct
import wave
import csv
import json
from typing import List, Tuple, Dict, Any
from filter_engine import DigitalFilter


def generate_signal(signal_type: str,
                    duration_sec: float,
                    sample_rate: float,
                    freq1: float = 1000.0,
                    freq2: float = 5000.0,
                    amplitude: float = 0.8,
                    noise_amp: float = 0.0) -> Tuple[List[float], List[float]]:
    """
    Generate synthetic test signal.
    Returns (time_array, samples_array).
    """
    num_samples = int(duration_sec * sample_rate)
    time = [i / sample_rate for i in range(num_samples)]
    samples = [0.0] * num_samples

    sig_type = signal_type.lower()

    if sig_type == 'sine':
        omega = 2.0 * math.pi * freq1
        for i in range(num_samples):
            samples[i] = amplitude * math.sin(omega * time[i])

    elif sig_type == 'multi-tone' or sig_type == 'multitone':
        # Two harmonic components
        w1 = 2.0 * math.pi * freq1
        w2 = 2.0 * math.pi * freq2
        for i in range(num_samples):
            samples[i] = amplitude * 0.5 * (math.sin(w1 * time[i]) + math.sin(w2 * time[i]))

    elif sig_type == 'sweep (linear)' or sig_type == 'linear chirp':
        # f(t) = f1 + (f2 - f1) * (t / T)
        t_max = max(duration_sec, 1e-4)
        beta = (freq2 - freq1) / (2.0 * t_max)
        for i in range(num_samples):
            t = time[i]
            phase = 2.0 * math.pi * (freq1 * t + beta * t * t)
            samples[i] = amplitude * math.sin(phase)

    elif sig_type == 'sweep (log)' or sig_type == 'log chirp':
        t_max = max(duration_sec, 1e-4)
        f_start = max(10.0, freq1)
        f_end = max(20.0, freq2)
        rate = math.log(f_end / f_start) / t_max
        for i in range(num_samples):
            t = time[i]
            phase = 2.0 * math.pi * f_start * ((math.exp(rate * t) - 1.0) / rate)
            samples[i] = amplitude * math.sin(phase)

    elif sig_type == 'square':
        period = 1.0 / max(1.0, freq1)
        for i in range(num_samples):
            t_rel = math.fmod(time[i], period)
            samples[i] = amplitude if t_rel < 0.5 * period else -amplitude

    elif sig_type == 'triangle':
        period = 1.0 / max(1.0, freq1)
        for i in range(num_samples):
            t_rel = math.fmod(time[i], period) / period
            if t_rel < 0.25:
                val = 4.0 * t_rel
            elif t_rel < 0.75:
                val = 2.0 - 4.0 * t_rel
            else:
                val = -4.0 + 4.0 * t_rel
            samples[i] = amplitude * val

    elif sig_type == 'sawtooth':
        period = 1.0 / max(1.0, freq1)
        for i in range(num_samples):
            t_rel = math.fmod(time[i], period) / period
            samples[i] = amplitude * (2.0 * t_rel - 1.0)

    elif sig_type == 'white noise':
        for i in range(num_samples):
            samples[i] = amplitude * (random.random() * 2.0 - 1.0)

    elif sig_type == 'pink noise':
        # Voss-McCartney 1/f noise approximation
        b0, b1, b2, b3, b4, b5, b6 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(num_samples):
            white = random.random() * 2.0 - 1.0
            b0 = 0.99886 * b0 + white * 0.0555179
            b1 = 0.99332 * b1 + white * 0.0750759
            b2 = 0.96900 * b2 + white * 0.1538520
            b3 = 0.86650 * b3 + white * 0.3104856
            b4 = 0.55000 * b4 + white * 0.5329522
            b5 = -0.7616 * b5 - white * 0.0168980
            pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + white * 0.5362
            b6 = white * 0.115926
            samples[i] = amplitude * 0.15 * pink

    elif sig_type == 'impulse':
        samples[0] = amplitude
        # Rest remain 0.0

    elif sig_type == 'step':
        # Unit step delayed by 10% of samples
        step_idx = int(0.1 * num_samples)
        for i in range(step_idx, num_samples):
            samples[i] = amplitude

    else:
        # Default sine
        omega = 2.0 * math.pi * freq1
        for i in range(num_samples):
            samples[i] = amplitude * math.sin(omega * time[i])

    # Optional additive white noise
    if noise_amp > 0.0:
        for i in range(num_samples):
            samples[i] += noise_amp * (random.random() * 2.0 - 1.0)

    return time, samples


def compute_fft(signal: List[float], sample_rate: float, max_points: int = 1024) -> Tuple[List[float], List[float]]:
    """
    Cooley-Tukey radix-2 FFT on standard Python list.
    Pads or truncates to power of 2 <= max_points.
    Returns (frequencies_Hz, magnitude_dB).
    """
    n = 1
    target = min(len(signal), max_points)
    while n * 2 <= target:
        n *= 2
    if n < 8:
        n = 8

    # Apply Hann window to reduce spectral leakage
    windowed = [0.0] * n
    for i in range(n):
        s = signal[i] if i < len(signal) else 0.0
        w = 0.5 * (1.0 - math.cos(2.0 * math.pi * i / (n - 1)))
        windowed[i] = s * w

    # Cooley-Tukey FFT
    def _fft(x: List[complex]) -> List[complex]:
        m = len(x)
        if m <= 1:
            return x
        even = _fft(x[0::2])
        odd = _fft(x[1::2])
        t_factors = [cmath.exp(-2j * math.pi * k / m) * odd[k] for k in range(m // 2)]
        return [even[k] + t_factors[k] for k in range(m // 2)] + \
               [even[k] - t_factors[k] for k in range(m // 2)]

    complex_input = [complex(v, 0.0) for v in windowed]
    fft_res = _fft(complex_input)

    # Half spectrum up to Nyquist
    half_n = n // 2
    freqs = [k * sample_rate / n for k in range(half_n)]
    mag_db = []
    norm = 2.0 / n
    for k in range(half_n):
        mag = abs(fft_res[k]) * norm
        db = 20.0 * math.log10(max(mag, 1e-6))
        mag_db.append(db)

    return freqs, mag_db


def export_data_to_csv(filepath: str, time: List[float], in_sig: List[float], out_sig: List[float]):
    """Export time domain input and output signals to CSV."""
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Time_s', 'Input_Sample', 'Output_Sample'])
        for t, x, y in zip(time, in_sig, out_sig):
            writer.writerow([f"{t:.8f}", f"{x:.8f}", f"{y:.8f}"])


def export_data_to_json(filepath: str, filter_metadata: Dict[str, Any],
                        time: List[float], in_sig: List[float], out_sig: List[float]):
    """Export filter config and signal data to JSON."""
    data = {
        "metadata": filter_metadata,
        "num_samples": len(time),
        "time": time,
        "input": in_sig,
        "output": out_sig
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def export_audio_to_wav(filepath: str, samples: List[float], sample_rate: int):
    """
    Export float samples [-1.0, 1.0] to a 16-bit PCM Mono WAV file.
    Uses standard library 'wave' and 'struct'.
    """
    with wave.open(filepath, 'w') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)

        raw_frames = bytearray()
        for s in samples:
            # Clamp to [-1.0, 1.0]
            val = max(-1.0, min(1.0, s))
            # 16-bit signed integer (-32768 to 32767)
            pcm_val = int(val * 32767.0)
            raw_frames.extend(struct.pack('<h', pcm_val))

        wf.writeframes(raw_frames)
