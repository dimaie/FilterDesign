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
import io
import csv
import json
import threading
import time
from typing import List, Tuple, Dict, Any, Optional
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

    elif 'periodic' in sig_type:
        period = max(2, int(sample_rate / max(1.0, freq1)))
        rng = random.Random(42)
        pbuf = [amplitude * (rng.random() * 2.0 - 1.0) for _ in range(period)]
        for i in range(num_samples):
            samples[i] = pbuf[i % period]

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


def audio_to_wav_bytes(samples: List[float], sample_rate: int, volume: float = 1.0) -> bytes:
    """Encode float samples as volume-scaled 16-bit PCM Mono WAV data."""
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)

        raw_frames = bytearray()
        for s in samples:
            val = max(-1.0, min(1.0, s * max(0.0, min(1.0, volume))))
            # 16-bit signed integer (-32768 to 32767)
            pcm_val = int(val * 32767.0)
            raw_frames.extend(struct.pack('<h', pcm_val))

        wf.writeframes(raw_frames)
    return buffer.getvalue()


def export_audio_to_wav(filepath: str, samples: List[float], sample_rate: int):
    """Export float samples [-1.0, 1.0] to a 16-bit PCM Mono WAV file."""
    with open(filepath, 'wb') as wav_file:
        wav_file.write(audio_to_wav_bytes(samples, sample_rate))


class ContinuousSignalGenerator:
    """
    Stateful real-time signal generator for continuous audio synthesis.
    Maintains phase continuity across successive block generation calls.
    Supports waveforms (similar to Paul Falstad's DFilter):
    - Sine (single tone)
    - Multi-tone (two harmonic components)
    - Sweep (Log) (smooth repeating logarithmic chirp)
    - Sweep (Linear) (smooth repeating linear chirp)
    - White Noise
    - Pink Noise (1/f acoustic noise)
    - Square wave
    - Triangle wave
    - Sawtooth wave
    - Periodic Noise (repeating random cycle)
    - Impulse train
    """
    def __init__(self, sample_rate: float = 48000.0):
        self.sample_rate = float(sample_rate)
        self.signal_type = "Sine"
        self.freq1 = 500.0
        self.freq2 = 5000.0
        self.amplitude = 0.8
        self.phase1 = 0.0
        self.phase2 = 0.0
        self.sweep_t = 0.0
        self.sweep_period = 2.0  # seconds per sweep cycle
        self.impulse_counter = 0
        self.pink_state = [0.0] * 7
        self.periodic_noise_buf: List[float] = []
        self.periodic_noise_idx = 0
        self._lock = threading.Lock()

    def set_params(self, signal_type: Optional[str] = None,
                   freq1: Optional[float] = None,
                   freq2: Optional[float] = None,
                   amplitude: Optional[float] = None,
                   sample_rate: Optional[float] = None):
        with self._lock:
            if signal_type is not None:
                self.signal_type = signal_type
            if freq1 is not None:
                self.freq1 = max(1.0, float(freq1))
            if freq2 is not None:
                self.freq2 = max(1.0, float(freq2))
            if amplitude is not None:
                self.amplitude = max(0.0, min(1.0, float(amplitude)))
            if sample_rate is not None and sample_rate != self.sample_rate:
                self.sample_rate = float(sample_rate)
                self.periodic_noise_buf.clear()
                self.periodic_noise_idx = 0

    def generate_block(self, num_samples: int) -> Tuple[List[float], List[float]]:
        """
        Generate next contiguous block of samples.
        Returns (time_block, samples_block).
        """
        with self._lock:
            return self._generate_block_locked(num_samples)

    def _generate_block_locked(self, num_samples: int) -> Tuple[List[float], List[float]]:
        sr = self.sample_rate
        dt = 1.0 / sr
        f1 = self.freq1
        f2 = self.freq2
        amp = self.amplitude
        sig_type = self.signal_type.lower()
        two_pi = 2.0 * math.pi

        samples = [0.0] * num_samples
        times = [0.0] * num_samples

        if 'sine' in sig_type and 'multi' not in sig_type:
            dphi = two_pi * f1 * dt
            p = self.phase1
            for i in range(num_samples):
                samples[i] = amp * math.sin(p)
                p = (p + dphi) % two_pi
            self.phase1 = p

        elif 'multi' in sig_type:
            dphi1 = two_pi * f1 * dt
            dphi2 = two_pi * f2 * dt
            p1 = self.phase1
            p2 = self.phase2
            for i in range(num_samples):
                samples[i] = amp * 0.5 * (math.sin(p1) + math.sin(p2))
                p1 = (p1 + dphi1) % two_pi
                p2 = (p2 + dphi2) % two_pi
            self.phase1 = p1
            self.phase2 = p2

        elif 'square' in sig_type:
            dphi = two_pi * f1 * dt
            p = self.phase1
            for i in range(num_samples):
                samples[i] = amp if p < math.pi else -amp
                p = (p + dphi) % two_pi
            self.phase1 = p

        elif 'triangle' in sig_type:
            dphi = two_pi * f1 * dt
            p = self.phase1
            for i in range(num_samples):
                u = p / two_pi
                if u < 0.25:
                    v = 4.0 * u
                elif u < 0.75:
                    v = 2.0 - 4.0 * u
                else:
                    v = -4.0 + 4.0 * u
                samples[i] = amp * v
                p = (p + dphi) % two_pi
            self.phase1 = p

        elif 'sawtooth' in sig_type:
            dphi = two_pi * f1 * dt
            p = self.phase1
            for i in range(num_samples):
                u = p / two_pi
                samples[i] = amp * (2.0 * u - 1.0)
                p = (p + dphi) % two_pi
            self.phase1 = p

        elif 'white' in sig_type:
            for i in range(num_samples):
                samples[i] = amp * (random.random() * 2.0 - 1.0)

        elif 'pink' in sig_type:
            b0, b1, b2, b3, b4, b5, b6 = self.pink_state
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
                samples[i] = amp * 0.15 * pink
            self.pink_state = [b0, b1, b2, b3, b4, b5, b6]

        elif 'periodic' in sig_type:
            period = max(2, int(sr / max(1.0, f1)))
            if len(self.periodic_noise_buf) != period:
                self.periodic_noise_buf = [amp * (random.random() * 2.0 - 1.0) for _ in range(period)]
                self.periodic_noise_idx = 0
            idx = self.periodic_noise_idx
            pbuf = self.periodic_noise_buf
            for i in range(num_samples):
                samples[i] = pbuf[idx]
                idx = (idx + 1) % period
            self.periodic_noise_idx = idx

        elif 'sweep' in sig_type or 'chirp' in sig_type:
            f_start = max(10.0, min(f1, f2))
            f_end = max(20.0, max(f1, f2))
            T = max(0.5, self.sweep_period)
            is_log = 'log' in sig_type
            p = self.phase1
            t = self.sweep_t
            for i in range(num_samples):
                frac = t / T
                if is_log:
                    inst_f = f_start * math.pow(f_end / f_start, frac)
                else:
                    inst_f = f_start + (f_end - f_start) * frac
                samples[i] = amp * math.sin(p)
                p = (p + two_pi * inst_f * dt) % two_pi
                t += dt
                if t >= T:
                    t -= T
            self.phase1 = p
            self.sweep_t = t

        elif 'impulse' in sig_type:
            period = max(1, int(sr / max(1.0, f1)))
            cnt = self.impulse_counter
            for i in range(num_samples):
                if cnt == 0:
                    samples[i] = amp
                else:
                    samples[i] = 0.0
                cnt = (cnt + 1) % period
            self.impulse_counter = cnt

        else:
            dphi = two_pi * f1 * dt
            p = self.phase1
            for i in range(num_samples):
                samples[i] = amp * math.sin(p)
                p = (p + dphi) % two_pi
            self.phase1 = p

        for i in range(num_samples):
            times[i] = i * dt

        return times, samples


class ContinuousAudioEngine:
    """
    Real-time continuous audio synthesis and filtering engine.
    Runs on a dedicated background thread, generating audio continuously,
    filtering it through the active DigitalFilter, maintaining a live snapshot
    for real-time oscilloscope/spectrum display, and streaming output to
    the audio hardware via Windows winmm (waveOut) API.
    Zero external dependencies.
    """
    def __init__(self, sample_rate: float = 48000.0, block_size: int = 1024):
        self.sample_rate = float(sample_rate)
        self.block_size = int(block_size)
        self.generator = ContinuousSignalGenerator(self.sample_rate)

        self.active_filter: Optional[DigitalFilter] = None
        self._pending_filter: Optional[DigitalFilter] = None
        self._filter_lock = threading.Lock()

        self.sound_enabled = True
        self.volume = 0.7  # 0.0 to 1.0
        self._vol_lock = threading.Lock()

        # Thread-safe live visual snapshot
        self._snapshot_lock = threading.Lock()
        self._recent_in = [0.0] * self.block_size
        self._recent_out = [0.0] * self.block_size
        self._metrics = {
            "rms_in": 0.0,
            "rms_out": 0.0,
            "gain_db": 0.0,
            "peak_out": 0.0,
            "unstable": False
        }

        self._running = False
        self._thread: Optional[threading.Thread] = None

        # winmm state
        self._hwave = None
        self._event = None
        self._buffers = []
        self._winmm_available = False
        self._pending_device_restart = False
        self._setup_winmm()

    def _setup_winmm(self):
        try:
            import ctypes
            if hasattr(ctypes, 'windll') and hasattr(ctypes.windll, 'winmm'):
                self._ctypes = ctypes
                self._winmm = ctypes.windll.winmm
                self._kernel32 = ctypes.windll.kernel32

                # Configure kernel32 function signatures for 64-bit safe execution
                self._kernel32.CreateEventW.restype = ctypes.c_void_p
                self._kernel32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]

                self._kernel32.WaitForSingleObject.restype = ctypes.c_ulong
                self._kernel32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]

                self._kernel32.CloseHandle.restype = ctypes.c_int
                self._kernel32.CloseHandle.argtypes = [ctypes.c_void_p]

                # Configure winmm function signatures for 64-bit safe execution
                self._winmm.waveOutOpen.restype = ctypes.c_uint
                self._winmm.waveOutOpen.argtypes = [
                    ctypes.POINTER(ctypes.c_void_p),
                    ctypes.c_uint,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_void_p,
                    ctypes.c_ulong
                ]

                self._winmm.waveOutPrepareHeader.restype = ctypes.c_uint
                self._winmm.waveOutPrepareHeader.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

                self._winmm.waveOutWrite.restype = ctypes.c_uint
                self._winmm.waveOutWrite.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

                self._winmm.waveOutUnprepareHeader.restype = ctypes.c_uint
                self._winmm.waveOutUnprepareHeader.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint]

                self._winmm.waveOutReset.restype = ctypes.c_uint
                self._winmm.waveOutReset.argtypes = [ctypes.c_void_p]

                self._winmm.waveOutClose.restype = ctypes.c_uint
                self._winmm.waveOutClose.argtypes = [ctypes.c_void_p]

                self._winmm_available = True
        except Exception:
            self._winmm_available = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._audio_loop, name="ContinuousAudioEngine", daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.6)
        self._thread = None
        self._close_device()

    def set_sound_enabled(self, enabled: bool):
        with self._vol_lock:
            self.sound_enabled = bool(enabled)

    def set_volume(self, volume_norm: float):
        with self._vol_lock:
            self.volume = max(0.0, min(1.0, float(volume_norm)))

    def set_filter(self, filt: Optional[DigitalFilter]):
        with self._filter_lock:
            self._pending_filter = filt.clone() if filt else None

    def set_signal_params(self, signal_type: Optional[str] = None,
                          freq1: Optional[float] = None,
                          freq2: Optional[float] = None,
                          amplitude: Optional[float] = None):
        self.generator.set_params(signal_type=signal_type, freq1=freq1,
                                  freq2=freq2, amplitude=amplitude)

    def set_sample_rate(self, sample_rate: float):
        sr = float(sample_rate)
        if sr != self.sample_rate:
            with self._filter_lock:
                self.sample_rate = sr
                self.generator.set_params(sample_rate=sr)
                self._pending_device_restart = True

    def get_snapshot(self) -> Tuple[List[float], List[float], Dict[str, Any]]:
        with self._snapshot_lock:
            return list(self._recent_in), list(self._recent_out), dict(self._metrics)

    def _open_device(self) -> bool:
        if not self._winmm_available:
            return False
        if self._hwave is not None:
            return True
        try:
            ct = self._ctypes
            class WAVEFORMATEX(ct.Structure):
                _fields_ = [
                    ('wFormatTag', ct.c_ushort),
                    ('nChannels', ct.c_ushort),
                    ('nSamplesPerSec', ct.c_ulong),
                    ('nAvgBytesPerSec', ct.c_ulong),
                    ('nBlockAlign', ct.c_ushort),
                    ('wBitsPerSample', ct.c_ushort),
                    ('cbSize', ct.c_ushort)
                ]
            class WAVEHDR(ct.Structure):
                pass
            WAVEHDR._fields_ = [
                ('lpData', ct.c_char_p),
                ('dwBufferLength', ct.c_ulong),
                ('dwBytesRecorded', ct.c_ulong),
                ('dwUser', ct.c_void_p),
                ('dwFlags', ct.c_ulong),
                ('dwLoops', ct.c_ulong),
                ('lpNext', ct.POINTER(WAVEHDR)),
                ('reserved', ct.c_void_p)
            ]
            self._WAVEHDR = WAVEHDR

            sr = int(self.sample_rate)
            wfx = WAVEFORMATEX(1, 1, sr, sr * 2, 2, 16, 0)
            self._event = self._kernel32.CreateEventW(None, False, False, None)
            hwave = ct.c_void_p()
            CALLBACK_EVENT = 0x00050000
            WAVE_MAPPER = 0xFFFFFFFF
            res = self._winmm.waveOutOpen(ct.byref(hwave), WAVE_MAPPER, ct.byref(wfx), self._event, 0, CALLBACK_EVENT)
            if res != 0:
                self._kernel32.CloseHandle(self._event)
                self._event = None
                return False
            self._hwave = hwave

            num_buffers = 4
            self._buffers = []
            for _ in range(num_buffers):
                raw_buf = ct.create_string_buffer(self.block_size * 2)
                hdr = WAVEHDR()
                hdr.lpData = ct.cast(raw_buf, ct.c_char_p)
                hdr.dwBufferLength = self.block_size * 2
                hdr.dwFlags = 0
                self._winmm.waveOutPrepareHeader(self._hwave, ct.byref(hdr), ct.sizeof(WAVEHDR))
                self._buffers.append((raw_buf, hdr))
            self._initial_filled = 0
            return True
        except Exception:
            return False

    def _close_device(self):
        if not self._winmm_available or self._hwave is None:
            return
        try:
            self._winmm.waveOutReset(self._hwave)
            for raw_buf, hdr in self._buffers:
                self._winmm.waveOutUnprepareHeader(self._hwave, self._ctypes.byref(hdr), self._ctypes.sizeof(self._WAVEHDR))
            self._buffers.clear()
            self._winmm.waveOutClose(self._hwave)
            self._hwave = None
            if self._event is not None:
                self._kernel32.CloseHandle(self._event)
                self._event = None
        except Exception:
            pass

    def _write_pcm_block(self, samples: List[float], volume: float):
        if not self._buffers or self._hwave is None:
            return
        WHDR_DONE = 0x00000001
        num_bufs = len(self._buffers)
        if self._initial_filled < num_bufs:
            raw_buf, hdr = self._buffers[self._initial_filled]
            self._initial_filled += 1
        else:
            raw_buf, hdr = None, None
            for _ in range(6):
                for b, h in self._buffers:
                    if h.dwFlags & WHDR_DONE:
                        raw_buf, hdr = b, h
                        break
                if raw_buf is not None:
                    break
                self._kernel32.WaitForSingleObject(self._event, 50)
            if raw_buf is None:
                # All buffers are currently busy in audio hardware DMA; skip to avoid corruption
                return

        hdr.dwFlags &= ~WHDR_DONE
        pcm = bytearray(self.block_size * 2)
        v_scale = max(0.0, min(1.0, volume)) * 32767.0
        for i in range(self.block_size):
            s = samples[i] if i < len(samples) else 0.0
            val = max(-1.0, min(1.0, s))
            struct.pack_into('<h', pcm, i * 2, int(val * v_scale))
        self._ctypes.memmove(raw_buf, bytes(pcm), self.block_size * 2)
        self._winmm.waveOutWrite(self._hwave, self._ctypes.byref(hdr), self._ctypes.sizeof(self._WAVEHDR))

    def _audio_loop(self):
        self._pending_device_restart = False
        device_open = False
        if self.sound_enabled:
            device_open = self._open_device()

        while self._running:
            if getattr(self, '_pending_device_restart', False):
                self._pending_device_restart = False
                if device_open:
                    self._close_device()
                    device_open = False

            with self._filter_lock:
                if self._pending_filter is not None:
                    self.active_filter = self._pending_filter
                    self._pending_filter = None

            # Generate contiguous input block
            _, in_samples = self.generator.generate_block(self.block_size)

            # Process through active filter
            unstable = False
            if self.active_filter is not None:
                try:
                    out_samples = self.active_filter.process_block(in_samples)
                    for val in out_samples:
                        if math.isnan(val) or math.isinf(val) or abs(val) > 1e4:
                            unstable = True
                            break
                    if unstable:
                        self.active_filter.reset()
                        out_samples = [0.0] * len(in_samples)
                except Exception:
                    out_samples = list(in_samples)
            else:
                out_samples = list(in_samples)

            # Calculate metrics
            rms_in = math.sqrt(sum(x * x for x in in_samples) / max(1, len(in_samples)))
            rms_out = math.sqrt(sum(y * y for y in out_samples) / max(1, len(out_samples)))
            ratio = (rms_out / rms_in) if rms_in > 1e-9 else 1.0
            gain_db = 20.0 * math.log10(max(ratio, 1e-6))
            peak_out = max((abs(y) for y in out_samples), default=0.0)

            with self._snapshot_lock:
                self._recent_in = in_samples
                self._recent_out = out_samples
                self._metrics = {
                    "rms_in": rms_in,
                    "rms_out": rms_out,
                    "gain_db": gain_db,
                    "peak_out": peak_out,
                    "unstable": unstable
                }

            with self._vol_lock:
                snd_on = self.sound_enabled
                vol = self.volume

            if snd_on:
                if not device_open:
                    device_open = self._open_device()
                if device_open:
                    self._write_pcm_block(out_samples, vol)
                else:
                    time.sleep(self.block_size / self.sample_rate)
            else:
                if device_open:
                    self._close_device()
                    device_open = False
                time.sleep(self.block_size / self.sample_rate)

        if device_open:
            self._close_device()
