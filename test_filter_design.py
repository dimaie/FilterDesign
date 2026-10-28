"""
Comprehensive automated unit tests for Audio Filter Design Studio:
- Filter prototype pole-zero calculations
- Bilinear transform frequency prewarping & mapping
- Filter responses: Butterworth, Chebyshev I, Chebyshev II, Bessel
- Filter types: LPF, HPF, BPF, BSF
- Signal generation and processing
- Standalone C code syntax & compilation test (if gcc/cl available) or validation
- Data export (CSV, JSON, WAV)
"""

import os
import math
import unittest
import tempfile
import json
import csv
import wave

from filter_engine import (DigitalFilter, butterworth_analog_poles,
                           chebyshev1_analog_poles, chebyshev2_analog_poles_zeros,
                           bessel_analog_poles, BiquadSection)
from signal_testbench import (generate_signal, compute_fft, export_data_to_csv,
                              export_data_to_json, export_audio_to_wav)


class TestFilterEngine(unittest.TestCase):
    def test_butterworth_poles(self):
        poles = butterworth_analog_poles(4)
        self.assertEqual(len(poles), 4)
        for p in poles:
            # Analog poles must be in left half-plane
            self.assertLess(p.real, 0.0)
            self.assertAlmostEqual(abs(p), 1.0, places=5)

    def test_chebyshev1_poles(self):
        poles = chebyshev1_analog_poles(4, ripple_db=1.0)
        self.assertEqual(len(poles), 4)
        for p in poles:
            self.assertLess(p.real, 0.0)

    def test_chebyshev2_poles_zeros(self):
        poles, zeros = chebyshev2_analog_poles_zeros(4, stopband_atten_db=40.0)
        self.assertEqual(len(poles), 4)
        for p in poles:
            self.assertLess(p.real, 0.0)
        for z in zeros:
            self.assertAlmostEqual(z.real, 0.0, places=5)

    def test_bessel_poles(self):
        poles = bessel_analog_poles(4)
        self.assertEqual(len(poles), 4)
        for p in poles:
            self.assertLess(p.real, 0.0)

    def test_lpf_filter_gain(self):
        # Butterworth 4th-order LPF at 1000 Hz, fs=48000
        filt = DigitalFilter('LPF', 'Butterworth', order=4, sample_rate=48000, f_cutoff=1000)
        freqs = [10.0, 1000.0, 10000.0]
        mags, phases, gd = filt.frequency_response(freqs)
        # DC / passband should be ~ 0 dB
        self.assertAlmostEqual(mags[0], 0.0, delta=0.2)
        # At cutoff w3db, should be ~ -3 dB
        self.assertAlmostEqual(mags[1], -3.01, delta=0.5)
        # In stopband (10 kHz), should be strongly attenuated
        self.assertLess(mags[2], -40.0)

    def test_hpf_filter_gain(self):
        # Butterworth 4th-order HPF at 1000 Hz, fs=48000
        filt = DigitalFilter('HPF', 'Butterworth', order=4, sample_rate=48000, f_cutoff=1000)
        freqs = [50.0, 1000.0, 15000.0]
        mags, phases, gd = filt.frequency_response(freqs)
        # Low frequency should be strongly attenuated
        self.assertLess(mags[0], -40.0)
        # At cutoff, ~ -3 dB
        self.assertAlmostEqual(mags[1], -3.01, delta=0.5)
        # High frequency passband should be ~ 0 dB
        self.assertAlmostEqual(mags[2], 0.0, delta=0.2)

    def test_bpf_filter_gain(self):
        # BPF from 1000 Hz to 4000 Hz
        filt = DigitalFilter('BPF', 'Butterworth', order=2, sample_rate=48000, f_cutoff=1000, f_cutoff2=4000)
        center_f = math.sqrt(1000.0 * 4000.0)
        freqs = [100.0, center_f, 18000.0]
        mags, phases, gd = filt.frequency_response(freqs)
        self.assertLess(mags[0], -20.0)
        self.assertAlmostEqual(mags[1], 0.0, delta=0.5)
        self.assertLess(mags[2], -20.0)

    def test_bsf_notch_filter_gain(self):
        # Notch from 1000 Hz to 2000 Hz
        filt = DigitalFilter('BSF', 'Butterworth', order=2, sample_rate=48000, f_cutoff=1000, f_cutoff2=2000)
        center_f = math.sqrt(1000.0 * 2000.0)
        freqs = [50.0, center_f, 20000.0]
        mags, phases, gd = filt.frequency_response(freqs)
        self.assertAlmostEqual(mags[0], 0.0, delta=0.5)
        self.assertLess(mags[1], -30.0)
        self.assertAlmostEqual(mags[2], 0.0, delta=0.5)

    def test_frequency_response_phase_and_delay(self):
        filt = DigitalFilter('LPF', 'Butterworth', order=4, sample_rate=48000, f_cutoff=1000)
        freqs = [10.0, 1000.0, 10000.0]
        mags, phases, gd = filt.frequency_response(freqs)
        self.assertEqual(len(phases), len(freqs))
        self.assertEqual(len(gd), len(freqs))
        # Low frequency phase should be close to 0 deg
        self.assertAlmostEqual(phases[0], 0.0, delta=2.0)
        # At cutoff for 4th-order Butterworth LPF, phase is -180 deg
        self.assertAlmostEqual(abs(phases[1]), 180.0, delta=5.0)
        # Group delay should be positive
        for g in gd:
            self.assertGreater(g, 0.0)

    def test_c_code_generation(self):
        filt = DigitalFilter('LPF', 'Chebyshev I', order=4, sample_rate=48000, f_cutoff=1500, ripple_db=0.5)
        c_code = filt.generate_c_code("test_filter")
        self.assertIn("typedef struct", c_code)
        self.assertIn("test_filter_process_sample", c_code)
        self.assertIn("test_filter_init", c_code)
        self.assertIn("BiquadSection", c_code)


class TestSignalTestbench(unittest.TestCase):
    def test_signal_generation_and_fft(self):
        for sig_name in ['sine', 'multi-tone', 'sweep (log)', 'sweep (linear)',
                         'square', 'triangle', 'sawtooth', 'white noise', 'impulse']:
            t, x = generate_signal(sig_name, duration_sec=0.01, sample_rate=48000, freq1=1000)
            self.assertEqual(len(t), 480)
            self.assertEqual(len(x), 480)

        f, db = compute_fft(x, 48000, max_points=256)
        self.assertGreater(len(f), 0)
        self.assertEqual(len(f), len(db))

    def test_export_utilities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "test.csv")
            json_path = os.path.join(tmpdir, "test.json")
            wav_path = os.path.join(tmpdir, "test.wav")

            t = [0.0, 0.01, 0.02]
            x = [0.0, 0.5, -0.5]
            y = [0.0, 0.4, -0.4]

            export_data_to_csv(csv_path, t, x, y)
            self.assertTrue(os.path.exists(csv_path))

            export_data_to_json(json_path, {"test": 123}, t, x, y)
            self.assertTrue(os.path.exists(json_path))

            export_audio_to_wav(wav_path, y, 48000)
            self.assertTrue(os.path.exists(wav_path))
            with wave.open(wav_path, 'r') as wf:
                self.assertEqual(wf.getnchannels(), 1)
                self.assertEqual(wf.getsampwidth(), 2)
                self.assertEqual(wf.getframerate(), 48000)


if __name__ == "__main__":
    unittest.main()
