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
import io
import wave

from filter_engine import (DigitalFilter, FIRFilter, butterworth_analog_poles,
                           chebyshev1_analog_poles, chebyshev2_analog_poles_zeros,
                           bessel_analog_poles, BiquadSection)
from signal_testbench import (generate_signal, compute_fft, export_data_to_csv,
                              export_data_to_json, export_audio_to_wav, audio_to_wav_bytes,
                              ContinuousSignalGenerator, ContinuousAudioEngine)


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

    def test_filter_clone(self):
        filt = DigitalFilter('LPF', 'Butterworth', order=4, sample_rate=48000, f_cutoff=1200)
        cloned = filt.clone()
        self.assertEqual(len(cloned.sections), len(filt.sections))
        self.assertEqual(cloned.order, filt.order)
        self.assertEqual(cloned.sample_rate, filt.sample_rate)
        # Modify cloned state and verify original is untouched
        cloned.sections[0].s1 = 42.0
        self.assertEqual(filt.sections[0].s1, 0.0)

    def test_fir_filter_design_and_processing(self):
        # Design FIR LPF 65 taps
        fir = FIRFilter('LPF', 65, 48000, 1000, window_name='Hamming')
        self.assertEqual(fir.num_taps, 65)
        self.assertEqual(fir.order, 64)
        self.assertEqual(fir.filter_family, 'FIR')

        # Linear phase property: Group delay must be exactly (N-1)/2 samples = 32.0
        freqs = [10.0, 500.0, 1000.0, 8000.0]
        mags, phases, gd = fir.frequency_response(freqs)
        for g in gd:
            self.assertAlmostEqual(g, 32.0, places=4)

        # Passband gain ~ 0 dB, stopband strong attenuation
        self.assertAlmostEqual(mags[0], 0.0, delta=0.5)
        self.assertLess(mags[-1], -40.0)

        # Test block processing
        samples_in = [0.1] * 128
        out = fir.process_block(samples_in)
        self.assertEqual(len(out), 128)

        # Test clone
        cloned_fir = fir.clone()
        self.assertEqual(cloned_fir.num_taps, 65)
        self.assertEqual(cloned_fir.taps, fir.taps)

        # Test C code generation
        c_code = fir.generate_c_code("test_fir")
        self.assertIn("FIRFilter", c_code)
        self.assertIn("test_fir_process_sample", c_code)
        self.assertIn("test_fir_init", c_code)

    def test_fir_filter_types(self):
        for ft in ['LPF', 'HPF', 'BPF', 'BSF']:
            fir = FIRFilter(ft, 45, 48000, 1000, f_cutoff2=4000, window_name='Hann')
            self.assertEqual(fir.filter_type, ft)
            self.assertEqual(fir.num_taps, 45)
            out = fir.process_block([0.5, -0.5, 0.2])
            self.assertEqual(len(out), 3)

        # Test FIR BPF frequency response: with sufficient taps (e.g. 255 taps for 257 Hz),
        # both stopbands (below fc1 and above fc2) are deeply attenuated while passband is ~0 dB.
        fir_bpf = FIRFilter('BPF', 255, 48000, 257.2, f_cutoff2=4000, window_name='Hamming')
        mags, _, _ = fir_bpf.frequency_response([10.0, 1000.0, 12000.0])
        self.assertLess(mags[0], -20.0, f"BPF low stopband gain should be attenuated, got {mags[0]:.1f} dB")
        self.assertGreater(mags[1], -1.0, f"Passband gain should be near 0 dB, got {mags[1]:.1f} dB")
        self.assertLess(mags[2], -20.0, f"BPF high stopband gain should be attenuated, got {mags[2]:.1f} dB")

        # Test FIR HPF rejects low frequencies
        fir_hpf = FIRFilter('HPF', 129, 48000, 1000.0, window_name='Hamming')
        mags_hpf, _, _ = fir_hpf.frequency_response([10.0, 5000.0])
        self.assertLess(mags_hpf[0], -20.0)
        self.assertGreater(mags_hpf[1], -1.0)

    def test_fir_comprehensive_windows(self):
        for w in FIRFilter.KNOWN_WINDOWS:
            fir = FIRFilter('LPF', 33, 48000, 1200, window_name=w,
                            kaiser_beta=6.0, gaussian_sigma=0.35, tukey_alpha=0.4)
            self.assertEqual(len(fir.taps), 33)
            # Verify finite and non-NaN values
            for tap in fir.taps:
                self.assertFalse(math.isnan(tap))
                self.assertFalse(math.isinf(tap))
            mags, phases, gd = fir.frequency_response([100.0, 1200.0, 5000.0])
            self.assertEqual(len(mags), 3)
            # Verify linear phase constant group delay
            for g in gd:
                self.assertAlmostEqual(g, 16.0, places=4)

    def test_check_criteria_match(self):
        # FIR with 65 taps at 45.1 Hz cannot resolve cutoff -> Mismatch expected
        fir_unresolved = FIRFilter('LPF', 65, 48000, 45.1, window_name='Hamming')
        matched, reason = fir_unresolved.check_criteria_match(45.1)
        self.assertFalse(matched)
        self.assertTrue("Under-resolved" in reason or "exceeds cutoff" in reason)

        # FIR with 65 taps at 2000 Hz is well resolved -> Satisfied expected
        fir_resolved = FIRFilter('LPF', 65, 48000, 2000.0, window_name='Hamming')
        matched, reason = fir_resolved.check_criteria_match(2000.0)
        self.assertTrue(matched)
        self.assertEqual(reason, "")

        # IIR at 45.1 Hz has feedback poles and can easily resolve 45 Hz -> Satisfied expected
        iir_resolved = DigitalFilter('LPF', 'Butterworth', 4, 48000, 45.1)
        matched, reason = iir_resolved.check_criteria_match(45.1)
        self.assertTrue(matched)
        self.assertEqual(reason, "")


    def test_chebyshev2_filter_cutoff(self):
        # 4th-order Chebyshev II LPF at 1000 Hz, 40 dB stopband attenuation
        filt = DigitalFilter('LPF', 'Chebyshev II', order=4, sample_rate=48000, f_cutoff=1000, stopband_db=40)
        freqs = [100.0, 1000.0, 4000.0]
        mags, _, _ = filt.frequency_response(freqs)
        # Passband should be ~0 dB
        self.assertAlmostEqual(mags[0], 0.0, delta=0.2)
        # At cutoff, gain must be ~ -3.0 dB
        self.assertAlmostEqual(mags[1], -3.01, delta=0.5)
        # Stopband must be strongly attenuated
        self.assertLess(mags[2], -39.0)

    def test_chebyshev1_even_order_gain_normalization(self):
        # Even-order Chebyshev I for HPF, BPF, and BSF must not exceed 0 dB in the passband
        for ftype in ['HPF', 'BPF', 'BSF']:
            filt = DigitalFilter(ftype, 'Chebyshev I', order=4, sample_rate=48000,
                                 f_cutoff=1000, f_cutoff2=4000, ripple_db=2.0)
            probe_freqs = [f for f in range(50, 23500, 100)]
            mags, _, _ = filt.frequency_response(probe_freqs)
            self.assertLessEqual(max(mags), 0.05, f"Filter {ftype} exceeded 0 dB: max={max(mags):.2f}")

    def test_iir_criteria_match_near_nyquist(self):
        # Cutoff at 15 kHz is well below Nyquist * 0.98 (23.52 kHz), should not trigger false warning
        filt = DigitalFilter('LPF', 'Butterworth', order=4, sample_rate=48000, f_cutoff=15000)
        matched, reason = filt.check_criteria_match(15000)
        self.assertTrue(matched, f"Unexpected mismatch: {reason}")

        # Cutoff at 23.6 kHz is too close to Nyquist (24 kHz)
        matched, reason = filt.check_criteria_match(23600)
        self.assertFalse(matched)
        self.assertIn("too close to Nyquist", reason)

    def test_fir_narrowband_peak_normalization(self):
        # Narrowband BPF with 511 taps must have maximum passband gain <= 0.05 dB
        fir = FIRFilter('BPF', 511, 48000, 1000, f_cutoff2=1060, window_name='Hamming')
        probe_freqs = [980.0, 1000.0, 1030.0, 1060.0, 1080.0]
        mags, _, _ = fir.frequency_response(probe_freqs)
        self.assertLessEqual(max(mags), 0.05, f"FIR BPF exceeded 0 dB: {max(mags):.2f} dB")

class TestSignalTestbench(unittest.TestCase):
    def test_audio_to_wav_bytes_applies_volume(self):
        wav_data = audio_to_wav_bytes([0.5], 48000, volume=0.5)
        with wave.open(io.BytesIO(wav_data), 'rb') as wav_file:
            self.assertEqual(wav_file.getframerate(), 48000)
            self.assertEqual(wav_file.readframes(1), b'\xff\x1f')

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

    def test_continuous_signal_generator(self):
        gen = ContinuousSignalGenerator(sample_rate=48000)
        types = ['Sine', 'Multi-tone', 'Square', 'Triangle', 'Sawtooth',
                 'White Noise', 'Pink Noise', 'Periodic Noise', 'Sweep (Log)', 'Sweep (Linear)', 'Impulse']
        for st in types:
            gen.set_params(signal_type=st, freq1=600.0, freq2=3000.0, amplitude=0.7)
            t_block, samples = gen.generate_block(512)
            self.assertEqual(len(samples), 512)
            self.assertEqual(len(t_block), 512)
            # Verify amplitude bound
            for s in samples:
                self.assertLessEqual(abs(s), 1.0)

    def test_continuous_audio_engine(self):
        engine = ContinuousAudioEngine(sample_rate=48000, block_size=512)
        engine.set_volume(0.6)
        self.assertEqual(engine.volume, 0.6)
        engine.set_sound_enabled(False) # Test without opening audio output device
        self.assertFalse(engine.sound_enabled)

        filt = DigitalFilter('LPF', 'Butterworth', order=4, sample_rate=48000, f_cutoff=1000)
        engine.set_filter(filt)
        engine.set_signal_params(signal_type="Sine", freq1=440.0)

        engine.start()
        import time
        time.sleep(0.08)

        in_s, out_s, metrics = engine.get_snapshot()
        self.assertEqual(len(in_s), 512)
        self.assertEqual(len(out_s), 512)
        self.assertIn("rms_in", metrics)
        self.assertIn("rms_out", metrics)
        self.assertIn("gain_db", metrics)

        # Update filter live
        filt2 = DigitalFilter('HPF', 'Butterworth', order=4, sample_rate=48000, f_cutoff=2000)
        engine.set_filter(filt2)
        time.sleep(0.05)

        engine.stop()


    def test_periodic_noise_signal_generation(self):
        t, x = generate_signal('periodic noise', duration_sec=0.05, sample_rate=48000, freq1=500.0)
        self.assertEqual(len(t), int(0.05 * 48000))
        self.assertEqual(len(x), int(0.05 * 48000))
        period = int(48000 / 500)
        # Check periodicity
        for i in range(period, period * 3):
            self.assertAlmostEqual(x[i], x[i % period], places=6)

if __name__ == "__main__":
    unittest.main()
