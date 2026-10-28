"""
Audio Filter Design DSP Engine
Supports Butterworth, Chebyshev Type I, Chebyshev Type II, and Bessel filters
for Low-pass (LPF), High-pass (HPF), Band-pass (BPF), and Band-stop/Notch (BSF).

Generates digital filter coefficients (Biquad Second-Order Sections and Direct Form)
using Bilinear Transform (BLT) with frequency pre-warping.
Pure Python standard library (math, cmath) - zero external dependencies required.
"""

import math
import cmath
from typing import List, Tuple, Dict, Any, Optional


class BiquadSection:
    """
    Second Order Section (Biquad):
    H(z) = (b0 + b1*z^-1 + b2*z^-2) / (a0 + a1*z^-1 + a2*z^-2)
    Normalized such that a0 = 1.0
    """
    def __init__(self, b0: float, b1: float, b2: float, a0: float, a1: float, a2: float):
        if abs(a0) < 1e-15:
            raise ValueError("a0 cannot be zero")
        self.b0 = b0 / a0
        self.b1 = b1 / a0
        self.b2 = b2 / a0
        self.a0 = 1.0
        self.a1 = a1 / a0
        self.a2 = a2 / a0

        # State registers for Direct Form II Transposed filtering
        self.s1 = 0.0
        self.s2 = 0.0

    def reset(self):
        self.s1 = 0.0
        self.s2 = 0.0

    def process_sample(self, x: float) -> float:
        """Process a single sample using Direct Form II Transposed structure."""
        y = self.b0 * x + self.s1
        self.s1 = self.b1 * x - self.a1 * y + self.s2
        self.s2 = self.b2 * x - self.a2 * y
        return y

    def response(self, w: float) -> complex:
        """
        Evaluate frequency response H(e^(jw)) where w is normalized radian frequency (0 to pi).
        """
        z_inv = cmath.exp(-1j * w)
        z_inv2 = z_inv * z_inv
        num = self.b0 + self.b1 * z_inv + self.b2 * z_inv2
        den = self.a0 + self.a1 * z_inv + self.a2 * z_inv2
        if abs(den) < 1e-15:
            return 1e10
        return num / den


def butterworth_analog_poles(order: int) -> List[complex]:
    """Calculate prototype analog low-pass poles for Butterworth filter (wc=1)."""
    poles = []
    for k in range(order):
        # theta = pi/2 + (2k + 1)*pi / (2*order)
        theta = math.pi * (2 * k + 1) / (2 * order) + math.pi / 2
        poles.append(cmath.rect(1.0, theta))
    return poles


def chebyshev1_analog_poles(order: int, ripple_db: float) -> List[complex]:
    """Calculate prototype analog low-pass poles for Chebyshev Type I (passband ripple_db, wc=1)."""
    if ripple_db <= 0:
        ripple_db = 0.5
    eps = math.sqrt(math.pow(10.0, ripple_db / 10.0) - 1.0)
    v0 = math.asinh(1.0 / eps) / order
    sinh_v0 = math.sinh(v0)
    cosh_v0 = math.cosh(v0)
    poles = []
    for k in range(order):
        theta = math.pi * (2 * k + 1) / (2 * order)
        sigma = -sinh_v0 * math.sin(theta)
        omega = cosh_v0 * math.cos(theta)
        poles.append(complex(sigma, omega))
    return poles


def chebyshev2_analog_poles_zeros(order: int, stopband_atten_db: float) -> Tuple[List[complex], List[complex]]:
    """
    Calculate prototype analog low-pass poles & zeros for Chebyshev Type II (stopband atten_db, wc=1).
    """
    if stopband_atten_db <= 0:
        stopband_atten_db = 40.0
    delta_s = math.pow(10.0, -stopband_atten_db / 20.0)
    eps = delta_s / math.sqrt(max(1.0 - delta_s * delta_s, 1e-10))
    v0 = math.asinh(1.0 / eps) / order
    sinh_v0 = math.sinh(v0)
    cosh_v0 = math.cosh(v0)

    poles = []
    zeros = []

    for k in range(order):
        theta = math.pi * (2 * k + 1) / (2 * order)
        sigma_p = -sinh_v0 * math.sin(theta)
        omega_p = cosh_v0 * math.cos(theta)
        p_ch1 = complex(sigma_p, omega_p)
        # Invert s -> 1/s to get Chebyshev II
        poles.append(1.0 / p_ch1)

        # Zeros are on imaginary axis where cos(theta) != 0
        cos_th = math.cos(theta)
        if abs(cos_th) > 1e-10:
            zeros.append(complex(0.0, 1.0 / cos_th))

    return poles, zeros


def bessel_reverse_polynomial(n: int) -> List[float]:
    """Reverse Bessel polynomial coefficients for order n (used for Bessel filter poles)."""
    # Recurrence: theta_{n}(s) = (2n-1) theta_{n-1}(s) + s^2 theta_{n-2}(s)
    # Roots can be computed by standard polynomial root finding or companion matrix.
    # Precomputed / computed roots:
    return []


def polynomial_roots_durand_kerner(poly: List[complex], max_iter: int = 150) -> List[complex]:
    """Find roots of polynomial sum(poly[i] * s^(n-i)) = 0 using Durand-Kerner method."""
    n = len(poly) - 1
    if n <= 0:
        return []
    # Normalize polynomial so leading coeff is 1
    a0 = poly[0]
    p = [c / a0 for c in poly]

    # Initial guesses
    roots = []
    r0 = complex(0.4, 0.9)
    for i in range(n):
        roots.append(r0 ** i)

    for _ in range(max_iter):
        max_diff = 0.0
        for i in range(n):
            # Evaluate p(roots[i])
            val = p[0]
            for c in p[1:]:
                val = val * roots[i] + c
            # Evaluate prod (roots[i] - roots[j])
            denom = 1.0 + 0j
            for j in range(n):
                if i != j:
                    denom *= (roots[i] - roots[j])
            if abs(denom) > 1e-15:
                delta = val / denom
                roots[i] -= delta
                if abs(delta) > max_diff:
                    max_diff = abs(delta)
        if max_diff < 1e-12:
            break
    return roots


def bessel_analog_poles(order: int) -> List[complex]:
    """Compute prototype analog low-pass poles for Bessel filter normalized for unit delay at w=0."""
    # Reverse Bessel polynomials B_n(s) = sum_{k=0}^n (2n-k)! / (2^(n-k) * k! * (n-k)!) * s^k
    coeffs = []
    for k in range(order, -1, -1):
        c = math.factorial(2 * order - k) // ((2 ** (order - k)) * math.factorial(k) * math.factorial(order - k))
        coeffs.append(complex(float(c), 0.0))

    roots = polynomial_roots_durand_kerner(coeffs)
    # Frequency scaling factor so that -3dB cutoff is near wc = 1
    # Scale poles so that H(0) has normalized delay
    scale_factor = (coeffs[-1].real / coeffs[0].real) ** (1.0 / order)
    # Alternatively, commonly Bessel poles are scaled so that gain at w=1 is 1/sqrt(2)
    # We will find the frequency where |H(jw)| = 1/sqrt(2)
    def mag(w: float) -> float:
        val = complex(1.0, 0.0)
        for r in roots:
            val *= (complex(0, w) - r)
        # H(jw) = B_n(0) / B_n(jw) = prod(-r) / prod(jw - r)
        h0 = 1.0
        for r in roots:
            h0 *= abs(r)
        return h0 / abs(val)

    # Simple bisection to find w3db
    w_low, w_high = 0.1, 5.0
    for _ in range(30):
        w_mid = 0.5 * (w_low + w_high)
        if mag(w_mid) > 1.0 / math.sqrt(2.0):
            w_low = w_mid
        else:
            w_high = w_mid
    w3db = 0.5 * (w_low + w_high)
    if w3db > 1e-5:
        roots = [r / w3db for r in roots]
    return roots


def pair_poles_and_zeros(poles: List[complex], zeros: List[complex]) -> List[Tuple[List[complex], List[complex]]]:
    """
    Group poles and zeros into conjugate pairs (and single real poles) to form 2nd-order sections.
    """
    # Sort poles by imaginary part >= 0, then by Q (proximity to imaginary axis)
    remaining_poles = list(poles)
    remaining_zeros = list(zeros)

    sections = []

    # Helper to find complex conjugate
    def find_conjugate(root_list: List[complex], r: complex, tol: float = 1e-4) -> Optional[int]:
        for idx, other in enumerate(root_list):
            if abs(other.real - r.real) < tol and abs(other.imag + r.imag) < tol:
                return idx
        return None

    pole_pairs: List[List[complex]] = []
    real_poles: List[complex] = []

    while remaining_poles:
        p = remaining_poles.pop(0)
        if abs(p.imag) < 1e-6:
            real_poles.append(complex(p.real, 0.0))
        else:
            cj_idx = find_conjugate(remaining_poles, p)
            if cj_idx is not None:
                p_cj = remaining_poles.pop(cj_idx)
                pole_pairs.append([p, p_cj])
            else:
                real_poles.append(complex(p.real, 0.0))

    # Pair single real poles together
    while len(real_poles) >= 2:
        p1 = real_poles.pop(0)
        p2 = real_poles.pop(0)
        pole_pairs.append([p1, p2])

    if len(real_poles) == 1:
        pole_pairs.append([real_poles.pop(0)])

    # Group zeros
    zero_pairs: List[List[complex]] = []
    real_zeros: List[complex] = []

    while remaining_zeros:
        z = remaining_zeros.pop(0)
        if abs(z.imag) < 1e-6:
            real_zeros.append(complex(z.real, 0.0))
        else:
            cj_idx = find_conjugate(remaining_zeros, z)
            if cj_idx is not None:
                z_cj = remaining_zeros.pop(cj_idx)
                zero_pairs.append([z, z_cj])
            else:
                real_zeros.append(complex(z.real, 0.0))

    while len(real_zeros) >= 2:
        z1 = real_zeros.pop(0)
        z2 = real_zeros.pop(0)
        zero_pairs.append([z1, z2])

    if len(real_zeros) == 1:
        zero_pairs.append([real_zeros.pop(0)])

    # Pair zero groups with pole groups
    for pp in pole_pairs:
        if zero_pairs:
            zz = zero_pairs.pop(0)
        else:
            zz = []
        sections.append((pp, zz))

    while zero_pairs:
        sections.append(([], zero_pairs.pop(0)))

    return sections


class DigitalFilter:
    """
    Digital Audio Filter composed of cascaded Biquad sections.
    """
    def __init__(self,
                 filter_type: str,       # 'LPF', 'HPF', 'BPF', 'BSF' (Notch)
                 response_type: str,     # 'Butterworth', 'Chebyshev I', 'Chebyshev II', 'Bessel'
                 order: int,             # 1 to 10
                 sample_rate: float,     # e.g., 44100, 48000
                 f_cutoff: float,        # Cutoff / Center frequency in Hz (LPF, HPF)
                 f_cutoff2: Optional[float] = None, # Second cutoff for BPF/BSF (f1, f2)
                 ripple_db: float = 1.0,            # Chebyshev I ripple
                 stopband_db: float = 40.0):        # Chebyshev II stopband attenuation
        self.filter_type = filter_type.upper()
        self.response_type = response_type
        self.order = max(1, min(12, int(order)))
        self.sample_rate = float(sample_rate)
        self.f_cutoff = float(f_cutoff)
        self.f_cutoff2 = float(f_cutoff2) if f_cutoff2 is not None else self.f_cutoff * 2.0
        self.ripple_db = float(ripple_db)
        self.stopband_db = float(stopband_db)

        self.sections: List[BiquadSection] = []
        self.gain: float = 1.0
        self.design()

    def design(self):
        """Perform analog prototype synthesis, frequency transformation, and Bilinear Transform."""
        sr = self.sample_rate
        nyquist = sr / 2.0

        # Validate frequency bounds
        f1 = max(10.0, min(self.f_cutoff, nyquist * 0.99))
        f2 = max(10.0, min(self.f_cutoff2, nyquist * 0.99))
        if f1 > f2:
            f1, f2 = f2, f1

        # 1. Analog Low-Pass Prototype Poles & Zeros
        n = self.order
        if self.response_type == 'Butterworth':
            proto_poles = butterworth_analog_poles(n)
            proto_zeros = []
        elif self.response_type == 'Chebyshev I':
            proto_poles = chebyshev1_analog_poles(n, self.ripple_db)
            proto_zeros = []
        elif self.response_type == 'Chebyshev II':
            proto_poles, proto_zeros = chebyshev2_analog_poles_zeros(n, self.stopband_db)
        elif self.response_type == 'Bessel':
            proto_poles = bessel_analog_poles(n)
            proto_zeros = []
        else:
            proto_poles = butterworth_analog_poles(n)
            proto_zeros = []

        # 2. Pre-warped analog cutoffs: w_a = 2 * fs * tan(pi * f / fs)
        # Bilinear parameter c = 2 * fs
        c = 2.0 * sr

        self.sections = []

        if self.filter_type == 'LPF':
            wa = 2.0 * sr * math.tan(math.pi * f1 / sr)
            # Transform prototype s -> s / wa
            # Pole p_transformed = p * wa, zero z_transformed = z * wa
            poles = [p * wa for p in proto_poles]
            zeros = [z * wa for z in proto_zeros]
            # Analog poles/zeros to discrete sections via Bilinear Transform
            self._bilinear_transform_to_sections(poles, zeros, c, dc_gain=1.0, eval_freq=0.0)

        elif self.filter_type == 'HPF':
            wa = 2.0 * sr * math.tan(math.pi * f1 / sr)
            # Transform prototype s -> wa / s
            # Pole p_transformed = wa / p
            poles = [wa / p for p in proto_poles]
            zeros = [wa / z for z in proto_zeros]
            # Add zeros at s = 0 for each pole not matched by an analog zero
            num_zeros_at_origin = len(proto_poles) - len(proto_zeros)
            zeros.extend([0.0 + 0j] * num_zeros_at_origin)
            self._bilinear_transform_to_sections(poles, zeros, c, dc_gain=1.0, eval_freq=nyquist)

        elif self.filter_type in ('BPF', 'BSF'):
            wa1 = 2.0 * sr * math.tan(math.pi * f1 / sr)
            wa2 = 2.0 * sr * math.tan(math.pi * f2 / sr)
            w0 = math.sqrt(wa1 * wa2)
            bw = max(wa2 - wa1, 1e-5)

            if self.filter_type == 'BPF':
                # s_proto -> (s^2 + w0^2) / (bw * s)
                # For each prototype pole/zero p: solve s^2 - bw*p*s + w0^2 = 0
                poles = []
                for p in proto_poles:
                    b_term = bw * p
                    disc = cmath.sqrt(b_term * b_term - 4.0 * (w0 * w0))
                    poles.append(0.5 * (b_term + disc))
                    poles.append(0.5 * (b_term - disc))

                zeros = []
                for z in proto_zeros:
                    b_term = bw * z
                    disc = cmath.sqrt(b_term * b_term - 4.0 * (w0 * w0))
                    zeros.append(0.5 * (b_term + disc))
                    zeros.append(0.5 * (b_term - disc))

                # Also zeros at origin for each original proto pole
                num_zeros_at_origin = len(proto_poles) - len(proto_zeros)
                zeros.extend([0.0 + 0j] * num_zeros_at_origin)

                center_hz = math.sqrt(f1 * f2)
                self._bilinear_transform_to_sections(poles, zeros, c, dc_gain=1.0, eval_freq=center_hz)

            else:  # BSF / Notch
                # s_proto -> bw * s / (s^2 + w0^2)
                # s^2 - (bw/p)*s + w0^2 = 0
                poles = []
                for p in proto_poles:
                    b_term = bw / p
                    disc = cmath.sqrt(b_term * b_term - 4.0 * (w0 * w0))
                    poles.append(0.5 * (b_term + disc))
                    poles.append(0.5 * (b_term - disc))

                zeros = []
                for z in proto_zeros:
                    b_term = bw / z
                    disc = cmath.sqrt(b_term * b_term - 4.0 * (w0 * w0))
                    zeros.append(0.5 * (b_term + disc))
                    zeros.append(0.5 * (b_term - disc))

                # Add zeros on imaginary axis at +- j*w0 for each proto pole not matched
                num_zeros_at_w0 = len(proto_poles) - len(proto_zeros)
                for _ in range(num_zeros_at_w0):
                    zeros.append(complex(0.0, w0))
                    zeros.append(complex(0.0, -w0))

                self._bilinear_transform_to_sections(poles, zeros, c, dc_gain=1.0, eval_freq=0.0)

    def _bilinear_transform_to_sections(self, poles: List[complex], zeros: List[complex],
                                        c: float, dc_gain: float, eval_freq: float):
        """
        Convert analog s-plane poles and zeros to discrete z-plane biquad sections using s = c * (z - 1)/(z + 1).
        z = (c + s) / (c - s).
        """
        # Map each pole to z-plane: z_p = (c + s_p) / (c - s_p)
        z_poles = []
        for p in poles:
            denom = c - p
            if abs(denom) < 1e-15:
                denom = 1e-15
            z_poles.append((c + p) / denom)

        # Map each zero to z-plane: z_z = (c + s_z) / (c - s_z)
        z_zeros = []
        for z in zeros:
            denom = c - z
            if abs(denom) < 1e-15:
                denom = 1e-15
            z_zeros.append((c + z) / denom)

        # Pad with zeros at z = -1 (which correspond to s = infinity) to match pole count
        while len(z_zeros) < len(z_poles):
            z_zeros.append(complex(-1.0, 0.0))

        # Group into 2nd order sections
        pairs = pair_poles_and_zeros(z_poles, z_zeros)
        raw_sections = []

        for p_list, z_list in pairs:
            # Construct (1 - z_1*z^-1)(1 - z_2*z^-1) = 1 - (z_1+z_2)z^-1 + (z1*z2)z^-2
            if len(p_list) == 2:
                p1, p2 = p_list[0], p_list[1]
                a0 = 1.0
                a1 = -(p1 + p2).real
                a2 = (p1 * p2).real
            elif len(p_list) == 1:
                p1 = p_list[0]
                a0 = 1.0
                a1 = -p1.real
                a2 = 0.0
            else:
                a0, a1, a2 = 1.0, 0.0, 0.0

            if len(z_list) == 2:
                z1, z2 = z_list[0], z_list[1]
                b0 = 1.0
                b1 = -(z1 + z2).real
                b2 = (z1 * z2).real
            elif len(z_list) == 1:
                z1 = z_list[0]
                b0 = 1.0
                b1 = -z1.real
                b2 = 0.0
            else:
                b0, b1, b2 = 1.0, 0.0, 0.0

            raw_sections.append(BiquadSection(b0, b1, b2, a0, a1, a2))

        # Calculate overall gain normalizer at eval_freq
        w_eval = 2.0 * math.pi * eval_freq / self.sample_rate
        total_resp = 1.0 + 0j
        for sec in raw_sections:
            total_resp *= sec.response(w_eval)

        gain_scale = 1.0
        if abs(total_resp) > 1e-12:
            target_gain = 1.0
            # For Chebyshev I, if order is even, DC gain is 1 / sqrt(1 + eps^2) = -ripple_db dB
            if self.response_type == 'Chebyshev I' and (self.order % 2 == 0) and (self.filter_type == 'LPF'):
                target_gain = math.pow(10.0, -self.ripple_db / 20.0)
            gain_scale = target_gain / abs(total_resp)

        # Distribute gain evenly or apply to the first section
        if raw_sections:
            raw_sections[0].b0 *= gain_scale
            raw_sections[0].b1 *= gain_scale
            raw_sections[0].b2 *= gain_scale

        self.sections = raw_sections

    def reset(self):
        """Reset state of all filter sections."""
        for sec in self.sections:
            sec.reset()

    def process_sample(self, x: float) -> float:
        """Process one sample through cascaded biquads."""
        out = x
        for sec in self.sections:
            out = sec.process_sample(out)
        return out

    def process_block(self, samples: List[float]) -> List[float]:
        """Process an array of audio samples."""
        output = [0.0] * len(samples)
        for i, sample in enumerate(samples):
            output[i] = self.process_sample(sample)
        return output

    def frequency_response(self, freqs: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Evaluate magnitude (dB), phase (degrees), and group delay for a list of frequencies (Hz).
        """
        mag_db = []
        phase_deg = []
        group_delays = []

        dw = 1e-4

        for f in freqs:
            w = 2.0 * math.pi * f / self.sample_rate
            h = 1.0 + 0j
            for sec in self.sections:
                h *= sec.response(w)

            # Magnitude
            mag = abs(h)
            mag_db_val = 20.0 * math.log10(max(mag, 1e-12))
            mag_db.append(mag_db_val)

            # Phase
            phase = math.atan2(h.imag, h.real)
            phase_deg.append(math.degrees(phase))

            # Numerical group delay: -d(phi)/dw
            w_plus = w + dw
            h_plus = 1.0 + 0j
            for sec in self.sections:
                h_plus *= sec.response(w_plus)
            phase_plus = math.atan2(h_plus.imag, h_plus.real)
            # Unwrap phase step
            d_phi = phase_plus - phase
            while d_phi > math.pi:
                d_phi -= 2 * math.pi
            while d_phi < -math.pi:
                d_phi += 2 * math.pi
            gd = -d_phi / dw
            group_delays.append(gd)

        return mag_db, phase_deg, group_delays

    def get_direct_form_coefficients(self) -> Tuple[List[float], List[float]]:
        """
        Expand all cascaded biquads into single Direct Form polynomial coefficients [B], [A].
        B(z) = b0 + b1*z^-1 + ...
        A(z) = 1 + a1*z^-1 + ...
        """
        b_poly = [1.0]
        a_poly = [1.0]

        for sec in self.sections:
            b_sec = [sec.b0, sec.b1, sec.b2]
            a_sec = [sec.a0, sec.a1, sec.a2]

            # Polynomial convolution
            b_new = [0.0] * (len(b_poly) + len(b_sec) - 1)
            for i, bp in enumerate(b_poly):
                for j, bs in enumerate(b_sec):
                    b_new[i + j] += bp * bs
            b_poly = b_new

            a_new = [0.0] * (len(a_poly) + len(a_sec) - 1)
            for i, ap in enumerate(a_poly):
                for j, as_ in enumerate(a_sec):
                    a_new[i + j] += ap * as_
            a_poly = a_new

        return b_poly, a_poly

    def generate_c_code(self, filter_var_name: str = "audio_filter") -> str:
        """
        Generate standalone ANSI C99 code for this filter (Second-Order Sections Biquad cascade).
        Does not require ANY external libraries (only standard C math / types).
        """
        num_sections = len(self.sections)
        c_code = []
        c_code.append("/*")
        c_code.append(f" * Generated C Audio Filter: {self.response_type} {self.filter_type}")
        c_code.append(f" * Order: {self.order}, Sample Rate: {self.sample_rate:.1f} Hz")
        if self.filter_type in ('LPF', 'HPF'):
            c_code.append(f" * Cutoff: {self.f_cutoff:.2f} Hz")
        else:
            c_code.append(f" * Cutoff 1: {self.f_cutoff:.2f} Hz, Cutoff 2: {self.f_cutoff2:.2f} Hz")
        if self.response_type == 'Chebyshev I':
            c_code.append(f" * Passband Ripple: {self.ripple_db:.2f} dB")
        elif self.response_type == 'Chebyshev II':
            c_code.append(f" * Stopband Attenuation: {self.stopband_db:.2f} dB")
        c_code.append(" * Standalone ANSI C99 - Zero external dependencies")
        c_code.append(" */\n")
        c_code.append("#include <stddef.h>\n")

        c_code.append("/* Biquad Second-Order Section (Direct Form II Transposed) */")
        c_code.append("typedef struct {")
        c_code.append("    float b0, b1, b2;")
        c_code.append("    float a1, a2;")
        c_code.append("    float s1, s2; /* internal state registers */")
        c_code.append("} BiquadSection;\n")

        c_code.append(f"#define {filter_var_name.upper()}_NUM_SECTIONS {num_sections}\n")

        c_code.append("typedef struct {")
        c_code.append(f"    BiquadSection sections[{filter_var_name.upper()}_NUM_SECTIONS];")
        c_code.append("} AudioFilter;\n")

        c_code.append(f"static void {filter_var_name}_init(AudioFilter *f) {{")
        for i, sec in enumerate(self.sections):
            c_code.append(f"    /* Section {i + 1} */")
            c_code.append(f"    f->sections[{i}].b0 = {sec.b0:.9e}f;")
            c_code.append(f"    f->sections[{i}].b1 = {sec.b1:.9e}f;")
            c_code.append(f"    f->sections[{i}].b2 = {sec.b2:.9e}f;")
            c_code.append(f"    f->sections[{i}].a1 = {sec.a1:.9e}f;")
            c_code.append(f"    f->sections[{i}].a2 = {sec.a2:.9e}f;")
            c_code.append(f"    f->sections[{i}].s1 = 0.0f;")
            c_code.append(f"    f->sections[{i}].s2 = 0.0f;")
        c_code.append("}\n")

        c_code.append(f"static void {filter_var_name}_reset(AudioFilter *f) {{")
        c_code.append(f"    for (size_t i = 0; i < {filter_var_name.upper()}_NUM_SECTIONS; ++i) {{")
        c_code.append("        f->sections[i].s1 = 0.0f;")
        c_code.append("        f->sections[i].s2 = 0.0f;")
        c_code.append("    }")
        c_code.append("}\n")

        c_code.append(f"static float {filter_var_name}_process_sample(AudioFilter *f, float in_sample) {{")
        c_code.append("    float acc = in_sample;")
        c_code.append(f"    for (size_t i = 0; i < {filter_var_name.upper()}_NUM_SECTIONS; ++i) {{")
        c_code.append("        BiquadSection *sec = &f->sections[i];")
        c_code.append("        float y = sec->b0 * acc + sec->s1;")
        c_code.append("        sec->s1 = sec->b1 * acc - sec->a1 * y + sec->s2;")
        c_code.append("        sec->s2 = sec->b2 * acc - sec->a2 * y;")
        c_code.append("        acc = y;")
        c_code.append("    }")
        c_code.append("    return acc;")
        c_code.append("}\n")

        c_code.append(f"static void {filter_var_name}_process_block(AudioFilter *f, const float *in, float *out, size_t count) {{")
        c_code.append("    for (size_t n = 0; n < count; ++n) {")
        c_code.append(f"        out[n] = {filter_var_name}_process_sample(f, in[n]);")
        c_code.append("    }")
        c_code.append("}\n")

        # Optional runnable self-test main if compiled as standalone executable
        c_code.append("#ifdef TEST_FILTER_STANDALONE")
        c_code.append("#include <stdio.h>")
        c_code.append("int main(void) {")
        c_code.append("    AudioFilter filter;")
        c_code.append(f"    {filter_var_name}_init(&filter);")
        c_code.append("    printf(\"Filter initialized successfully! Running impulse response test:\\n\");")
        c_code.append("    for (int i = 0; i < 16; ++i) {")
        c_code.append("        float x = (i == 0) ? 1.0f : 0.0f;")
        c_code.append(f"        float y = {filter_var_name}_process_sample(&filter, x);")
        c_code.append("        printf(\"y[%2d] = %+.6f\\n\", i, y);")
        c_code.append("    }")
        c_code.append("    return 0;")
        c_code.append("}")
        c_code.append("#endif\n")

        return "\n".join(c_code)
