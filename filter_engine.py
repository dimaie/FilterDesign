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

    def clone(self) -> 'BiquadSection':
        sec = BiquadSection(self.b0, self.b1, self.b2, 1.0, self.a1, self.a2)
        sec.s1 = self.s1
        sec.s2 = self.s2
        return sec

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
        self.filter_family = 'IIR'
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

    def clone(self) -> 'DigitalFilter':
        """Create a clone of this digital filter with independent section states."""
        f = DigitalFilter.__new__(DigitalFilter)
        f.filter_family = getattr(self, 'filter_family', 'IIR')
        f.filter_type = self.filter_type
        f.response_type = self.response_type
        f.order = self.order
        f.sample_rate = self.sample_rate
        f.f_cutoff = self.f_cutoff
        f.f_cutoff2 = self.f_cutoff2
        f.ripple_db = self.ripple_db
        f.stopband_db = self.stopband_db
        f.gain = self.gain
        f.sections = [sec.clone() for sec in self.sections]
        return f

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


class FIRFilter:
    """
    Linear-Phase Finite Impulse Response (FIR) Filter.
    Designed using windowed-sinc method (Type I, symmetric odd-length kernel).
    Features exact linear phase and constant group delay = (N - 1) / 2 samples.
    Zero feedback, guaranteed unconditionally stable.
    Supports windows:
    - Rectangular
    - Hamming (default)
    - Hann (Hanning)
    - Blackman (high stopband attenuation ~74 dB)
    - Kaiser (adjustable beta parameter)
    - Bartlett (triangular)
    """
    def __init__(self,
                 filter_type: str,            # 'LPF', 'HPF', 'BPF', 'BSF' (Notch)
                 num_taps: int,               # Number of taps / filter length (odd integer)
                 sample_rate: float,          # Sampling rate (Hz)
                 f_cutoff: float,             # Cutoff frequency 1 (Hz)
                 f_cutoff2: Optional[float] = None, # Cutoff frequency 2 (Hz) for BPF/BSF
                 window_name: str = 'Hamming',# 'Hamming', 'Hann', 'Blackman', 'Kaiser', 'Rectangular', 'Bartlett'
                 kaiser_beta: float = 5.0):
        self.filter_family = 'FIR'
        self.filter_type = filter_type.upper()
        self.window_name = window_name
        self.response_type = f"FIR ({window_name})"
        n = int(num_taps)
        if n % 2 == 0:
            n += 1
        self.num_taps = max(3, min(511, n))
        self.order = self.num_taps - 1
        self.sample_rate = float(sample_rate)
        self.f_cutoff = float(f_cutoff)
        self.f_cutoff2 = float(f_cutoff2) if f_cutoff2 is not None else self.f_cutoff * 2.0
        self.kaiser_beta = float(kaiser_beta)
        self.ripple_db = 0.0
        self.stopband_db = 0.0

        self.taps: List[float] = []
        self.sections: List[BiquadSection] = []  # Empty for FIR
        self.buffer = [0.0] * (self.num_taps * 2)
        self.buf_idx = self.num_taps

        self.design()

    def clone(self) -> 'FIRFilter':
        """Clone FIR filter with independent state."""
        f = FIRFilter.__new__(FIRFilter)
        f.filter_family = 'FIR'
        f.filter_type = self.filter_type
        f.window_name = self.window_name
        f.response_type = self.response_type
        f.num_taps = self.num_taps
        f.order = self.order
        f.sample_rate = self.sample_rate
        f.f_cutoff = self.f_cutoff
        f.f_cutoff2 = self.f_cutoff2
        f.kaiser_beta = self.kaiser_beta
        f.ripple_db = self.ripple_db
        f.stopband_db = self.stopband_db
        f.taps = list(self.taps)
        f.sections = []
        f.buffer = list(self.buffer)
        f.buf_idx = self.buf_idx
        return f

    def reset(self):
        """Clear delay-line history."""
        self.buffer = [0.0] * (self.num_taps * 2)
        self.buf_idx = self.num_taps

    @staticmethod
    def _besseli0(x: float) -> float:
        """Modified Bessel function of the first kind, order 0 (for Kaiser window)."""
        ax = abs(x)
        if ax < 3.75:
            y = (x / 3.75) ** 2
            return 1.0 + y * (3.5156229 + y * (3.0899424 + y * (1.2067492 +
                   y * (0.2659732 + y * (0.360768e-1 + y * 0.45813e-2)))))
        else:
            y = 3.75 / ax
            return (math.exp(ax) / math.sqrt(ax)) * (0.39894228 + y * (0.1328592e-1 +
                   y * (0.225319e-2 + y * (-0.157565e-2 + y * (0.916281e-2 +
                   y * (-0.2057706e-1 + y * (0.2635537e-1 + y * (-0.1647633e-1 +
                   y * 0.392377e-2))))))))

    def design(self):
        """Compute windowed-sinc impulse response taps."""
        N = self.num_taps
        M = (N - 1) / 2.0
        fs = self.sample_rate
        nyquist = fs / 2.0

        f1 = max(10.0, min(self.f_cutoff, nyquist * 0.99))
        f2 = max(10.0, min(self.f_cutoff2, nyquist * 0.99))
        if f1 > f2:
            f1, f2 = f2, f1

        wc1 = 2.0 * math.pi * f1 / fs
        wc2 = 2.0 * math.pi * f2 / fs

        # 1. Compute Window function w[n]
        w = [0.0] * N
        n_minus_1 = max(1, N - 1)
        w_name = self.window_name.lower()
        if 'rect' in w_name:
            w = [1.0] * N
        elif 'hann' in w_name:
            for i in range(N):
                w[i] = 0.50 - 0.50 * math.cos(2.0 * math.pi * i / n_minus_1)
        elif 'blackman' in w_name:
            for i in range(N):
                w[i] = 0.42 - 0.50 * math.cos(2.0 * math.pi * i / n_minus_1) + 0.08 * math.cos(4.0 * math.pi * i / n_minus_1)
        elif 'kaiser' in w_name:
            i0_beta = self._besseli0(self.kaiser_beta)
            for i in range(N):
                q = (2.0 * i / n_minus_1) - 1.0
                val = self.kaiser_beta * math.sqrt(max(0.0, 1.0 - q * q))
                w[i] = self._besseli0(val) / i0_beta
        elif 'bartlett' in w_name or 'triang' in w_name:
            for i in range(N):
                w[i] = 1.0 - abs(2.0 * i / n_minus_1 - 1.0)
        else:  # Default: Hamming
            for i in range(N):
                w[i] = 0.54 - 0.46 * math.cos(2.0 * math.pi * i / n_minus_1)

        # 2. Compute Ideal Sinc Kernel
        h = [0.0] * N
        for n in range(N):
            k = n - M
            if abs(k) < 1e-9:
                sinc1 = wc1 / math.pi
                sinc2 = wc2 / math.pi
            else:
                sinc1 = math.sin(wc1 * k) / (math.pi * k)
                sinc2 = math.sin(wc2 * k) / (math.pi * k)

            if self.filter_type == 'LPF':
                ideal = sinc1
            elif self.filter_type == 'HPF':
                delta = 1.0 if abs(k) < 1e-9 else 0.0
                ideal = delta - sinc1
            elif self.filter_type == 'BPF':
                ideal = sinc2 - sinc1
            elif self.filter_type in ('BSF', 'NOTCH'):
                delta = 1.0 if abs(k) < 1e-9 else 0.0
                ideal = delta - (sinc2 - sinc1)
            else:
                ideal = sinc1

            h[n] = ideal * w[n]

        # 3. Normalize Gain at Center of Passband
        if self.filter_type in ('LPF', 'BSF'):
            eval_f = 0.0
        elif self.filter_type == 'HPF':
            eval_f = nyquist * 0.99
        else:  # BPF
            eval_f = math.sqrt(f1 * f2)

        w_eval = 2.0 * math.pi * eval_f / fs
        H_eval = sum(h[n] * cmath.exp(-1j * w_eval * n) for n in range(N))
        gain = abs(H_eval)
        if gain > 1e-9:
            h = [val / gain for val in h]

        self.taps = h

    def process_sample(self, x: float) -> float:
        """Process one sample through circular delay buffer."""
        N = self.num_taps
        buf = self.buffer
        idx = self.buf_idx
        buf[idx] = x
        buf[idx - N] = x
        taps = self.taps
        acc = sum(taps[i] * buf[idx - i] for i in range(N))
        idx += 1
        if idx >= 2 * N:
            idx = N
        self.buf_idx = idx
        return acc

    def process_block(self, samples: List[float]) -> List[float]:
        """Process an array of audio samples through FIR delay line."""
        N = self.num_taps
        buf = self.buffer
        idx = self.buf_idx
        taps = self.taps
        out = [0.0] * len(samples)
        for s_idx, x in enumerate(samples):
            buf[idx] = x
            buf[idx - N] = x
            acc = sum(taps[i] * buf[idx - i] for i in range(N))
            out[s_idx] = acc
            idx += 1
            if idx >= 2 * N:
                idx = N
        self.buf_idx = idx
        return out

    def frequency_response(self, freqs: List[float]) -> Tuple[List[float], List[float], List[float]]:
        """
        Evaluate magnitude (dB), phase (degrees), and group delay for freqs (Hz).
        For linear-phase FIR of length N, group delay is exactly constant: (N - 1) / 2 samples.
        """
        mag_db = []
        phase_deg = []
        group_delays = []
        N = self.num_taps
        M = (N - 1) / 2.0
        two_pi = 2.0 * math.pi
        sr = self.sample_rate

        for f in freqs:
            w = two_pi * f / sr
            H = sum(self.taps[n] * cmath.exp(-1j * w * n) for n in range(N))
            mag = max(abs(H), 1e-12)
            mag_db.append(20.0 * math.log10(mag))
            phase = math.atan2(H.imag, H.real)
            phase_deg.append(math.degrees(phase))
            group_delays.append(M)

        return mag_db, phase_deg, group_delays

    def get_direct_form_coefficients(self) -> Tuple[List[float], List[float]]:
        """Return feedforward coefficients [b0, b1, ..., b_{N-1}] and feedback [1.0]."""
        return list(self.taps), [1.0]

    def generate_c_code(self, filter_var_name: str = "fir_filter") -> str:
        """
        Generate standalone ANSI C99 code for this Linear-Phase FIR filter.
        Uses double-buffer unrolled circular delay line for maximum performance.
        Zero external libraries.
        """
        N = self.num_taps
        lines = []
        lines.append("/*")
        lines.append(f" * Generated C Audio Filter: {self.response_type} {self.filter_type}")
        lines.append(f" * Number of Taps: {N}, Order: {self.order}, Sample Rate: {self.sample_rate:.1f} Hz")
        if self.filter_type in ('LPF', 'HPF'):
            lines.append(f" * Cutoff: {self.f_cutoff:.2f} Hz")
        else:
            lines.append(f" * Cutoff 1: {self.f_cutoff:.2f} Hz, Cutoff 2: {self.f_cutoff2:.2f} Hz")
        lines.append(" * Type: Linear-Phase Finite Impulse Response (FIR)")
        lines.append(f" * Group Delay: {(N - 1) / 2.0:.1f} samples (constant)")
        lines.append(" * Standalone ANSI C99 - Zero external dependencies")
        lines.append(" */\n")
        lines.append("#include <stddef.h>\n")
        lines.append(f"#define {filter_var_name.upper()}_NUM_TAPS {N}\n")
        lines.append("typedef struct {")
        lines.append(f"    float buffer[{filter_var_name.upper()}_NUM_TAPS * 2];")
        lines.append("    size_t buf_idx;")
        lines.append("} FIRFilter;\n")
        lines.append(f"static const float {filter_var_name.upper()}_TAPS[{filter_var_name.upper()}_NUM_TAPS] = {{")
        for i, tap in enumerate(self.taps):
            comment = f" /* tap {i} */" if (i % 4 == 0 or i == N - 1) else ""
            lines.append(f"    {tap:.9e}f,{comment}")
        lines.append("};\n")
        lines.append(f"static void {filter_var_name}_init(FIRFilter *f) {{")
        lines.append(f"    for (size_t i = 0; i < {filter_var_name.upper()}_NUM_TAPS * 2; ++i) {{")
        lines.append("        f->buffer[i] = 0.0f;")
        lines.append("    }")
        lines.append(f"    f->buf_idx = {filter_var_name.upper()}_NUM_TAPS;")
        lines.append("}\n")
        lines.append(f"static void {filter_var_name}_reset(FIRFilter *f) {{")
        lines.append(f"    {filter_var_name}_init(f);")
        lines.append("}\n")
        lines.append(f"static float {filter_var_name}_process_sample(FIRFilter *f, float in_sample) {{")
        lines.append("    size_t idx = f->buf_idx;")
        lines.append("    f->buffer[idx] = in_sample;")
        lines.append(f"    f->buffer[idx - {filter_var_name.upper()}_NUM_TAPS] = in_sample;")
        lines.append("    float acc = 0.0f;")
        lines.append(f"    for (size_t i = 0; i < {filter_var_name.upper()}_NUM_TAPS; ++i) {{")
        lines.append(f"        acc += {filter_var_name.upper()}_TAPS[i] * f->buffer[idx - i];")
        lines.append("    }")
        lines.append("    idx++;")
        lines.append(f"    if (idx >= {filter_var_name.upper()}_NUM_TAPS * 2) idx = {filter_var_name.upper()}_NUM_TAPS;")
        lines.append("    f->buf_idx = idx;")
        lines.append("    return acc;")
        lines.append("}\n")
        lines.append(f"static void {filter_var_name}_process_block(FIRFilter *f, const float *in, float *out, size_t count) {{")
        lines.append("    for (size_t n = 0; n < count; ++n) {")
        lines.append(f"        out[n] = {filter_var_name}_process_sample(f, in[n]);")
        lines.append("    }")
        lines.append("}\n")
        lines.append("#ifdef TEST_FILTER_STANDALONE")
        lines.append("#include <stdio.h>")
        lines.append("int main(void) {")
        lines.append("    FIRFilter filter;")
        lines.append(f"    {filter_var_name}_init(&filter);")
        lines.append("    printf(\"FIR filter initialized! Running impulse test:\\n\");")
        lines.append("    for (int i = 0; i < 16; ++i) {")
        lines.append("        float x = (i == 0) ? 1.0f : 0.0f;")
        lines.append(f"        float y = {filter_var_name}_process_sample(&filter, x);")
        lines.append("        printf(\"y[%2d] = %+.6f\\n\", i, y);")
        lines.append("    }")
        lines.append("    return 0;")
        lines.append("}")
        lines.append("#endif\n")
        return "\n".join(lines)
