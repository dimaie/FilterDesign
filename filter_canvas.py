"""
Custom Interactive Interactive Tkinter Canvas for Frequency Response Plotting
with draggable cutoffs, passband/stopband indicators, and dual-mode rendering.
Supports logarithmic frequency scale (10 Hz - 24 kHz) and dB magnitude / phase plots.
"""

import math
import tkinter as tk
from typing import Callable, Optional, List, Tuple


class InteractiveResponseCanvas(tk.Canvas):
    """
    High-performance custom canvas that renders:
    - Logarithmic grid (10 Hz to Nyquist / 24 kHz)
    - Magnitude Response curve (dB) with gridlines (-80 dB to +12 dB)
    - Phase Response curve (optional overlay or toggle)
    - Draggable vertical cutoff cursors (fc1, fc2) with real-time feedback
    - Shaded passband / stopband / transition band zones
    - Hover coordinate tooltip
    """
    def __init__(self, master, on_cutoff_dragged: Optional[Callable[[float, Optional[float]], None]] = None, **kwargs):
        super().__init__(master, bg='#ffffff', highlightthickness=0, **kwargs)
        self.on_cutoff_dragged = on_cutoff_dragged

        # Frequency and Plot ranges
        self.f_min = 10.0
        self.f_max = 24000.0
        self.db_min = -80.0
        self.db_max = 15.0

        # Current cutoffs
        self.fc1 = 1000.0
        self.fc2 = 4000.0
        self.filter_type = 'LPF'
        self.has_two_cutoffs = False

        # Data series to plot
        self.freq_points: List[float] = []
        self.mag_points: List[float] = []
        self.phase_points: List[float] = []
        self.show_phase = False

        # Dragging state
        self.drag_target: Optional[str] = None  # 'fc1', 'fc2', or None
        self.hover_info: Optional[Tuple[float, float, Optional[float]]] = None

        # Design criteria validation state
        self.criteria_met = True
        self.mismatch_reason = ""

        # Margins for axes
        self.margin_left = 55
        self.margin_right = 50
        self.margin_top = 25
        self.margin_bottom = 42

        # Bindings
        self.bind("<Configure>", self._on_resize)
        self.bind("<ButtonPress-1>", self._on_mouse_down)
        self.bind("<B1-Motion>", self._on_mouse_drag)
        self.bind("<ButtonRelease-1>", self._on_mouse_up)
        self.bind("<Motion>", self._on_mouse_move)
        self.bind("<Leave>", self._on_mouse_leave)

    def set_filter_state(self, filter_type: str, fc1: float, fc2: Optional[float],
                         sample_rate: float, freqs: List[float], mags: List[float],
                         phases: List[float],
                         criteria_met: bool = True,
                         mismatch_reason: str = ""):
        """Update filter state and trigger re-draw."""
        self.criteria_met = criteria_met
        self.mismatch_reason = mismatch_reason
        self.filter_type = filter_type.upper()
        self.has_two_cutoffs = (self.filter_type in ('BPF', 'BSF'))
        self.f_max = sample_rate / 2.0
        self.fc1 = max(self.f_min, min(self.f_max * 0.98, fc1))
        if fc2 is not None:
            self.fc2 = max(self.f_min, min(self.f_max * 0.98, fc2))
            if self.fc1 > self.fc2:
                self.fc1, self.fc2 = self.fc2, self.fc1

        self.freq_points = freqs
        self.mag_points = mags
        self.phase_points = phases
        self.redraw()

    def toggle_phase(self, show: bool):
        self.show_phase = show
        self.redraw()

    def _on_resize(self, event):
        self.redraw()

    # --- Coordinate conversions ---
    def f_to_x(self, f: float) -> float:
        w = max(10, self.winfo_width() - self.margin_left - self.margin_right)
        f_clamped = max(self.f_min, min(self.f_max, f))
        ratio = (math.log10(f_clamped) - math.log10(self.f_min)) / (math.log10(self.f_max) - math.log10(self.f_min))
        return self.margin_left + ratio * w

    def x_to_f(self, x: float) -> float:
        w = max(10, self.winfo_width() - self.margin_left - self.margin_right)
        ratio = (x - self.margin_left) / w
        ratio = max(0.0, min(1.0, ratio))
        log_f = math.log10(self.f_min) + ratio * (math.log10(self.f_max) - math.log10(self.f_min))
        return math.pow(10.0, log_f)

    def db_to_y(self, db: float) -> float:
        h = max(10, self.winfo_height() - self.margin_top - self.margin_bottom)
        db_clamped = max(self.db_min, min(self.db_max, db))
        ratio = (db_clamped - self.db_min) / (self.db_max - self.db_min)
        # y is inverted in canvas (0 at top)
        return self.margin_top + (1.0 - ratio) * h

    def y_to_db(self, y: float) -> float:
        h = max(10, self.winfo_height() - self.margin_top - self.margin_bottom)
        ratio = 1.0 - ((y - self.margin_top) / h)
        return self.db_min + ratio * (self.db_max - self.db_min)

    def phase_to_y(self, deg: float) -> float:
        h = max(10, self.winfo_height() - self.margin_top - self.margin_bottom)
        # Phase from -180 to +180 deg
        ratio = (deg - (-180.0)) / (180.0 - (-180.0))
        ratio = max(0.0, min(1.0, ratio))
        return self.margin_top + (1.0 - ratio) * h

    # --- Mouse interactions for dragging cutoffs ---
    def _on_mouse_down(self, event):
        x = event.x
        x_fc1 = self.f_to_x(self.fc1)
        tol = 12  # pixel tolerance for grab

        if self.has_two_cutoffs:
            x_fc2 = self.f_to_x(self.fc2)
            d1 = abs(x - x_fc1)
            d2 = abs(x - x_fc2)
            if d1 < tol and d1 <= d2:
                self.drag_target = 'fc1'
            elif d2 < tol:
                self.drag_target = 'fc2'
            else:
                self.drag_target = None
        else:
            if abs(x - x_fc1) < tol:
                self.drag_target = 'fc1'
            else:
                self.drag_target = None

    def _on_mouse_drag(self, event):
        if not self.drag_target:
            return
        new_f = self.x_to_f(event.x)
        # Snap/clamp
        new_f = max(self.f_min, min(self.f_max * 0.98, new_f))

        if self.drag_target == 'fc1':
            if self.has_two_cutoffs and new_f >= self.fc2 - 20:
                new_f = max(self.f_min, self.fc2 - 20)
            self.fc1 = new_f
        elif self.drag_target == 'fc2':
            if new_f <= self.fc1 + 20:
                new_f = min(self.f_max * 0.98, self.fc1 + 20)
            self.fc2 = new_f

        if self.on_cutoff_dragged:
            if self.has_two_cutoffs:
                self.on_cutoff_dragged(self.fc1, self.fc2)
            else:
                self.on_cutoff_dragged(self.fc1, None)

        self.redraw()

    def _on_mouse_up(self, event):
        self.drag_target = None

    def _on_mouse_move(self, event):
        # Update cursor when hovering over handles
        x = event.x
        x_fc1 = self.f_to_x(self.fc1)
        tol = 10
        on_handle = abs(x - x_fc1) < tol
        if self.has_two_cutoffs:
            x_fc2 = self.f_to_x(self.fc2)
            on_handle = on_handle or (abs(x - x_fc2) < tol)

        if on_handle:
            self.config(cursor="sb_h_double_arrow")
        else:
            self.config(cursor="crosshair")

        # Track tooltip hover
        if self.margin_left <= event.x <= (self.winfo_width() - self.margin_right) and \
           self.margin_top <= event.y <= (self.winfo_height() - self.margin_bottom):
            cur_f = self.x_to_f(event.x)
            cur_db = None
            if self.freq_points and self.mag_points:
                # Interpolate filter curve magnitude at cur_f
                for i in range(len(self.freq_points) - 1):
                    f0, f1 = self.freq_points[i], self.freq_points[i + 1]
                    if f0 <= cur_f <= f1:
                        t = (cur_f - f0) / max(1e-9, f1 - f0)
                        cur_db = self.mag_points[i] + t * (self.mag_points[i + 1] - self.mag_points[i])
                        break
                if cur_db is None and self.freq_points:
                    cur_db = self.mag_points[-1] if cur_f >= self.freq_points[-1] else self.mag_points[0]
            if cur_db is None:
                cur_db = self.y_to_db(event.y)

            cur_phase = None
            if self.show_phase and self.freq_points and self.phase_points:
                # Interpolate phase at cur_f
                for i in range(len(self.freq_points) - 1):
                    f0, f1 = self.freq_points[i], self.freq_points[i + 1]
                    if f0 <= cur_f <= f1:
                        t = (cur_f - f0) / max(1e-9, f1 - f0)
                        cur_phase = self.phase_points[i] + t * (self.phase_points[i + 1] - self.phase_points[i])
                        break
                if cur_phase is None and self.freq_points:
                    cur_phase = self.phase_points[-1] if cur_f >= self.freq_points[-1] else self.phase_points[0]

            self.hover_info = (cur_f, cur_db, cur_phase)
        else:
            self.hover_info = None
        self.redraw()

    def _on_mouse_leave(self, event):
        self.hover_info = None
        self.redraw()

    # --- Drawing Routines ---
    def redraw(self):
        self.delete("all")
        w = self.winfo_width()
        h = self.winfo_height()
        if w < 100 or h < 100:
            return

        plot_w = w - self.margin_left - self.margin_right
        plot_h = h - self.margin_top - self.margin_bottom

        # Plot background: clean white when satisfied, soft warm amber tint when criteria mismatched
        bg_fill = '#ffffff' if self.criteria_met else '#fff6f0'
        border_outline = '#d0d7de' if self.criteria_met else '#cf222e'
        border_width = 1 if self.criteria_met else 2

        self.create_rectangle(self.margin_left, self.margin_top,
                              self.margin_left + plot_w, self.margin_top + plot_h,
                              fill=bg_fill, outline=border_outline, width=border_width)

        # Draw Passband/Stopband Shading
        self._draw_zones(plot_h)

        # Grid lines (Log frequency & Linear dB)
        self._draw_grid(plot_w, plot_h)

        # Draw Phase Response if enabled
        if self.show_phase and self.freq_points and self.phase_points:
            self._draw_phase_curve()

        # Draw Magnitude Response Curve
        if self.freq_points and self.mag_points:
            self._draw_magnitude_curve()

        # Draw Cutoff Draggable Cursors
        self._draw_cutoffs(plot_h)

        # Draw Criteria Mismatch Warning Banner inside plot
        if not self.criteria_met and self.mismatch_reason:
            banner_y = self.margin_top + 8
            banner_h = 22
            banner_w = min(plot_w - 16, max(380, len(self.mismatch_reason) * 6 + 40))
            self.create_rectangle(self.margin_left + 8, banner_y,
                                  self.margin_left + 8 + banner_w, banner_y + banner_h,
                                  fill='#fff0e6', outline='#cf222e', width=1)
            self.create_text(self.margin_left + 16, banner_y + banner_h / 2,
                             text=f"⚠️ {self.mismatch_reason}",
                             fill='#a40e26', font=('Segoe UI', 8, 'bold'), anchor='w')

        # Draw Tooltip (offset below criteria mismatch banner if banner is active)
        if self.hover_info:
            f_val, db_val, phase_val = self.hover_info
            if phase_val is not None:
                text = f"{f_val:,.1f} Hz | {db_val:+.1f} dB | {phase_val:+.1f}°"
                box_w = 230
            else:
                text = f"{f_val:,.1f} Hz | {db_val:+.1f} dB"
                box_w = 160
            tooltip_y = (self.margin_top + 36) if (not self.criteria_met and self.mismatch_reason) else (self.margin_top + 10)
            self.create_rectangle(self.margin_left + 10, tooltip_y,
                                  self.margin_left + 10 + box_w, tooltip_y + 22,
                                  fill='#ffffff', outline='#0969da', width=1)
            self.create_text(self.margin_left + 10 + box_w / 2, tooltip_y + 11,
                             text=text, fill='#24292f', font=('Segoe UI', 9, 'bold'))

    def _draw_zones(self, plot_h):
        """Draw subtle translucent background shading for passbands."""
        top = self.margin_top
        bot = self.margin_top + plot_h
        passband_fill = '#e8f5e9' if self.criteria_met else '#ffe8dc'
        stopband_fill = '#ffebee' if self.criteria_met else '#ffdcd6'

        if self.filter_type == 'LPF':
            x1 = self.margin_left
            x2 = self.f_to_x(self.fc1)
            self.create_rectangle(x1, top, x2, bot, fill=passband_fill, outline='', width=0)
        elif self.filter_type == 'HPF':
            x1 = self.f_to_x(self.fc1)
            x2 = self.winfo_width() - self.margin_right
            self.create_rectangle(x1, top, x2, bot, fill=passband_fill, outline='', width=0)
        elif self.filter_type == 'BPF':
            x1 = self.f_to_x(self.fc1)
            x2 = self.f_to_x(self.fc2)
            self.create_rectangle(x1, top, x2, bot, fill=passband_fill, outline='', width=0)
        elif self.filter_type == 'BSF':
            x1 = self.f_to_x(self.fc1)
            x2 = self.f_to_x(self.fc2)
            self.create_rectangle(x1, top, x2, bot, fill=stopband_fill, outline='', width=0)

    def _draw_grid(self, plot_w, plot_h):
        # dB Grid (Horizontal)
        for db in [-80, -60, -40, -20, -10, -6, -3, 0, 10]:
            if self.db_min <= db <= self.db_max:
                y = self.db_to_y(db)
                color = '#57606a' if db == 0 else '#e1e4e8'
                dash = () if db == 0 else (2, 4)
                self.create_line(self.margin_left, y, self.margin_left + plot_w, y,
                                 fill=color, dash=dash, width=1)
                # Label
                self.create_text(self.margin_left - 8, y, text=f"{db:d}",
                                 fill='#57606a' if db != 0 else '#1a7f37',
                                 font=('Segoe UI', 8), anchor='e')

        # Frequency Grid (Vertical Decades & Sub-decades)
        decades = [10, 100, 1000, 10000]
        multipliers = [1, 2, 5]
        for dec in decades:
            for m in multipliers:
                f = dec * m
                if self.f_min <= f <= self.f_max:
                    x = self.f_to_x(f)
                    is_major = (m == 1)
                    color = '#d0d7de' if is_major else '#eaeef2'
                    dash = () if is_major else (2, 4)
                    self.create_line(x, self.margin_top, x, self.margin_top + plot_h,
                                     fill=color, dash=dash, width=1)

                    # Labels for 10Hz, 100Hz, 1kHz, 10kHz, 20kHz
                    if is_major or f in (20, 50, 200, 500, 2000, 5000, 20000):
                        lbl = f"{f/1000:.0f}k" if f >= 1000 else f"{f:.0f}"
                        self.create_text(x, self.margin_top + plot_h + 8, text=lbl,
                                         fill='#57606a', font=('Segoe UI', 8), anchor='n')

        # Axis unit labels
        self.create_text(self.margin_left - 10, self.margin_top - 12, text="Gain [dB]",
                         fill='#24292f', font=('Segoe UI', 8, 'bold'), anchor='w')
        if self.show_phase:
            self.create_text(self.margin_left + plot_w + 10, self.margin_top - 12, text="Phase [°]",
                             fill='#b08800', font=('Segoe UI', 8, 'bold'), anchor='e')
        # Center frequency axis label to avoid overlapping with right-most tick labels (e.g. 20k)
        self.create_text(self.margin_left + plot_w / 2, self.margin_top + plot_h + 24, text="Frequency [Hz]",
                         fill='#24292f', font=('Segoe UI', 8, 'bold'), anchor='n')

        # Phase Grid & Axis (Right Side) when enabled
        if self.show_phase:
            for deg in [-180, -135, -90, -45, 0, 45, 90, 135, 180]:
                y = self.phase_to_y(deg)
                self.create_line(self.margin_left + plot_w, y, self.margin_left + plot_w + 5, y,
                                 fill='#b08800', width=1)
                self.create_text(self.margin_left + plot_w + 8, y, text=f"{deg:+d}°" if deg != 0 else "0°",
                                 fill='#b08800', font=('Segoe UI', 8), anchor='w')

    def _draw_magnitude_curve(self):
        coords = []
        plot_top = self.margin_top
        plot_bot = self.winfo_height() - self.margin_bottom

        for f, db in zip(self.freq_points, self.mag_points):
            if f < self.f_min or f > self.f_max:
                continue
            x = self.f_to_x(f)
            y = self.db_to_y(db)
            # Clamp y inside canvas area
            y_clamped = max(plot_top, min(plot_bot, y))
            coords.extend([x, y_clamped])

        if len(coords) >= 4:
            # Subtle glow and clear primary green curve
            self.create_line(coords, fill='#81c784', width=4, smooth=True)
            self.create_line(coords, fill='#1b5e20', width=2, smooth=True)

    def _draw_phase_curve(self):
        coords = []
        for f, deg in zip(self.freq_points, self.phase_points):
            if f < self.f_min or f > self.f_max:
                continue
            x = self.f_to_x(f)
            y = self.phase_to_y(deg)
            coords.extend([x, y])

        if len(coords) >= 4:
            self.create_line(coords, fill='#b08800', width=1.5, dash=(4, 2), smooth=False)

    def _draw_cutoffs(self, plot_h):
        top = self.margin_top
        bot = self.margin_top + plot_h

        # Helper to draw a single cursor
        def draw_cursor(f: float, label: str, color: str):
            x = self.f_to_x(f)
            # Vertical dashed line
            self.create_line(x, top, x, bot, fill=color, dash=(4, 3), width=2)

            # Triangular grab handle at top
            poly_top = [x - 8, top - 14, x + 8, top - 14, x + 8, top - 5, x, top, x - 8, top - 5]
            self.create_polygon(poly_top, fill=color, outline='#24292f', width=1)
            self.create_text(x, top - 18, text=f"{label}: {f:,.0f}Hz",
                             fill=color, font=('Segoe UI', 8, 'bold'), anchor='s')

            # Bottom arrow handle
            poly_bot = [x - 7, bot + 6, x + 7, bot + 6, x, bot]
            self.create_polygon(poly_bot, fill=color, outline='#24292f', width=1)

        draw_cursor(self.fc1, "Fc1" if self.has_two_cutoffs else "Cutoff", "#0969da")
        if self.has_two_cutoffs:
            draw_cursor(self.fc2, "Fc2", "#cf222e")
