"""
Audio Filter Design Studio - Main Application GUI
A modern, dark-themed, desktop application for interactive audio filter design.
Features:
1. Filter types: LPF, HPF, BPF, BSF (Notch)
2. Responses: Butterworth, Chebyshev I, Chebyshev II, Bessel
3. Real-time draggable cutoffs on frequency response canvas + precision sliders & entry
4. Full interactive Test Mode:
   - Input signals: Sine, Multi-tone, Log/Linear Chirp, White/Pink noise, Square, Triangle, Sawtooth, Impulse, Step
   - Waveform viewer (Input vs Output overlay with zoom)
   - FFT Spectrum viewer (Input vs Output spectrum)
5. Export capabilities:
   - Standalone ANSI C99 source code (no extra libraries required)
   - Test data export to CSV
   - Test data export to JSON
   - Audio export to 16-bit PCM WAV
   - Direct Form & Biquad coefficients table export
"""

import os
import math
from typing import Optional, List
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from filter_engine import DigitalFilter
from filter_canvas import InteractiveResponseCanvas
from signal_testbench import (generate_signal, compute_fft,
                              export_data_to_csv, export_data_to_json, export_audio_to_wav)


class FilterDesignApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Filter Design Studio - LPF / HPF / BPF / BSF")
        self.geometry("1240x860")
        self.minsize(980, 680)

        # Apply dark theme styling
        self.configure(bg='#1e2227')
        self._setup_styles()

        # Application State
        self.sample_rate = 48000.0
        self.filter_type = 'LPF'
        self.response_type = 'Butterworth'
        self.order = 4
        self.f_cutoff = 1000.0
        self.f_cutoff2 = 4000.0
        self.ripple_db = 0.5
        self.stopband_db = 40.0

        self.current_filter: Optional[DigitalFilter] = None

        # Test mode data cache
        self.test_time = []
        self.test_in = []
        self.test_out = []

        self._initialized = False

        # Build UI layout
        self._build_layout()

        self._initialized = True

        # Initial filter calculation
        self.recalculate_filter()
        self.run_test_signal()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        # Color palette (One Dark inspired)
        bg = '#1e2227'
        card_bg = '#21252b'
        fg = '#abb2bf'
        accent = '#61afef'
        active_tab = '#2c313a'

        style.configure('.', background=bg, foreground=fg, font=('Segoe UI', 9))
        style.configure('TNotebook', background=bg, borderwidth=0)
        style.configure('TNotebook.Tab', background=card_bg, foreground=fg,
                        padding=[14, 6], font=('Segoe UI', 9, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', active_tab), ('active', '#282c34')],
                  foreground=[('selected', accent), ('active', '#ffffff')])

        style.configure('TFrame', background=bg)
        style.configure('Card.TFrame', background=card_bg, relief='flat')
        style.configure('TLabel', background=bg, foreground=fg)
        style.configure('Card.TLabel', background=card_bg, foreground=fg)
        style.configure('Header.TLabel', background=card_bg, foreground='#e5c07b', font=('Segoe UI', 10, 'bold'))
        style.configure('Title.TLabel', background=bg, foreground='#98c379', font=('Segoe UI', 12, 'bold'))

        style.configure('TCombobox', fieldbackground='#282c34', background='#282c34',
                        foreground='#ffffff', arrowcolor=accent, padding=4)
        style.map('TCombobox', fieldbackground=[('readonly', '#282c34')],
                  foreground=[('readonly', '#ffffff')])

        style.configure('TEntry', fieldbackground='#282c34', foreground='#ffffff', insertcolor='#ffffff')
        style.configure('TScale', background=card_bg, troughcolor='#181a1f')

        style.configure('Accent.TButton', background='#3e4451', foreground='#ffffff',
                        font=('Segoe UI', 9, 'bold'), borderwidth=1, padding=5)
        style.map('Accent.TButton',
                  background=[('pressed', '#61afef'), ('active', '#4b5263')],
                  foreground=[('pressed', '#000000'), ('active', '#61afef')])

        style.configure('Action.TButton', background='#98c379', foreground='#1e2227',
                        font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=5)
        style.map('Action.TButton',
                  background=[('pressed', '#7cb342'), ('active', '#b5e890')])

    def _build_layout(self):
        # Top Header Bar
        top_bar = ttk.Frame(self, padding=(16, 10))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = ttk.Label(top_bar, text="🎵 Audio Filter Design Studio", style='Title.TLabel')
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = ttk.Label(top_bar, text="— Interactive DSP Biquad Synthesizer & C-Code Generator",
                                 foreground='#5c6370', font=('Segoe UI', 9, 'italic'))
        subtitle_lbl.pack(side=tk.LEFT, padx=10)

        # Main Split: Left Sidebar (Controls) + Right Content (Plots & Tabs)
        main_pane = ttk.Frame(self, padding=(10, 0, 10, 10))
        main_pane.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # --- LEFT SIDEBAR CONTROLS ---
        sidebar = ttk.Frame(main_pane, style='Card.TFrame', padding=12, width=310)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)

        # Filter Parameters Section
        lbl_sec1 = ttk.Label(sidebar, text="FILTER SPECIFICATIONS", style='Header.TLabel')
        lbl_sec1.pack(anchor='w', pady=(0, 8))

        # Filter Type (LPF, HPF, BPF, BSF)
        ttk.Label(sidebar, text="Filter Type:", style='Card.TLabel').pack(anchor='w')
        self.var_filter_type = tk.StringVar(value="LPF")
        cb_ftype = ttk.Combobox(sidebar, textvariable=self.var_filter_type,
                                values=["LPF", "HPF", "BPF", "BSF (Notch)"], state="readonly")
        cb_ftype.pack(fill=tk.X, pady=(2, 8))
        cb_ftype.bind("<<ComboboxSelected>>", self._on_filter_type_changed)

        # Response Type (Butterworth, Chebyshev I, Chebyshev II, Bessel)
        ttk.Label(sidebar, text="Response Approximation:", style='Card.TLabel').pack(anchor='w')
        self.var_response_type = tk.StringVar(value="Butterworth")
        cb_rtype = ttk.Combobox(sidebar, textvariable=self.var_response_type,
                                values=["Butterworth", "Chebyshev I", "Chebyshev II", "Bessel"],
                                state="readonly")
        cb_rtype.pack(fill=tk.X, pady=(2, 8))
        cb_rtype.bind("<<ComboboxSelected>>", self._on_param_changed)

        # Filter Order
        f_order_row = ttk.Frame(sidebar, style='Card.TFrame')
        f_order_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(f_order_row, text="Order:", style='Card.TLabel').pack(side=tk.LEFT)
        self.var_order_val = tk.StringVar(value="4")
        self.lbl_order_num = ttk.Label(f_order_row, textvariable=self.var_order_val,
                                       foreground='#61afef', font=('Segoe UI', 9, 'bold'))
        self.lbl_order_num.pack(side=tk.RIGHT)

        self.scale_order = ttk.Scale(sidebar, from_=1, to=10, style='TScale',
                                     command=self._on_order_scale)
        self.scale_order.set(4)
        self.scale_order.pack(fill=tk.X, pady=(0, 10))

        # Sample Rate
        ttk.Label(sidebar, text="Sample Rate (Hz):", style='Card.TLabel').pack(anchor='w')
        self.var_sr = tk.StringVar(value="48000")
        cb_sr = ttk.Combobox(sidebar, textvariable=self.var_sr,
                             values=["44100", "48000", "88200", "96000", "192000"], state="readonly")
        cb_sr.pack(fill=tk.X, pady=(2, 10))
        cb_sr.bind("<<ComboboxSelected>>", self._on_sr_changed)

        # Cutoff Frequency 1 (with slider and entry)
        ttk.Label(sidebar, text="Cutoff Frequency 1 (Hz):", style='Card.TLabel').pack(anchor='w')
        f1_row = ttk.Frame(sidebar, style='Card.TFrame')
        f1_row.pack(fill=tk.X, pady=(2, 2))
        self.var_fc1 = tk.DoubleVar(value=1000.0)
        self.entry_fc1 = ttk.Entry(f1_row, width=10)
        self.entry_fc1.insert(0, "1000.0")
        self.entry_fc1.pack(side=tk.LEFT)
        self.entry_fc1.bind("<Return>", self._on_fc1_entry)
        self.entry_fc1.bind("<FocusOut>", self._on_fc1_entry)
        ttk.Label(f1_row, text=" Hz (drag graph / slide)", style='Card.TLabel', font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=4)

        # Slider on log-scale mapping (0 to 1)
        self.scale_fc1 = ttk.Scale(sidebar, from_=0.0, to=1.0, style='TScale',
                                   command=self._on_scale_fc1)
        self.scale_fc1.pack(fill=tk.X, pady=(0, 8))
        self._update_slider_pos_from_hz(self.scale_fc1, 1000.0)

        # Cutoff Frequency 2 (for BPF / BSF)
        self.lbl_fc2 = ttk.Label(sidebar, text="Cutoff Frequency 2 (Hz):", style='Card.TLabel')
        self.lbl_fc2.pack(anchor='w')
        self.f2_row = ttk.Frame(sidebar, style='Card.TFrame')
        self.f2_row.pack(fill=tk.X, pady=(2, 2))
        self.var_fc2 = tk.DoubleVar(value=4000.0)
        self.entry_fc2 = ttk.Entry(self.f2_row, width=10)
        self.entry_fc2.insert(0, "4000.0")
        self.entry_fc2.pack(side=tk.LEFT)
        self.entry_fc2.bind("<Return>", self._on_fc2_entry)
        self.entry_fc2.bind("<FocusOut>", self._on_fc2_entry)
        ttk.Label(self.f2_row, text=" Hz", style='Card.TLabel', font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=4)

        self.scale_fc2 = ttk.Scale(sidebar, from_=0.0, to=1.0, style='TScale',
                                   command=self._on_scale_fc2)
        self.scale_fc2.pack(fill=tk.X, pady=(0, 8))
        self._update_slider_pos_from_hz(self.scale_fc2, 4000.0)

        # Dynamic Ripple / Stopband options
        self.frame_dynamic_params = ttk.Frame(sidebar, style='Card.TFrame')
        self.frame_dynamic_params.pack(fill=tk.X, pady=(0, 10))

        self.lbl_ripple = ttk.Label(self.frame_dynamic_params, text="Passband Ripple (dB):", style='Card.TLabel')
        self.entry_ripple = ttk.Entry(self.frame_dynamic_params, width=8)
        self.entry_ripple.insert(0, "0.5")
        self.entry_ripple.bind("<Return>", self._on_param_changed)

        self.lbl_stopband = ttk.Label(self.frame_dynamic_params, text="Stopband Atten (dB):", style='Card.TLabel')
        self.entry_stopband = ttk.Entry(self.frame_dynamic_params, width=8)
        self.entry_stopband.insert(0, "40.0")
        self.entry_stopband.bind("<Return>", self._on_param_changed)

        # Checkbox for Phase curve
        self.var_show_phase = tk.BooleanVar(value=False)
        chk_phase = tk.Checkbutton(sidebar, text="Show Phase Response", variable=self.var_show_phase,
                                   bg='#21252b', fg='#abb2bf', selectcolor='#181a1f',
                                   activebackground='#21252b', activeforeground='#61afef',
                                   command=self._on_toggle_phase)
        chk_phase.pack(anchor='w', pady=(4, 12))

        # Bottom Quick Action Buttons in sidebar
        btn_gen_c = ttk.Button(sidebar, text="⚡ View / Export C Code", style='Accent.TButton',
                               command=self._show_c_code_tab)
        btn_gen_c.pack(fill=tk.X, pady=(4, 4))

        btn_export_data = ttk.Button(sidebar, text="💾 Export Filter / Test Data", style='Accent.TButton',
                                     command=self._export_menu_action)
        btn_export_data.pack(fill=tk.X, pady=(4, 4))

        # --- RIGHT NOTEBOOK WORKSPACE ---
        right_panel = ttk.Frame(main_pane)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(right_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # Tab 1: Interactive Frequency Response & Design
        self.tab_response = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_response, text=" 📊 Frequency Response (Interactive Design) ")
        self._build_tab_response(self.tab_response)

        # Tab 2: Test Mode & Signal Verification
        self.tab_test = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_test, text=" 🧪 Test Mode & Signal Verification ")
        self._build_tab_test(self.tab_test)

        # Tab 3: C Code Generation
        self.tab_code = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_code, text=" 💻 ANSI C Code Generator ")
        self._build_tab_code(self.tab_code)

        # Tab 4: Coefficients & Filter Inspection
        self.tab_coeffs = ttk.Frame(self.notebook)
        self.notebook.add(self.tab_coeffs, text=" 📐 Coefficients & Biquads ")
        self._build_tab_coeffs(self.tab_coeffs)

    # --- TAB 1: INTERACTIVE DESIGN ---
    def _build_tab_response(self, parent):
        # Instruction Banner
        banner = ttk.Frame(parent, style='Card.TFrame', padding=(10, 6))
        banner.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        info_lbl = ttk.Label(banner,
                             text="💡 Hint: Drag the colored vertical cursor handles directly on the graph to set filter cutoffs in real time!",
                             style='Card.TLabel', foreground='#98c379', font=('Segoe UI', 9, 'bold'))
        info_lbl.pack(side=tk.LEFT)

        self.lbl_stats = ttk.Label(banner, text="", style='Card.TLabel', foreground='#61afef')
        self.lbl_stats.pack(side=tk.RIGHT)

        # Interactive Response Canvas
        self.canvas_response = InteractiveResponseCanvas(parent, on_cutoff_dragged=self._on_canvas_cutoff_drag)
        self.canvas_response.pack(fill=tk.BOTH, expand=True)

    # --- TAB 2: TEST MODE & VERIFICATION ---
    def _build_tab_test(self, parent):
        # Top Control Bar for Signal Generation
        test_ctrls = ttk.Frame(parent, style='Card.TFrame', padding=10)
        test_ctrls.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        # Row 1: Signal selection
        r1 = ttk.Frame(test_ctrls, style='Card.TFrame')
        r1.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(r1, text="Test Signal Type:", style='Card.TLabel', font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 6))
        self.var_test_sig = tk.StringVar(value="Sweep (Log)")
        cb_test_sig = ttk.Combobox(r1, textvariable=self.var_test_sig, width=15, state="readonly",
                                   values=["Sine", "Multi-tone", "Sweep (Log)", "Sweep (Linear)",
                                           "White Noise", "Pink Noise", "Square", "Triangle", "Sawtooth",
                                           "Impulse", "Step"])
        cb_test_sig.pack(side=tk.LEFT, padx=(0, 15))
        cb_test_sig.bind("<<ComboboxSelected>>", lambda e: self.run_test_signal())

        ttk.Label(r1, text="Freq 1 (Hz):", style='Card.TLabel').pack(side=tk.LEFT, padx=(0, 4))
        self.var_test_f1 = tk.StringVar(value="500")
        ent_f1 = ttk.Entry(r1, textvariable=self.var_test_f1, width=7)
        ent_f1.pack(side=tk.LEFT, padx=(0, 15))
        ent_f1.bind("<Return>", lambda e: self.run_test_signal())

        ttk.Label(r1, text="Freq 2 / End (Hz):", style='Card.TLabel').pack(side=tk.LEFT, padx=(0, 4))
        self.var_test_f2 = tk.StringVar(value="10000")
        ent_f2 = ttk.Entry(r1, textvariable=self.var_test_f2, width=7)
        ent_f2.pack(side=tk.LEFT, padx=(0, 15))
        ent_f2.bind("<Return>", lambda e: self.run_test_signal())

        ttk.Label(r1, text="Duration (s):", style='Card.TLabel').pack(side=tk.LEFT, padx=(0, 4))
        self.var_test_dur = tk.StringVar(value="0.05")
        ent_dur = ttk.Entry(r1, textvariable=self.var_test_dur, width=5)
        ent_dur.pack(side=tk.LEFT, padx=(0, 15))
        ent_dur.bind("<Return>", lambda e: self.run_test_signal())

        btn_run = ttk.Button(r1, text="▶ Run Verification", style='Action.TButton',
                             command=self.run_test_signal)
        btn_run.pack(side=tk.RIGHT)

        # Row 2: Signal Metrics & Actions
        r2 = ttk.Frame(test_ctrls, style='Card.TFrame')
        r2.pack(fill=tk.X)

        self.lbl_test_metrics = ttk.Label(r2, text="Metrics: Calculating...",
                                          style='Card.TLabel', foreground='#61afef', font=('Segoe UI', 9))
        self.lbl_test_metrics.pack(side=tk.LEFT)

        btn_save_wav = ttk.Button(r2, text="Export Output to WAV", style='Accent.TButton',
                                  command=self._export_test_wav)
        btn_save_wav.pack(side=tk.RIGHT, padx=4)

        btn_save_csv = ttk.Button(r2, text="Export Data to CSV", style='Accent.TButton',
                                  command=self._export_test_csv)
        btn_save_csv.pack(side=tk.RIGHT, padx=4)

        # Split Waveform and FFT Views
        vis_split = ttk.PanedWindow(parent, orient=tk.VERTICAL)
        vis_split.pack(fill=tk.BOTH, expand=True)

        # Upper plot: Time Domain Waveform (Input vs Output)
        frame_wave = ttk.Frame(vis_split, style='Card.TFrame')
        vis_split.add(frame_wave, weight=1)

        header_wave = ttk.Frame(frame_wave, style='Card.TFrame', padding=(8, 4))
        header_wave.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header_wave, text="Time Domain Oscilloscope (Input vs Filtered Output)",
                  style='Header.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_wave, text="— Cyan: Input Signal  |  Green: Filtered Output",
                  style='Card.TLabel', foreground='#abb2bf', font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=10)

        self.canvas_wave = tk.Canvas(frame_wave, bg='#181a1f', highlightthickness=0)
        self.canvas_wave.pack(fill=tk.BOTH, expand=True)

        # Lower plot: FFT Frequency Spectrum (Input vs Output)
        frame_fft = ttk.Frame(vis_split, style='Card.TFrame')
        vis_split.add(frame_fft, weight=1)

        header_fft = ttk.Frame(frame_fft, style='Card.TFrame', padding=(8, 4))
        header_fft.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header_fft, text="Frequency Domain Spectrum (FFT Magnitude)",
                  style='Header.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_fft, text="— Yellow (dashed): Input Spectrum  |  Green: Filtered Output Spectrum",
                  style='Card.TLabel', foreground='#abb2bf', font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=10)

        self.canvas_fft = tk.Canvas(frame_fft, bg='#181a1f', highlightthickness=0)
        self.canvas_fft.pack(fill=tk.BOTH, expand=True)

    # --- TAB 3: ANSI C CODE GENERATOR ---
    def _build_tab_code(self, parent):
        top_bar = ttk.Frame(parent, padding=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="Pure ANSI C99 Code (Direct Form II Transposed Biquad Cascade - Zero External Dependencies)",
                  font=('Segoe UI', 9, 'bold'), foreground='#98c379').pack(side=tk.LEFT)

        btn_copy = ttk.Button(top_bar, text="📋 Copy Code to Clipboard", style='Accent.TButton',
                              command=self._copy_c_code)
        btn_copy.pack(side=tk.RIGHT, padx=4)

        btn_save_c = ttk.Button(top_bar, text="💾 Save to .c File", style='Action.TButton',
                                command=self._save_c_code_file)
        btn_save_c.pack(side=tk.RIGHT, padx=4)

        # Code text display
        self.text_code = tk.Text(parent, bg='#1e2227', fg='#abb2bf', insertbackground='#ffffff',
                                 font=('Consolas', 10), relief='flat', padx=12, pady=12)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.text_code.yview)
        self.text_code.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_code.pack(fill=tk.BOTH, expand=True)

    # --- TAB 4: COEFFICIENTS & BIQUADS ---
    def _build_tab_coeffs(self, parent):
        top_bar = ttk.Frame(parent, padding=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="Detailed Second-Order Sections (Biquads) and Direct Form Polynomials",
                  font=('Segoe UI', 9, 'bold'), foreground='#e5c07b').pack(side=tk.LEFT)

        btn_save_json = ttk.Button(top_bar, text="💾 Export Coefficients JSON", style='Accent.TButton',
                                   command=self._export_coeffs_json)
        btn_save_json.pack(side=tk.RIGHT)

        # Coefficients display
        self.text_coeffs = tk.Text(parent, bg='#1e2227', fg='#abb2bf', insertbackground='#ffffff',
                                   font=('Consolas', 10), relief='flat', padx=12, pady=12)
        scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.text_coeffs.yview)
        self.text_coeffs.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text_coeffs.pack(fill=tk.BOTH, expand=True)

    # --- Event Handlers & Controller ---
    def _on_filter_type_changed(self, event=None):
        raw = self.var_filter_type.get()
        if "BSF" in raw:
            self.filter_type = 'BSF'
        elif "BPF" in raw:
            self.filter_type = 'BPF'
        elif "HPF" in raw:
            self.filter_type = 'HPF'
        else:
            self.filter_type = 'LPF'

        is_band = self.filter_type in ('BPF', 'BSF')
        if is_band:
            self.lbl_fc2.pack(anchor='w')
            self.f2_row.pack(fill=tk.X, pady=(2, 2))
            self.scale_fc2.pack(fill=tk.X, pady=(0, 8))
        else:
            self.lbl_fc2.pack_forget()
            self.f2_row.pack_forget()
            self.scale_fc2.pack_forget()

        self.recalculate_filter()
        self.run_test_signal()

    def _on_order_scale(self, val):
        self.order = int(round(float(val)))
        self.var_order_val.set(str(self.order))
        if getattr(self, '_initialized', False):
            self.recalculate_filter()
            self.run_test_signal()

    def _on_sr_changed(self, event=None):
        try:
            self.sample_rate = float(self.var_sr.get())
        except ValueError:
            self.sample_rate = 48000.0
        self.recalculate_filter()
        self.run_test_signal()

    def _on_param_changed(self, event=None):
        self.response_type = self.var_response_type.get()

        # Update visibility of ripple / stopband entries
        self.lbl_ripple.pack_forget()
        self.entry_ripple.pack_forget()
        self.lbl_stopband.pack_forget()
        self.entry_stopband.pack_forget()

        if self.response_type == 'Chebyshev I':
            self.lbl_ripple.pack(anchor='w')
            self.entry_ripple.pack(fill=tk.X, pady=(2, 6))
        elif self.response_type == 'Chebyshev II':
            self.lbl_stopband.pack(anchor='w')
            self.entry_stopband.pack(fill=tk.X, pady=(2, 6))

        try:
            self.ripple_db = float(self.entry_ripple.get())
        except ValueError:
            self.ripple_db = 0.5

        try:
            self.stopband_db = float(self.entry_stopband.get())
        except ValueError:
            self.stopband_db = 40.0

        self.recalculate_filter()
        self.run_test_signal()

    def _on_toggle_phase(self):
        self.canvas_response.toggle_phase(self.var_show_phase.get())

    # --- Sliders <-> Frequency mappings ---
    def _hz_to_slider_pos(self, f: float) -> float:
        f_min = 10.0
        f_max = self.sample_rate / 2.0
        f_clamped = max(f_min, min(f_max, f))
        return (math.log10(f_clamped) - math.log10(f_min)) / (math.log10(f_max) - math.log10(f_min))

    def _slider_pos_to_hz(self, pos: float) -> float:
        f_min = 10.0
        f_max = self.sample_rate / 2.0
        log_f = math.log10(f_min) + pos * (math.log10(f_max) - math.log10(f_min))
        return math.pow(10.0, log_f)

    def _update_slider_pos_from_hz(self, slider: ttk.Scale, f: float):
        pos = self._hz_to_slider_pos(f)
        slider.set(pos)

    def _on_scale_fc1(self, val):
        f = self._slider_pos_to_hz(float(val))
        self.f_cutoff = f
        self.entry_fc1.delete(0, tk.END)
        self.entry_fc1.insert(0, f"{f:.1f}")
        if getattr(self, '_initialized', False):
            self.recalculate_filter()
            self.run_test_signal()

    def _on_scale_fc2(self, val):
        f = self._slider_pos_to_hz(float(val))
        self.f_cutoff2 = f
        self.entry_fc2.delete(0, tk.END)
        self.entry_fc2.insert(0, f"{f:.1f}")
        if getattr(self, '_initialized', False):
            self.recalculate_filter()
            self.run_test_signal()

    def _on_fc1_entry(self, event=None):
        try:
            val = float(self.entry_fc1.get())
            self.f_cutoff = max(10.0, min(self.sample_rate / 2.0 * 0.98, val))
            self._update_slider_pos_from_hz(self.scale_fc1, self.f_cutoff)
            self.recalculate_filter()
            self.run_test_signal()
        except ValueError:
            pass

    def _on_fc2_entry(self, event=None):
        try:
            val = float(self.entry_fc2.get())
            self.f_cutoff2 = max(10.0, min(self.sample_rate / 2.0 * 0.98, val))
            self._update_slider_pos_from_hz(self.scale_fc2, self.f_cutoff2)
            self.recalculate_filter()
            self.run_test_signal()
        except ValueError:
            pass

    def _on_canvas_cutoff_drag(self, fc1: float, fc2: Optional[float]):
        """Callback from mouse drag on response canvas."""
        self.f_cutoff = fc1
        self.entry_fc1.delete(0, tk.END)
        self.entry_fc1.insert(0, f"{fc1:.1f}")
        self._update_slider_pos_from_hz(self.scale_fc1, fc1)

        if fc2 is not None:
            self.f_cutoff2 = fc2
            self.entry_fc2.delete(0, tk.END)
            self.entry_fc2.insert(0, f"{fc2:.1f}")
            self._update_slider_pos_from_hz(self.scale_fc2, fc2)

        self.recalculate_filter()
        self.run_test_signal()

    # --- Core Calculation & UI Update ---
    def recalculate_filter(self):
        try:
            self.current_filter = DigitalFilter(
                filter_type=self.filter_type,
                response_type=self.response_type,
                order=self.order,
                sample_rate=self.sample_rate,
                f_cutoff=self.f_cutoff,
                f_cutoff2=self.f_cutoff2,
                ripple_db=self.ripple_db,
                stopband_db=self.stopband_db
            )
        except Exception as e:
            messagebox.showerror("Filter Design Error", str(e))
            return

        # Generate dense logarithmic frequency points for plotting (250 points)
        f_nyq = self.sample_rate / 2.0
        log_f_min = math.log10(10.0)
        log_f_max = math.log10(f_nyq * 0.99)
        num_pts = 280
        freqs = [math.pow(10.0, log_f_min + (i / (num_pts - 1)) * (log_f_max - log_f_min)) for i in range(num_pts)]

        mags, phases, gd = self.current_filter.frequency_response(freqs)

        # Update Canvas
        fc2_val = self.f_cutoff2 if self.filter_type in ('BPF', 'BSF') else None
        self.canvas_response.set_filter_state(
            filter_type=self.filter_type,
            fc1=self.f_cutoff,
            fc2=fc2_val,
            sample_rate=self.sample_rate,
            freqs=freqs,
            mags=mags,
            phases=phases
        )

        # Update banner status
        biquad_count = len(self.current_filter.sections)
        self.lbl_stats.config(
            text=f"Sections: {biquad_count} Biquad(s) | Sample Rate: {int(self.sample_rate)} Hz | Total Poles: {biquad_count * 2}"
        )

        # Update C Code tab
        self._update_c_code_text()

        # Update Coefficients tab
        self._update_coeffs_text()

    def _update_c_code_text(self):
        if not self.current_filter:
            return
        code = self.current_filter.generate_c_code("dsp_filter")
        self.text_code.delete("1.0", tk.END)
        self.text_code.insert(tk.END, code)

    def _update_coeffs_text(self):
        if not self.current_filter:
            return
        txt = []
        txt.append("==========================================================================")
        txt.append(f"FILTER SPECIFICATION SUMMARY:")
        txt.append(f"  Type:           {self.current_filter.filter_type} ({self.current_filter.response_type})")
        txt.append(f"  Order:          {self.current_filter.order}")
        txt.append(f"  Sample Rate:    {self.current_filter.sample_rate:.1f} Hz")
        if self.filter_type in ('LPF', 'HPF'):
            txt.append(f"  Cutoff Freq:    {self.current_filter.f_cutoff:.2f} Hz")
        else:
            txt.append(f"  Band Cutoffs:   {self.current_filter.f_cutoff:.2f} Hz -> {self.current_filter.f_cutoff2:.2f} Hz")
        txt.append(f"  Total Sections: {len(self.current_filter.sections)} Biquads")
        txt.append("==========================================================================\n")

        txt.append("CASCADED SECOND-ORDER SECTIONS (BIQUADS):")
        txt.append("H(z) = prod_k [ (b0_k + b1_k*z^-1 + b2_k*z^-2) / (1 + a1_k*z^-1 + a2_k*z^-2) ]\n")
        for i, sec in enumerate(self.current_filter.sections):
            txt.append(f"--- Section {i + 1} ---")
            txt.append(f"  b0 = {sec.b0:+.10e}")
            txt.append(f"  b1 = {sec.b1:+.10e}")
            txt.append(f"  b2 = {sec.b2:+.10e}")
            txt.append(f"  a0 = 1.0000000000e+00 (normalized)")
            txt.append(f"  a1 = {sec.a1:+.10e}")
            txt.append(f"  a2 = {sec.a2:+.10e}\n")

        b_poly, a_poly = self.current_filter.get_direct_form_coefficients()
        txt.append("==========================================================================")
        txt.append("EXPANDED DIRECT FORM POLYNOMIALS:")
        txt.append("Numerator B(z):")
        for i, b in enumerate(b_poly):
            txt.append(f"  b[{i:2d}] = {b:+.10e}")
        txt.append("\nDenominator A(z):")
        for i, a in enumerate(a_poly):
            txt.append(f"  a[{i:2d}] = {a:+.10e}")

        # Phase & Delay Characteristics at key frequencies
        txt.append("\n==========================================================================")
        txt.append("PHASE & GROUP DELAY CHARACTERISTICS AT KEY FREQUENCIES:")
        txt.append(f"{'Frequency (Hz)':<16} | {'Magnitude (dB)':<16} | {'Phase (deg)':<14} | {'Group Delay (samples)':<22} | {'Time Delay (ms)':<15}")
        txt.append("-" * 92)

        probe_freqs = [10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0]
        # Include filter cutoffs
        if self.current_filter.f_cutoff not in probe_freqs and self.current_filter.f_cutoff < (self.sample_rate / 2):
            probe_freqs.append(self.current_filter.f_cutoff)
        if self.filter_type in ('BPF', 'BSF') and self.current_filter.f_cutoff2 not in probe_freqs and self.current_filter.f_cutoff2 < (self.sample_rate / 2):
            probe_freqs.append(self.current_filter.f_cutoff2)
        probe_freqs = sorted(set(f for f in probe_freqs if f <= (self.sample_rate / 2 * 0.99)))

        mags_p, phases_p, gd_p = self.current_filter.frequency_response(probe_freqs)
        for f, m, p, g in zip(probe_freqs, mags_p, phases_p, gd_p):
            t_ms = (g / self.sample_rate) * 1000.0
            cutoff_tag = ""
            if abs(f - self.current_filter.f_cutoff) < 1e-2:
                cutoff_tag = " [Fc1]"
            elif self.filter_type in ('BPF', 'BSF') and abs(f - self.current_filter.f_cutoff2) < 1e-2:
                cutoff_tag = " [Fc2]"
            txt.append(f"{f:10.1f} Hz{cutoff_tag:<5} | {m:+14.2f} dB | {p:+12.2f}° | {g:20.2f}   | {t_ms:13.3f} ms")

        self.text_coeffs.delete("1.0", tk.END)
        self.text_coeffs.insert(tk.END, "\n".join(txt))

    # --- Test Mode Execution & Plotting ---
    def run_test_signal(self):
        if not self.current_filter:
            return

        sig_type = self.var_test_sig.get()
        try:
            f1 = float(self.var_test_f1.get())
            f2 = float(self.var_test_f2.get())
            dur = max(0.005, min(0.5, float(self.var_test_dur.get())))
        except ValueError:
            f1, f2, dur = 500.0, 10000.0, 0.05

        time_vec, in_vec = generate_signal(sig_type, duration_sec=dur, sample_rate=self.sample_rate,
                                           freq1=f1, freq2=f2, amplitude=0.9)
        self.current_filter.reset()
        out_vec = self.current_filter.process_block(in_vec)

        self.test_time = time_vec
        self.test_in = in_vec
        self.test_out = out_vec

        # Calculate metrics (RMS, peak, attenuation)
        rms_in = math.sqrt(sum(x * x for x in in_vec) / max(1, len(in_vec)))
        rms_out = math.sqrt(sum(y * y for y in out_vec) / max(1, len(out_vec)))
        ratio = (rms_out / rms_in) if rms_in > 1e-9 else 1.0
        ratio_db = 20.0 * math.log10(max(ratio, 1e-6))

        self.lbl_test_metrics.config(
            text=f"Signal: {sig_type} | Samples: {len(in_vec)} | RMS In: {rms_in:.3f} | RMS Out: {rms_out:.3f} | Gain: {ratio_db:+.2f} dB"
        )

        # Draw waveform oscilloscope
        self._draw_waveform_canvas()

        # Compute & draw FFT spectrum
        self._draw_fft_canvas()

    def _draw_waveform_canvas(self):
        canvas = self.canvas_wave
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 50 or h < 50:
            canvas.after(50, self._draw_waveform_canvas)
            return

        margin_x = 45
        margin_y = 20
        plot_w = w - 2 * margin_x
        plot_h = h - 2 * margin_y
        mid_y = margin_y + plot_h / 2.0

        # Axes & grid
        canvas.create_rectangle(margin_x, margin_y, margin_x + plot_w, margin_y + plot_h,
                                fill='#21252b', outline='#2c313a')
        canvas.create_line(margin_x, mid_y, margin_x + plot_w, mid_y, fill='#3e4451', dash=(2, 4))
        canvas.create_text(margin_x - 5, mid_y, text="0.0", fill='#7f848e', font=('Segoe UI', 8), anchor='e')
        canvas.create_text(margin_x - 5, margin_y, text="+1.0", fill='#7f848e', font=('Segoe UI', 8), anchor='e')
        canvas.create_text(margin_x - 5, margin_y + plot_h, text="-1.0", fill='#7f848e', font=('Segoe UI', 8), anchor='e')

        num_pts = min(len(self.test_in), 800)
        step = max(1, len(self.test_in) // num_pts)
        pts_in = []
        pts_out = []

        scale_y = (plot_h / 2.0) * 0.9

        for i in range(0, len(self.test_in), step):
            x = margin_x + (i / max(1, len(self.test_in) - 1)) * plot_w
            y_in = mid_y - self.test_in[i] * scale_y
            y_out = mid_y - self.test_out[i] * scale_y
            pts_in.extend([x, y_in])
            pts_out.extend([x, y_out])

        if len(pts_in) >= 4:
            canvas.create_line(pts_in, fill='#56b6c2', width=1)
        if len(pts_out) >= 4:
            canvas.create_line(pts_out, fill='#98c379', width=2)

    def _draw_fft_canvas(self):
        canvas = self.canvas_fft
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 50 or h < 50:
            canvas.after(50, self._draw_fft_canvas)
            return

        margin_x = 45
        margin_y = 20
        plot_w = w - 2 * margin_x
        plot_h = h - 2 * margin_y

        canvas.create_rectangle(margin_x, margin_y, margin_x + plot_w, margin_y + plot_h,
                                fill='#21252b', outline='#2c313a')

        # Compute FFT for input and output
        f_in, db_in = compute_fft(self.test_in, self.sample_rate, max_points=1024)
        f_out, db_out = compute_fft(self.test_out, self.sample_rate, max_points=1024)

        db_min = -80.0
        db_max = 10.0

        # Draw dB lines
        for db in [-60, -40, -20, 0]:
            ratio = (db - db_min) / (db_max - db_min)
            y = margin_y + (1.0 - ratio) * plot_h
            canvas.create_line(margin_x, y, margin_x + plot_w, y, fill='#2c313a', dash=(2, 4))
            canvas.create_text(margin_x - 5, y, text=f"{db}dB", fill='#7f848e', font=('Segoe UI', 8), anchor='e')

        # Plot curves (linear freq axis up to Nyquist)
        f_nyq = self.sample_rate / 2.0

        coords_in = []
        for f, db in zip(f_in, db_in):
            x = margin_x + (f / f_nyq) * plot_w
            ratio = (max(db_min, min(db_max, db)) - db_min) / (db_max - db_min)
            y = margin_y + (1.0 - ratio) * plot_h
            coords_in.extend([x, y])

        coords_out = []
        for f, db in zip(f_out, db_out):
            x = margin_x + (f / f_nyq) * plot_w
            ratio = (max(db_min, min(db_max, db)) - db_min) / (db_max - db_min)
            y = margin_y + (1.0 - ratio) * plot_h
            coords_out.extend([x, y])

        if len(coords_in) >= 4:
            canvas.create_line(coords_in, fill='#e5c07b', width=1, dash=(3, 2))
        if len(coords_out) >= 4:
            canvas.create_line(coords_out, fill='#98c379', width=2)

    # --- Actions & Exports ---
    def _show_c_code_tab(self):
        self.notebook.select(self.tab_code)

    def _copy_c_code(self):
        code = self.text_code.get("1.0", tk.END)
        self.clipboard_clear()
        self.clipboard_append(code)
        messagebox.showinfo("Copied", "ANSI C Code successfully copied to clipboard!")

    def _save_c_code_file(self):
        code = self.text_code.get("1.0", tk.END)
        fpath = filedialog.asksaveasfilename(defaultextension=".c",
                                             filetypes=[("C Source Files", "*.c"), ("All Files", "*.*")],
                                             initialfile=f"filter_{self.filter_type.lower()}_{self.response_type.lower().replace(' ', '_')}.c")
        if fpath:
            with open(fpath, "w", encoding="utf-8") as f:
                f.write(code)
            messagebox.showinfo("Saved", f"C Code successfully written to:\n{fpath}")

    def _export_menu_action(self):
        """Open export selector dialog."""
        dlg = tk.Toplevel(self)
        dlg.title("Export Filter & Test Data")
        dlg.geometry("400x260")
        dlg.configure(bg='#1e2227')
        dlg.transient(self)
        dlg.grab_set()

        ttk.Label(dlg, text="Choose Export Format:", font=('Segoe UI', 10, 'bold')).pack(pady=(15, 10))

        ttk.Button(dlg, text="1. Standalone ANSI C Code (.c)", style='Accent.TButton',
                   command=lambda: [dlg.destroy(), self._save_c_code_file()]).pack(fill=tk.X, padx=30, pady=4)

        ttk.Button(dlg, text="2. Time Domain Signal Test Data (.csv)", style='Accent.TButton',
                   command=lambda: [dlg.destroy(), self._export_test_csv()]).pack(fill=tk.X, padx=30, pady=4)

        ttk.Button(dlg, text="3. Filter Config & Signal Data (.json)", style='Accent.TButton',
                   command=lambda: [dlg.destroy(), self._export_test_json()]).pack(fill=tk.X, padx=30, pady=4)

        ttk.Button(dlg, text="4. Filtered Audio Signal (.wav)", style='Accent.TButton',
                   command=lambda: [dlg.destroy(), self._export_test_wav()]).pack(fill=tk.X, padx=30, pady=4)

    def _export_test_csv(self):
        if not self.test_time:
            messagebox.showwarning("No Data", "Please run the test verification first.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                                             initialfile="filter_test_data.csv")
        if fpath:
            export_data_to_csv(fpath, self.test_time, self.test_in, self.test_out)
            messagebox.showinfo("Export Successful", f"Test data saved to:\n{fpath}")

    def _export_test_json(self):
        if not self.test_time or not self.current_filter:
            messagebox.showwarning("No Data", "Please run the test verification first.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                                             initialfile="filter_test_data.json")
        if fpath:
            meta = {
                "filter_type": self.filter_type,
                "response_type": self.response_type,
                "order": self.order,
                "sample_rate": self.sample_rate,
                "f_cutoff": self.f_cutoff,
                "f_cutoff2": self.f_cutoff2,
                "ripple_db": self.ripple_db,
                "stopband_db": self.stopband_db
            }
            export_data_to_json(fpath, meta, self.test_time, self.test_in, self.test_out)
            messagebox.showinfo("Export Successful", f"JSON data saved to:\n{fpath}")

    def _export_test_wav(self):
        if not self.test_out:
            messagebox.showwarning("No Data", "Please run the test verification first.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".wav",
                                             filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")],
                                             initialfile="filtered_signal.wav")
        if fpath:
            export_audio_to_wav(fpath, self.test_out, int(self.sample_rate))
            messagebox.showinfo("Export Successful", f"WAV Audio saved to:\n{fpath}")

    def _export_coeffs_json(self):
        if not self.current_filter:
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                                             initialfile="filter_biquads.json")
        if fpath:
            sections_data = []
            for i, sec in enumerate(self.current_filter.sections):
                sections_data.append({
                    "section": i + 1,
                    "b0": sec.b0, "b1": sec.b1, "b2": sec.b2,
                    "a0": sec.a0, "a1": sec.a1, "a2": sec.a2
                })
            b_poly, a_poly = self.current_filter.get_direct_form_coefficients()
            data = {
                "filter_type": self.filter_type,
                "response_type": self.response_type,
                "order": self.order,
                "sample_rate": self.sample_rate,
                "f_cutoff": self.f_cutoff,
                "f_cutoff2": self.f_cutoff2,
                "sections": sections_data,
                "direct_form": {
                    "b": b_poly,
                    "a": a_poly
                }
            }
            with open(fpath, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, indent=2)
            messagebox.showinfo("Export Successful", f"Coefficients exported to:\n{fpath}")


def main():
    app = FilterDesignApp()
    app.mainloop()


if __name__ == "__main__":
    main()
