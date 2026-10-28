"""
Audio Filter Design Studio - Main Application GUI
A modern, light-themed desktop application for interactive audio filter design.
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
import tempfile
from typing import Optional, List
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from filter_engine import DigitalFilter, FIRFilter
from filter_canvas import InteractiveResponseCanvas
from signal_testbench import (generate_signal, compute_fft,
                              export_data_to_csv, export_data_to_json, export_audio_to_wav,
                              audio_to_wav_bytes, ContinuousAudioEngine)


class FilterDesignApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Audio Filter Design Studio - LPF / HPF / BPF / BSF")
        self.geometry("1240x860")
        self.minsize(980, 680)

        # Apply light theme styling
        self.configure(bg='#f6f8fa')
        self._setup_styles()

        # Application State
        self.sample_rate = 48000.0
        self.filter_family = 'IIR'  # 'IIR' or 'FIR'
        self.filter_type = 'LPF'
        self.response_type = 'Butterworth'
        self.order = 4
        self.fir_taps = 65
        self.fir_window = 'Hamming'
        self.f_cutoff = 1000.0
        self.f_cutoff2 = 4000.0
        self.ripple_db = 0.5
        self.stopband_db = 40.0
        self.kaiser_beta = 5.0
        self.gaussian_sigma = 0.4
        self.tukey_alpha = 0.5

        self.current_filter = None

        # Test mode data cache
        self.test_time = []
        self.test_in = []
        self.test_out = []

        # Real-time continuous audio engine (Falstad DFilter style)
        self.audio_engine = ContinuousAudioEngine(sample_rate=self.sample_rate, block_size=1024)
        self.audio_engine.set_volume(0.70)
        self.audio_engine.set_sound_enabled(True)

        self._initialized = False

        # Build UI layout
        self._build_layout()

        self._initialized = True

        # Initial filter calculation
        self.recalculate_filter()
        self.audio_engine.set_signal_params(signal_type=self.var_test_sig.get(), freq1=500.0, freq2=5000.0)
        self.audio_engine.start()

        # Window close handler for graceful audio device release
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Start live continuous visualizer update loop
        self._schedule_live_visuals()

    def _setup_styles(self):
        style = ttk.Style(self)
        style.theme_use('clam')

        # Color palette (Modern Clean Light theme)
        bg = '#f6f8fa'
        card_bg = '#ffffff'
        fg = '#24292f'
        accent = '#0969da'
        active_tab = '#ffffff'

        style.configure('.', background=bg, foreground=fg, font=('Segoe UI', 9))
        style.configure('TNotebook', background=bg, borderwidth=0)
        style.configure('TNotebook.Tab', background='#eaeef2', foreground='#57606a',
                        padding=[14, 6], font=('Segoe UI', 9, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', active_tab), ('active', '#f3f4f6')],
                  foreground=[('selected', accent), ('active', '#24292f')])

        style.configure('TFrame', background=bg)
        style.configure('Card.TFrame', background=card_bg, relief='flat')
        style.configure('TLabel', background=bg, foreground=fg)
        style.configure('Card.TLabel', background=card_bg, foreground=fg)
        style.configure('Header.TLabel', background=card_bg, foreground='#1f2328', font=('Segoe UI', 10, 'bold'))
        style.configure('Title.TLabel', background=bg, foreground='#1a7f37', font=('Segoe UI', 12, 'bold'))

        style.configure('TCombobox', fieldbackground='#ffffff', background='#f6f8fa',
                        foreground='#24292f', arrowcolor='#57606a', padding=4)
        style.map('TCombobox', fieldbackground=[('readonly', '#ffffff')],
                  foreground=[('readonly', '#24292f')])

        style.configure('TEntry', fieldbackground='#ffffff', foreground='#24292f', insertcolor='#24292f')
        style.configure('TScale', background=card_bg, troughcolor='#eaeef2')

        style.configure('Accent.TButton', background='#eaeef2', foreground='#24292f',
                        font=('Segoe UI', 9, 'bold'), borderwidth=1, padding=5)
        style.map('Accent.TButton',
                  background=[('pressed', '#0969da'), ('active', '#d0d7de')],
                  foreground=[('pressed', '#ffffff'), ('active', '#0969da')])

        style.configure('Action.TButton', background='#1a7f37', foreground='#ffffff',
                        font=('Segoe UI', 9, 'bold'), borderwidth=0, padding=5)
        style.map('Action.TButton',
                  background=[('pressed', '#116329'), ('active', '#2da44e')])

    def _build_layout(self):
        # Top Header Bar
        top_bar = ttk.Frame(self, padding=(16, 10))
        top_bar.pack(side=tk.TOP, fill=tk.X)

        title_lbl = ttk.Label(top_bar, text="🎵 Audio Filter Design Studio", style='Title.TLabel')
        title_lbl.pack(side=tk.LEFT)

        subtitle_lbl = ttk.Label(top_bar, text="— Interactive DSP Biquad Synthesizer & C-Code Generator",
                                 foreground='#57606a', font=('Segoe UI', 9, 'italic'))
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

        # Filter Architecture (IIR vs FIR)
        ttk.Label(sidebar, text="Filter Architecture:", style='Card.TLabel').pack(anchor='w')
        self.var_filter_family = tk.StringVar(value="IIR (Infinite Impulse Response)")
        self.cb_family = ttk.Combobox(sidebar, textvariable=self.var_filter_family,
                                      values=["IIR (Infinite Impulse Response)",
                                              "FIR (Finite Impulse Response)"],
                                      state="readonly")
        self.cb_family.pack(fill=tk.X, pady=(2, 8))
        self.cb_family.bind("<<ComboboxSelected>>", self._on_filter_family_changed)

        # Filter Type (LPF, HPF, BPF, BSF)
        ttk.Label(sidebar, text="Filter Type:", style='Card.TLabel').pack(anchor='w')
        self.var_filter_type = tk.StringVar(value="LPF")
        cb_ftype = ttk.Combobox(sidebar, textvariable=self.var_filter_type,
                                values=["LPF", "HPF", "BPF", "BSF (Notch)"], state="readonly")
        cb_ftype.pack(fill=tk.X, pady=(2, 8))
        cb_ftype.bind("<<ComboboxSelected>>", self._on_filter_type_changed)

        # Response Type / Window Function
        f_resp_hdr = ttk.Frame(sidebar, style='Card.TFrame')
        f_resp_hdr.pack(fill=tk.X)
        self.lbl_response_heading = ttk.Label(f_resp_hdr, text="Response Approximation:", style='Card.TLabel')
        self.lbl_response_heading.pack(side=tk.LEFT)

        self.btn_select_window = ttk.Button(f_resp_hdr, text="🔍 Browse Windows...",
                                            style='Accent.TButton',
                                            command=self._open_window_selection_dialog)
        # Hidden initially for IIR, shown for FIR

        self.var_response_type = tk.StringVar(value="Butterworth")
        self.cb_rtype = ttk.Combobox(sidebar, textvariable=self.var_response_type,
                                     values=["Butterworth", "Chebyshev I", "Chebyshev II", "Bessel"],
                                     state="readonly")
        self.cb_rtype.pack(fill=tk.X, pady=(2, 8))
        self.cb_rtype.bind("<<ComboboxSelected>>", self._on_param_changed)

        # Filter Order / Tap Count
        self.f_order_row = ttk.Frame(sidebar, style='Card.TFrame')
        self.f_order_row.pack(fill=tk.X, pady=(0, 4))
        self.lbl_order_title = ttk.Label(self.f_order_row, text="Order:", style='Card.TLabel')
        self.lbl_order_title.pack(side=tk.LEFT)
        self.var_order_val = tk.StringVar(value="4")
        self.lbl_order_num = ttk.Label(self.f_order_row, textvariable=self.var_order_val,
                                       foreground='#0969da', font=('Segoe UI', 9, 'bold'))
        self.lbl_order_num.pack(side=tk.RIGHT)

        self.scale_order = ttk.Scale(sidebar, from_=1, to=10, style='TScale',
                                     command=self._on_order_scale)
        self.scale_order.set(4)
        self.scale_order.pack(fill=tk.X, pady=(0, 8))

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
        self.f2_row = ttk.Frame(sidebar, style='Card.TFrame')
        self.var_fc2 = tk.DoubleVar(value=4000.0)
        self.entry_fc2 = ttk.Entry(self.f2_row, width=10)
        self.entry_fc2.insert(0, "4000.0")
        self.entry_fc2.pack(side=tk.LEFT)
        self.entry_fc2.bind("<Return>", self._on_fc2_entry)
        self.entry_fc2.bind("<FocusOut>", self._on_fc2_entry)
        ttk.Label(self.f2_row, text=" Hz", style='Card.TLabel', font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=4)

        self.scale_fc2 = ttk.Scale(sidebar, from_=0.0, to=1.0, style='TScale',
                                   command=self._on_scale_fc2)
        self._update_slider_pos_from_hz(self.scale_fc2, 4000.0)

        # Dynamic Ripple / Stopband / Kaiser Beta options
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

        self.lbl_kaiser = ttk.Label(self.frame_dynamic_params, text="Kaiser Beta:", style='Card.TLabel')
        self.entry_kaiser = ttk.Entry(self.frame_dynamic_params, width=8)
        self.entry_kaiser.insert(0, "5.0")
        self.entry_kaiser.bind("<Return>", self._on_param_changed)

        self.lbl_gaussian = ttk.Label(self.frame_dynamic_params, text="Gaussian Sigma (0.1 - 1.0):", style='Card.TLabel')
        self.entry_gaussian = ttk.Entry(self.frame_dynamic_params, width=8)
        self.entry_gaussian.insert(0, "0.4")
        self.entry_gaussian.bind("<Return>", self._on_param_changed)

        self.lbl_tukey = ttk.Label(self.frame_dynamic_params, text="Tukey Alpha (0.0 - 1.0):", style='Card.TLabel')
        self.entry_tukey = ttk.Entry(self.frame_dynamic_params, width=8)
        self.entry_tukey.insert(0, "0.5")
        self.entry_tukey.bind("<Return>", self._on_param_changed)

        # Checkbox for Phase curve
        self.var_show_phase = tk.BooleanVar(value=False)
        chk_phase = tk.Checkbutton(sidebar, text="Show Phase Response", variable=self.var_show_phase,
                                   bg='#ffffff', fg='#24292f', selectcolor='#ffffff',
                                   activebackground='#ffffff', activeforeground='#0969da',
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
                             style='Card.TLabel', foreground='#1a7f37', font=('Segoe UI', 9, 'bold'))
        info_lbl.pack(side=tk.LEFT)

        self.lbl_stats = ttk.Label(banner, text="", style='Card.TLabel', foreground='#0969da')
        self.lbl_stats.pack(side=tk.RIGHT)

        # Interactive Response Canvas
        self.canvas_response = InteractiveResponseCanvas(parent, on_cutoff_dragged=self._on_canvas_cutoff_drag)
        self.canvas_response.pack(fill=tk.BOTH, expand=True)

    # --- TAB 2: TEST MODE & VERIFICATION ---
    def _build_tab_test(self, parent):
        # Top Control Bar for Signal Generation & Audio Controls
        test_ctrls = ttk.Frame(parent, style='Card.TFrame', padding=10)
        test_ctrls.pack(side=tk.TOP, fill=tk.X, pady=(0, 6))

        # Row 1: Signal selection & Real-Time Frequency Slider
        r1 = ttk.Frame(test_ctrls, style='Card.TFrame')
        r1.pack(fill=tk.X, pady=(0, 6))

        ttk.Label(r1, text="Input Signal:", style='Card.TLabel', font=('Segoe UI', 9, 'bold')).pack(side=tk.LEFT, padx=(0, 6))
        self.var_test_sig = tk.StringVar(value="Sweep (Log)")
        cb_test_sig = ttk.Combobox(r1, textvariable=self.var_test_sig, width=14, state="readonly",
                                   values=["Sine", "Multi-tone", "Sweep (Log)", "Sweep (Linear)",
                                           "White Noise", "Pink Noise", "Square", "Triangle", "Sawtooth",
                                           "Periodic Noise", "Impulse"])
        cb_test_sig.pack(side=tk.LEFT, padx=(0, 12))
        cb_test_sig.bind("<<ComboboxSelected>>", self._on_test_signal_param_changed)

        ttk.Label(r1, text="Input Freq 1 (Hz):", style='Card.TLabel').pack(side=tk.LEFT, padx=(0, 4))
        self.var_test_f1 = tk.StringVar(value="500.0")
        self.entry_test_f1 = ttk.Entry(r1, textvariable=self.var_test_f1, width=7)
        self.entry_test_f1.pack(side=tk.LEFT, padx=(0, 4))
        self.entry_test_f1.bind("<Return>", self._on_entry_test_f1)
        self.entry_test_f1.bind("<FocusOut>", self._on_entry_test_f1)

        self.scale_test_f1 = ttk.Scale(r1, from_=0.0, to=1.0, style='TScale', length=120,
                                       command=self._on_scale_test_f1)
        self.scale_test_f1.pack(side=tk.LEFT, padx=(0, 12))
        self._update_slider_pos_from_hz(self.scale_test_f1, 500.0)

        self.lbl_test_f2 = ttk.Label(r1, text="Freq 2 / End (Hz):", style='Card.TLabel')
        self.lbl_test_f2.pack(side=tk.LEFT, padx=(0, 4))
        self.var_test_f2 = tk.StringVar(value="5000.0")
        self.entry_test_f2 = ttk.Entry(r1, textvariable=self.var_test_f2, width=7)
        self.entry_test_f2.pack(side=tk.LEFT, padx=(0, 8))
        self.entry_test_f2.bind("<Return>", self._on_test_signal_param_changed)
        self.entry_test_f2.bind("<FocusOut>", self._on_test_signal_param_changed)

        # Row 2: Continuous Sound Controls, Volume Slider & Live Metrics
        r2 = ttk.Frame(test_ctrls, style='Card.TFrame')
        r2.pack(fill=tk.X, pady=(2, 0))

        self.var_test_sound = tk.BooleanVar(value=True)
        self.chk_test_sound = ttk.Checkbutton(r2, text="🔊 Sound On", variable=self.var_test_sound,
                                              command=self._on_test_sound_toggle)
        self.chk_test_sound.pack(side=tk.LEFT, padx=(0, 10))

        ttk.Label(r2, text="Volume:", style='Card.TLabel').pack(side=tk.LEFT, padx=(0, 4))
        self.var_test_volume = tk.DoubleVar(value=70.0)
        self.scale_test_volume = ttk.Scale(r2, from_=0, to=100, variable=self.var_test_volume,
                                           command=self._on_test_volume_changed, length=120)
        self.scale_test_volume.pack(side=tk.LEFT)
        self.lbl_test_volume = ttk.Label(r2, text="70%", style='Card.TLabel', width=4)
        self.lbl_test_volume.pack(side=tk.LEFT, padx=(4, 15))

        self.lbl_test_metrics = ttk.Label(r2, text="Metrics: Initializing...",
                                          style='Card.TLabel', foreground='#0969da', font=('Segoe UI', 9))
        self.lbl_test_metrics.pack(side=tk.LEFT)

        btn_save_wav = ttk.Button(r2, text="Export Snapshot WAV", style='Accent.TButton',
                                  command=self._export_test_wav)
        btn_save_wav.pack(side=tk.RIGHT, padx=4)

        btn_save_csv = ttk.Button(r2, text="Export Data CSV", style='Accent.TButton',
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
        ttk.Label(header_wave, text="— Blue: Input Signal  |  Green: Filtered Output",
                  style='Card.TLabel', foreground='#57606a', font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=10)

        self.canvas_wave = tk.Canvas(frame_wave, bg='#ffffff', highlightthickness=0)
        self.canvas_wave.pack(fill=tk.BOTH, expand=True)

        # Lower plot: FFT Frequency Spectrum (Input vs Output)
        frame_fft = ttk.Frame(vis_split, style='Card.TFrame')
        vis_split.add(frame_fft, weight=1)

        header_fft = ttk.Frame(frame_fft, style='Card.TFrame', padding=(8, 4))
        header_fft.pack(side=tk.TOP, fill=tk.X)
        ttk.Label(header_fft, text="Frequency Domain Spectrum (FFT Magnitude)",
                  style='Header.TLabel').pack(side=tk.LEFT)
        ttk.Label(header_fft, text="— Amber (dashed): Input Spectrum  |  Green: Filtered Output Spectrum",
                  style='Card.TLabel', foreground='#57606a', font=('Segoe UI', 8)).pack(side=tk.LEFT, padx=10)

        self.canvas_fft = tk.Canvas(frame_fft, bg='#ffffff', highlightthickness=0)
        self.canvas_fft.pack(fill=tk.BOTH, expand=True)

    # --- TAB 3: ANSI C CODE GENERATOR ---
    def _build_tab_code(self, parent):
        top_bar = ttk.Frame(parent, padding=8)
        top_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top_bar, text="Pure ANSI C99 Code (Direct Form II Transposed Biquad Cascade - Zero External Dependencies)",
                  font=('Segoe UI', 9, 'bold'), foreground='#1a7f37').pack(side=tk.LEFT)

        btn_copy = ttk.Button(top_bar, text="📋 Copy Code to Clipboard", style='Accent.TButton',
                              command=self._copy_c_code)
        btn_copy.pack(side=tk.RIGHT, padx=4)

        btn_save_c = ttk.Button(top_bar, text="💾 Save to .c File", style='Action.TButton',
                                command=self._save_c_code_file)
        btn_save_c.pack(side=tk.RIGHT, padx=4)

        # Code text display
        self.text_code = tk.Text(parent, bg='#ffffff', fg='#24292f', insertbackground='#24292f',
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
                  font=('Segoe UI', 9, 'bold'), foreground='#1f2328').pack(side=tk.LEFT)

        btn_save_json = ttk.Button(top_bar, text="💾 Export Coefficients JSON", style='Accent.TButton',
                                   command=self._export_coeffs_json)
        btn_save_json.pack(side=tk.RIGHT)

        # Coefficients display
        self.text_coeffs = tk.Text(parent, bg='#ffffff', fg='#24292f', insertbackground='#24292f',
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

    def _on_filter_family_changed(self, event=None):
        raw = self.var_filter_family.get()
        if "FIR" in raw:
            self.filter_family = 'FIR'
            self.lbl_response_heading.config(text="Window Function:")
            self.btn_select_window.pack(side=tk.RIGHT, pady=(0, 2))
            fir_windows = FIRFilter.KNOWN_WINDOWS
            self.cb_rtype.config(values=fir_windows)
            if self.var_response_type.get() not in fir_windows:
                self.var_response_type.set("Hamming")
            self.lbl_order_title.config(text="Taps (Length):")
            # For FIR, order slider controls taps 7 to 255 (odd)
            self.scale_order.config(from_=7, to=255)
            self.scale_order.set(self.fir_taps if self.fir_taps else 65)
            self.var_order_val.set(f"{int(self.scale_order.get())} taps (ord {int(self.scale_order.get())-1})")
        else:
            self.filter_family = 'IIR'
            self.lbl_response_heading.config(text="Response Approximation:")
            self.btn_select_window.pack_forget()
            iir_responses = ["Butterworth", "Chebyshev I", "Chebyshev II", "Bessel"]
            self.cb_rtype.config(values=iir_responses)
            if self.var_response_type.get() not in iir_responses:
                self.var_response_type.set("Butterworth")
            self.lbl_order_title.config(text="Order:")
            self.scale_order.config(from_=1, to=10)
            self.scale_order.set(self.order if self.order <= 10 else 4)
            self.var_order_val.set(str(int(self.scale_order.get())))

        self._update_dynamic_param_visibility()
        self.recalculate_filter()
        self.run_test_signal()

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
        num = int(round(float(val)))
        if self.filter_family == 'FIR':
            if num % 2 == 0:
                num += 1
            self.fir_taps = num
            self.var_order_val.set(f"{num} taps (ord {num-1})")
        else:
            self.order = num
            self.var_order_val.set(str(self.order))

        if getattr(self, '_initialized', False):
            self.recalculate_filter()
            self.run_test_signal()

    def _on_sr_changed(self, event=None):
        try:
            self.sample_rate = float(self.var_sr.get())
        except ValueError:
            self.sample_rate = 48000.0
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_sample_rate(self.sample_rate)
        self.recalculate_filter()

    def _update_dynamic_param_visibility(self):
        self.lbl_ripple.pack_forget()
        self.entry_ripple.pack_forget()
        self.lbl_stopband.pack_forget()
        self.entry_stopband.pack_forget()
        self.lbl_kaiser.pack_forget()
        self.entry_kaiser.pack_forget()
        self.lbl_gaussian.pack_forget()
        self.entry_gaussian.pack_forget()
        self.lbl_tukey.pack_forget()
        self.entry_tukey.pack_forget()

        if self.filter_family == 'IIR':
            if self.response_type == 'Chebyshev I':
                self.lbl_ripple.pack(anchor='w')
                self.entry_ripple.pack(fill=tk.X, pady=(2, 6))
            elif self.response_type == 'Chebyshev II':
                self.lbl_stopband.pack(anchor='w')
                self.entry_stopband.pack(fill=tk.X, pady=(2, 6))
        else: # FIR
            resp_lower = self.response_type.lower()
            if 'kaiser' in resp_lower:
                self.lbl_kaiser.pack(anchor='w')
                self.entry_kaiser.pack(fill=tk.X, pady=(2, 6))
            elif 'gaussian' in resp_lower:
                self.lbl_gaussian.pack(anchor='w')
                self.entry_gaussian.pack(fill=tk.X, pady=(2, 6))
            elif 'tukey' in resp_lower:
                self.lbl_tukey.pack(anchor='w')
                self.entry_tukey.pack(fill=tk.X, pady=(2, 6))

    def _open_window_selection_dialog(self):
        """Open detailed window function selection & comparison dialog."""
        dlg = tk.Toplevel(self)
        dlg.title("Select FIR Window Function")
        dlg.geometry("520x460")
        dlg.configure(bg='#f6f8fa')
        dlg.transient(self)
        dlg.grab_set()

        hdr = ttk.Frame(dlg, style='Card.TFrame', padding=12)
        hdr.pack(fill=tk.X)
        ttk.Label(hdr, text="FIR Window Functions", font=('Segoe UI', 11, 'bold'),
                  foreground='#0969da').pack(anchor='w')
        ttk.Label(hdr, text="Choose a window function to control side-lobe suppression and transition roll-off:",
                  foreground='#57606a', font=('Segoe UI', 9)).pack(anchor='w', pady=(2, 0))

        content_frame = ttk.Frame(dlg, padding=(12, 8))
        content_frame.pack(fill=tk.BOTH, expand=True)

        list_frame = ttk.Frame(content_frame)
        list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        lb_windows = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                font=('Segoe UI', 10), selectmode=tk.SINGLE,
                                bg='#ffffff', fg='#24292f',
                                selectbackground='#0969da', selectforeground='#ffffff',
                                activestyle='none', relief='flat', borderwidth=1)
        scrollbar.config(command=lb_windows.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        lb_windows.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        for w in FIRFilter.KNOWN_WINDOWS:
            lb_windows.insert(tk.END, w)

        cur_win = self.var_response_type.get()
        if cur_win in FIRFilter.KNOWN_WINDOWS:
            idx = FIRFilter.KNOWN_WINDOWS.index(cur_win)
            lb_windows.selection_set(idx)
            lb_windows.see(idx)

        info_frame = ttk.Frame(content_frame, style='Card.TFrame', padding=10, width=220)
        info_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        info_frame.pack_propagate(False)

        lbl_win_name = ttk.Label(info_frame, text="Window Details", font=('Segoe UI', 10, 'bold'),
                                 foreground='#1f2328')
        lbl_win_name.pack(anchor='w', pady=(0, 6))

        txt_info = tk.Text(info_frame, wrap=tk.WORD, bg='#ffffff', fg='#24292f',
                           font=('Segoe UI', 9), relief='flat', height=12)
        txt_info.pack(fill=tk.BOTH, expand=True)

        def update_desc(evt=None):
            sel = lb_windows.curselection()
            if sel:
                chosen = lb_windows.get(sel[0])
                lbl_win_name.config(text=chosen)
                desc = FIRFilter.WINDOW_DESCRIPTIONS.get(chosen, "Standard FIR windowing function.")
                txt_info.delete("1.0", tk.END)
                txt_info.insert(tk.END, f"{desc}\n\n")
                if chosen == "Kaiser":
                    txt_info.insert(tk.END, "Parameters:\n• Beta: controls side-lobe level vs main-lobe width.\nTypical beta: 4.0 (50 dB) to 9.0 (90 dB).")
                elif chosen == "Gaussian":
                    txt_info.insert(tk.END, "Parameters:\n• Sigma: standard deviation (0.1 to 1.0).\nSmaller sigma = wider main lobe, lower side lobes.")
                elif chosen == "Tukey":
                    txt_info.insert(tk.END, "Parameters:\n• Alpha: fraction of taper (0.0 = Rectangular, 1.0 = Hann).")

        lb_windows.bind("<<ListboxSelect>>", update_desc)
        update_desc()

        btn_row = ttk.Frame(dlg, padding=12)
        btn_row.pack(fill=tk.X, side=tk.BOTTOM)

        def apply_choice():
            sel = lb_windows.curselection()
            if sel:
                chosen = lb_windows.get(sel[0])
                self.var_response_type.set(chosen)
                self.fir_window = chosen
                self._update_dynamic_param_visibility()
                self.recalculate_filter()
                self.run_test_signal()
            dlg.destroy()

        lb_windows.bind("<Double-Button-1>", lambda e: apply_choice())

        ttk.Button(btn_row, text="Cancel", style='Accent.TButton',
                   command=dlg.destroy).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_row, text="✓ Select Window", style='Action.TButton',
                   command=apply_choice).pack(side=tk.RIGHT, padx=4)

    def _on_param_changed(self, event=None):
        self.response_type = self.var_response_type.get()
        if self.filter_family == 'FIR':
            self.fir_window = self.response_type

        self._update_dynamic_param_visibility()

        try:
            self.ripple_db = float(self.entry_ripple.get())
        except ValueError:
            self.ripple_db = 0.5

        try:
            self.stopband_db = float(self.entry_stopband.get())
        except ValueError:
            self.stopband_db = 40.0

        try:
            self.kaiser_beta = float(self.entry_kaiser.get())
        except ValueError:
            self.kaiser_beta = 5.0

        try:
            self.gaussian_sigma = float(self.entry_gaussian.get())
        except ValueError:
            self.gaussian_sigma = 0.4

        try:
            self.tukey_alpha = float(self.entry_tukey.get())
        except ValueError:
            self.tukey_alpha = 0.5

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
            if self.filter_family == 'FIR':
                self.current_filter = FIRFilter(
                    filter_type=self.filter_type,
                    num_taps=self.fir_taps,
                    sample_rate=self.sample_rate,
                    f_cutoff=self.f_cutoff,
                    f_cutoff2=self.f_cutoff2,
                    window_name=self.fir_window,
                    kaiser_beta=self.kaiser_beta,
                    gaussian_sigma=self.gaussian_sigma,
                    tukey_alpha=self.tukey_alpha
                )
            else:
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
        if self.filter_family == 'FIR':
            self.lbl_stats.config(
                text=f"FIR Type I | Taps: {self.current_filter.num_taps} | Order: {self.current_filter.order} | Delay: {(self.current_filter.num_taps-1)/2:.1f} smp (Linear Phase)"
            )
        else:
            biquad_count = len(self.current_filter.sections)
            self.lbl_stats.config(
                text=f"Sections: {biquad_count} Biquad(s) | Sample Rate: {int(self.sample_rate)} Hz | Total Poles: {biquad_count * 2}"
            )

        # Update real-time continuous audio engine
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_filter(self.current_filter)

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
        txt.append(f"  Family:         {getattr(self.current_filter, 'filter_family', 'IIR')}")
        txt.append(f"  Type:           {self.current_filter.filter_type} ({self.current_filter.response_type})")
        txt.append(f"  Order:          {self.current_filter.order}")
        if getattr(self.current_filter, 'filter_family', 'IIR') == 'FIR':
            txt.append(f"  Taps (Length):  {self.current_filter.num_taps}")
            txt.append(f"  Linear Phase:   Yes (Constant Group Delay = {(self.current_filter.num_taps - 1) / 2.0:.1f} samples)")
        txt.append(f"  Sample Rate:    {self.current_filter.sample_rate:.1f} Hz")
        if self.filter_type in ('LPF', 'HPF'):
            txt.append(f"  Cutoff Freq:    {self.current_filter.f_cutoff:.2f} Hz")
        else:
            txt.append(f"  Band Cutoffs:   {self.current_filter.f_cutoff:.2f} Hz -> {self.current_filter.f_cutoff2:.2f} Hz")
        if getattr(self.current_filter, 'filter_family', 'IIR') == 'IIR':
            txt.append(f"  Total Sections: {len(self.current_filter.sections)} Biquads")
        txt.append("==========================================================================\n")

        if getattr(self.current_filter, 'filter_family', 'IIR') == 'IIR':
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
        else:
            txt.append("FINITE IMPULSE RESPONSE (FIR) COEFFICIENTS:")
            txt.append("H(z) = sum_{n=0}^{N-1} h[n] * z^{-n}\n")
            for i, tap in enumerate(self.current_filter.taps):
                txt.append(f"  h[{i:3d}] = {tap:+.10e}")
            txt.append("")

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

    # --- Test Mode Continuous Audio & Plotting ---
    def _on_test_sound_toggle(self):
        enabled = self.var_test_sound.get()
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_sound_enabled(enabled)

    def _on_test_volume_changed(self, value):
        vol = float(value)
        self.lbl_test_volume.configure(text=f"{int(vol)}%")
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_volume(vol / 100.0)

    def _on_scale_test_f1(self, val):
        f = self._slider_pos_to_hz(float(val))
        self.entry_test_f1.delete(0, tk.END)
        self.entry_test_f1.insert(0, f"{f:.1f}")
        self._on_test_signal_param_changed()

    def _on_entry_test_f1(self, event=None):
        try:
            val = float(self.entry_test_f1.get())
            self._update_slider_pos_from_hz(self.scale_test_f1, val)
            self._on_test_signal_param_changed()
        except ValueError:
            pass

    def _on_test_signal_param_changed(self, event=None):
        sig_type = self.var_test_sig.get()
        try:
            f1 = float(self.var_test_f1.get())
        except ValueError:
            f1 = 500.0
        try:
            f2 = float(self.var_test_f2.get())
        except ValueError:
            f2 = 5000.0
        if hasattr(self, 'audio_engine'):
            self.audio_engine.set_signal_params(signal_type=sig_type, freq1=f1, freq2=f2)

    def _schedule_live_visuals(self):
        self._update_live_visuals()
        self.after(35, self._schedule_live_visuals)

    def _update_live_visuals(self):
        if not getattr(self, '_initialized', False):
            return
        try:
            in_vec, out_vec, metrics = self.audio_engine.get_snapshot()
            self.test_in = in_vec
            self.test_out = out_vec
            self.test_time = [i / self.sample_rate for i in range(len(in_vec))]

            cur_tab = self.notebook.select()
            if cur_tab == str(self.tab_test):
                sig_type = self.var_test_sig.get()
                unstable_flag = " | ⚠️ UNSTABLE" if metrics.get("unstable") else ""
                self.lbl_test_metrics.config(
                    text=f"Signal: {sig_type} | RMS In: {metrics['rms_in']:.3f} | RMS Out: {metrics['rms_out']:.3f} | Gain: {metrics['gain_db']:+.2f} dB | Peak: {metrics['peak_out']:.2f}{unstable_flag}"
                )
                self._draw_waveform_canvas()
                self._draw_fft_canvas()
        except Exception:
            pass

    def _on_close(self):
        try:
            if hasattr(self, 'audio_engine'):
                self.audio_engine.stop()
        except Exception:
            pass
        self.destroy()

    def run_test_signal(self):
        """Compatibility trigger to synchronize signal parameters."""
        self._on_test_signal_param_changed()

    def _draw_waveform_canvas(self):
        canvas = self.canvas_wave
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 50 or h < 50:
            return

        margin_x = 45
        margin_y = 20
        plot_w = w - 2 * margin_x
        plot_h = h - 2 * margin_y
        mid_y = margin_y + plot_h / 2.0

        # Axes & grid
        canvas.create_rectangle(margin_x, margin_y, margin_x + plot_w, margin_y + plot_h,
                                fill='#ffffff', outline='#d0d7de')
        canvas.create_line(margin_x, mid_y, margin_x + plot_w, mid_y, fill='#d0d7de', dash=(2, 4))
        canvas.create_text(margin_x - 5, mid_y, text="0.0", fill='#57606a', font=('Segoe UI', 8), anchor='e')
        canvas.create_text(margin_x - 5, margin_y, text="+1.0", fill='#57606a', font=('Segoe UI', 8), anchor='e')
        canvas.create_text(margin_x - 5, margin_y + plot_h, text="-1.0", fill='#57606a', font=('Segoe UI', 8), anchor='e')

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
            canvas.create_line(pts_in, fill='#0969da', width=1.5)
        if len(pts_out) >= 4:
            canvas.create_line(pts_out, fill='#1a7f37', width=2)

    def _draw_fft_canvas(self):
        canvas = self.canvas_fft
        canvas.delete("all")
        w = canvas.winfo_width()
        h = canvas.winfo_height()
        if w < 50 or h < 50:
            return

        margin_x = 45
        margin_y = 20
        plot_w = w - 2 * margin_x
        plot_h = h - 2 * margin_y

        canvas.create_rectangle(margin_x, margin_y, margin_x + plot_w, margin_y + plot_h,
                                fill='#ffffff', outline='#d0d7de')

        # Compute FFT for input and output
        f_in, db_in = compute_fft(self.test_in, self.sample_rate, max_points=1024)
        f_out, db_out = compute_fft(self.test_out, self.sample_rate, max_points=1024)

        db_min = -80.0
        db_max = 10.0

        # Draw dB lines
        for db in [-60, -40, -20, 0]:
            ratio = (db - db_min) / (db_max - db_min)
            y = margin_y + (1.0 - ratio) * plot_h
            canvas.create_line(margin_x, y, margin_x + plot_w, y, fill='#eaeef2', dash=(2, 4))
            canvas.create_text(margin_x - 5, y, text=f"{db}dB", fill='#57606a', font=('Segoe UI', 8), anchor='e')

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
            canvas.create_line(coords_in, fill='#b08800', width=1, dash=(3, 2))
        if len(coords_out) >= 4:
            canvas.create_line(coords_out, fill='#1a7f37', width=2)

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
        dlg.configure(bg='#f6f8fa')
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
        if not self.test_out and not self.current_filter:
            messagebox.showwarning("No Data", "No audio signal data available to export.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".csv",
                                             filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
                                             initialfile="filter_test_data.csv")
        if fpath:
            sig_type = self.var_test_sig.get()
            try:
                f1 = float(self.var_test_f1.get())
            except ValueError:
                f1 = 500.0
            try:
                f2 = float(self.var_test_f2.get())
            except ValueError:
                f2 = 5000.0
            t_vec, in_vec = generate_signal(sig_type, duration_sec=0.2, sample_rate=self.sample_rate,
                                            freq1=f1, freq2=f2, amplitude=0.8)
            if self.current_filter:
                f_clone = self.current_filter.clone()
                f_clone.reset()
                out_vec = f_clone.process_block(in_vec)
            else:
                out_vec = in_vec
            export_data_to_csv(fpath, t_vec, in_vec, out_vec)
            messagebox.showinfo("Export Successful", f"Test data saved to:\n{fpath}")

    def _export_test_json(self):
        if not self.current_filter:
            messagebox.showwarning("No Data", "No filter design available.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                                             initialfile="filter_test_data.json")
        if fpath:
            sig_type = self.var_test_sig.get()
            try:
                f1 = float(self.var_test_f1.get())
            except ValueError:
                f1 = 500.0
            try:
                f2 = float(self.var_test_f2.get())
            except ValueError:
                f2 = 5000.0
            t_vec, in_vec = generate_signal(sig_type, duration_sec=0.2, sample_rate=self.sample_rate,
                                            freq1=f1, freq2=f2, amplitude=0.8)
            f_clone = self.current_filter.clone()
            f_clone.reset()
            out_vec = f_clone.process_block(in_vec)
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
            export_data_to_json(fpath, meta, t_vec, in_vec, out_vec)
            messagebox.showinfo("Export Successful", f"JSON data saved to:\n{fpath}")

    def _export_test_wav(self):
        if not self.current_filter and not self.test_out:
            messagebox.showwarning("No Data", "No audio data available for export.")
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".wav",
                                             filetypes=[("WAV Audio Files", "*.wav"), ("All Files", "*.*")],
                                             initialfile="filtered_signal.wav")
        if fpath:
            sig_type = self.var_test_sig.get()
            try:
                f1 = float(self.var_test_f1.get())
            except ValueError:
                f1 = 500.0
            try:
                f2 = float(self.var_test_f2.get())
            except ValueError:
                f2 = 5000.0
            _, in_vec = generate_signal(sig_type, duration_sec=1.0, sample_rate=self.sample_rate,
                                        freq1=f1, freq2=f2, amplitude=0.8)
            if self.current_filter:
                f_clone = self.current_filter.clone()
                f_clone.reset()
                out_vec = f_clone.process_block(in_vec)
            else:
                out_vec = in_vec
            export_audio_to_wav(fpath, out_vec, int(self.sample_rate))
            messagebox.showinfo("Export Successful", f"WAV Audio saved to:\n{fpath}")

    def _export_coeffs_json(self):
        if not self.current_filter:
            return
        fpath = filedialog.asksaveasfilename(defaultextension=".json",
                                             filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")],
                                             initialfile="filter_coefficients.json")
        if fpath:
            b_poly, a_poly = self.current_filter.get_direct_form_coefficients()
            data = {
                "family": getattr(self.current_filter, 'filter_family', 'IIR'),
                "filter_type": self.filter_type,
                "response_type": self.response_type,
                "order": self.order,
                "sample_rate": self.sample_rate,
                "f_cutoff": self.f_cutoff,
                "f_cutoff2": self.f_cutoff2,
                "direct_form": {
                    "b": b_poly,
                    "a": a_poly
                }
            }
            if getattr(self.current_filter, 'filter_family', 'IIR') == 'IIR':
                sections_data = []
                for i, sec in enumerate(self.current_filter.sections):
                    sections_data.append({
                        "section": i + 1,
                        "b0": sec.b0, "b1": sec.b1, "b2": sec.b2,
                        "a0": sec.a0, "a1": sec.a1, "a2": sec.a2
                    })
                data["sections"] = sections_data
            else:
                data["taps"] = list(self.current_filter.taps)
                data["num_taps"] = self.current_filter.num_taps
            with open(fpath, "w", encoding="utf-8") as f:
                import json
                json.dump(data, f, indent=2)
            messagebox.showinfo("Export Successful", f"Coefficients exported to:\n{fpath}")


def main():
    app = FilterDesignApp()
    app.mainloop()


if __name__ == "__main__":
    main()
