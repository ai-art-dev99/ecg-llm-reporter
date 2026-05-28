"""
ECG Feature Extractor
---------------------
Extracts clinically meaningful features from raw ECG signals:
  - Heart rate & HRV
  - Wave intervals (PR, QRS, QT, QTc)
  - Rhythm classification
  - Morphology analysis
  - Anomaly flags
"""

import warnings
from dataclasses import dataclass, field, asdict
from typing import Optional

import neurokit2 as nk
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ─── Feature Data Classes ─────────────────────────────────────────────────────

@dataclass
class HeartRateFeatures:
    mean_bpm: float
    min_bpm: float
    max_bpm: float
    std_bpm: float
    classification: str          # normal | tachycardia | bradycardia


@dataclass
class HRVFeatures:
    rmssd_ms: Optional[float]    # Root mean square of successive differences
    sdnn_ms: Optional[float]     # SD of NN intervals
    pnn50: Optional[float]       # % NN intervals > 50ms diff
    mean_rr_ms: Optional[float]


@dataclass
class WaveIntervals:
    pr_interval_ms: Optional[float]   # Normal: 120–200ms
    qrs_duration_ms: Optional[float]  # Normal: 80–120ms
    qt_interval_ms: Optional[float]   # Normal: 350–440ms
    qtc_ms: Optional[float]           # Corrected QT (Bazett)
    rr_interval_ms: Optional[float]


@dataclass
class WaveAmplitudes:
    p_amplitude_mv: Optional[float]
    r_amplitude_mv: Optional[float]
    s_amplitude_mv: Optional[float]
    t_amplitude_mv: Optional[float]
    st_elevation_mv: Optional[float]  # ST segment deviation


@dataclass
class AnomalyFlags:
    tachycardia: bool = False
    bradycardia: bool = False
    qt_prolonged: bool = False
    qt_short: bool = False
    wide_qrs: bool = False
    pr_prolonged: bool = False         # First-degree AV block
    st_elevation: bool = False
    st_depression: bool = False
    irregular_rhythm: bool = False
    poor_signal_quality: bool = False

    def active_flags(self) -> list[str]:
        return [k for k, v in asdict(self).items() if v]


@dataclass
class ECGFeatures:
    record_name: str
    sampling_rate: int
    duration_seconds: float
    signal_quality: float           # 0–1 (neurokit quality index)

    heart_rate: HeartRateFeatures
    hrv: HRVFeatures
    intervals: WaveIntervals
    amplitudes: WaveAmplitudes
    anomalies: AnomalyFlags

    rhythm_classification: str      # e.g. "Normal Sinus Rhythm"
    summary: str                    # One-line human-readable summary
    r_peak_count: int
    processing_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """Format features as structured text for LLM prompt injection."""
        flags = self.anomalies.active_flags()
        flags_str = ", ".join(flags) if flags else "None"

        return f"""
ECG MEASUREMENTS (Record: {self.record_name})
============================================
Duration        : {self.duration_seconds:.1f} seconds
Signal Quality  : {self.signal_quality:.2f} / 1.00
Lead            : Standard Lead II (MLII)

HEART RATE
  Mean          : {self.heart_rate.mean_bpm:.1f} bpm
  Range         : {self.heart_rate.min_bpm:.1f} – {self.heart_rate.max_bpm:.1f} bpm
  Variability   : ±{self.heart_rate.std_bpm:.1f} bpm
  Classification: {self.heart_rate.classification}

INTERVALS
  PR Interval   : {self._fmt(self.intervals.pr_interval_ms)} ms  [Normal: 120–200]
  QRS Duration  : {self._fmt(self.intervals.qrs_duration_ms)} ms  [Normal: 80–120]
  QT Interval   : {self._fmt(self.intervals.qt_interval_ms)} ms  [Normal: 350–440]
  QTc (Bazett)  : {self._fmt(self.intervals.qtc_ms)} ms  [Normal: <440 M / <460 F]
  Mean RR       : {self._fmt(self.intervals.rr_interval_ms)} ms

WAVE AMPLITUDES
  P wave        : {self._fmt(self.amplitudes.p_amplitude_mv)} mV
  R wave        : {self._fmt(self.amplitudes.r_amplitude_mv)} mV
  S wave        : {self._fmt(self.amplitudes.s_amplitude_mv)} mV
  T wave        : {self._fmt(self.amplitudes.t_amplitude_mv)} mV
  ST deviation  : {self._fmt(self.amplitudes.st_elevation_mv)} mV

HRV (Heart Rate Variability)
  RMSSD         : {self._fmt(self.hrv.rmssd_ms)} ms
  SDNN          : {self._fmt(self.hrv.sdnn_ms)} ms
  pNN50         : {self._fmt(self.hrv.pnn50)} %

RHYTHM
  Classification: {self.rhythm_classification}
  R-peaks found : {self.r_peak_count}

ANOMALY FLAGS: {flags_str}

SUMMARY: {self.summary}
""".strip()

    @staticmethod
    def _fmt(val: Optional[float]) -> str:
        return f"{val:.2f}" if val is not None else "N/A"


# ─── Extractor ────────────────────────────────────────────────────────────────

class ECGFeatureExtractor:
    """
    Full ECG feature extraction pipeline using NeuroKit2.

    Usage:
        extractor = ECGFeatureExtractor()
        features = extractor.extract(signal, sampling_rate=360, record_name="100")
    """

    # Clinical thresholds
    HR_TACHY_THRESHOLD = 100    # bpm
    HR_BRADY_THRESHOLD = 60     # bpm
    QTC_LONG_THRESHOLD = 440    # ms (male); use 460 for female
    QTC_SHORT_THRESHOLD = 340   # ms
    QRS_WIDE_THRESHOLD = 120    # ms
    PR_LONG_THRESHOLD = 200     # ms
    ST_ELEVATION_THRESHOLD = 0.1    # mV (1mm)
    ST_DEPRESSION_THRESHOLD = -0.05 # mV

    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        record_name: str = "unknown",
    ) -> ECGFeatures:
        """Main entry point — returns a fully populated ECGFeatures object."""
        notes = []

        # ── 1. Clean signal ──────────────────────────────────────────────────
        try:
            ecg_clean = nk.ecg_clean(signal, sampling_rate=sampling_rate)
        except Exception as e:
            notes.append(f"Cleaning failed: {e}. Using raw signal.")
            ecg_clean = signal.copy()

        # ── 2. Process (peaks + intervals) ──────────────────────────────────
        try:
            ecg_signals, info = nk.ecg_process(ecg_clean, sampling_rate=sampling_rate)
        except Exception as e:
            notes.append(f"Processing failed: {e}")
            return self._empty_features(record_name, sampling_rate, len(signal), notes)

        # ── 3. Signal quality ────────────────────────────────────────────────
        quality = float(ecg_signals["ECG_Quality"].mean())

        if quality < 0.5:
            notes.append(f"Low signal quality ({quality:.2f}). Results may be unreliable.")

        # ── 4. Heart rate ────────────────────────────────────────────────────
        hr_series = ecg_signals["ECG_Rate"].dropna()
        if len(hr_series) == 0:
            notes.append("Could not compute heart rate.")
            hr = HeartRateFeatures(0, 0, 0, 0, "unknown")
        else:
            mean_hr = float(hr_series.mean())
            hr = HeartRateFeatures(
                mean_bpm=round(mean_hr, 1),
                min_bpm=round(float(hr_series.min()), 1),
                max_bpm=round(float(hr_series.max()), 1),
                std_bpm=round(float(hr_series.std()), 1),
                classification=self._classify_hr(mean_hr),
            )

        # ── 5. Intervals ─────────────────────────────────────────────────────
        intervals = self._extract_intervals(ecg_signals, info, sampling_rate, notes)

        # ── 6. Amplitudes ────────────────────────────────────────────────────
        amplitudes = self._extract_amplitudes(ecg_signals, info, ecg_clean)

        # ── 7. HRV ───────────────────────────────────────────────────────────
        hrv = self._extract_hrv(info, sampling_rate, notes)

        # ── 8. Anomalies ─────────────────────────────────────────────────────
        anomalies = self._detect_anomalies(hr, intervals, amplitudes, quality)

        # ── 9. Rhythm classification ─────────────────────────────────────────
        r_peaks_raw = info.get("ECG_R_Peaks", [])
        r_peaks = np.array(r_peaks_raw) if not isinstance(r_peaks_raw, np.ndarray) else r_peaks_raw
        rhythm = self._classify_rhythm(hr, r_peaks, sampling_rate)

        # ── 10. Summary ──────────────────────────────────────────────────────
        summary = self._build_summary(hr, intervals, anomalies, rhythm)

        return ECGFeatures(
            record_name=record_name,
            sampling_rate=sampling_rate,
            duration_seconds=round(len(signal) / sampling_rate, 2),
            signal_quality=round(quality, 3),
            heart_rate=hr,
            hrv=hrv,
            intervals=intervals,
            amplitudes=amplitudes,
            anomalies=anomalies,
            rhythm_classification=rhythm,
            summary=summary,
            r_peak_count=len(r_peaks),
            processing_notes=notes,
        )

    # ── Interval Extraction ───────────────────────────────────────────────────

    def _extract_intervals(self, signals, info, fs, notes) -> WaveIntervals:
        ms = 1000 / fs  # samples → ms conversion

        def to_arr(key):
            val = info.get(key, [])
            return np.array(val) if not isinstance(val, np.ndarray) else val

        try:
            # RR interval from R-peaks
            r_peaks = to_arr("ECG_R_Peaks")
            rr_ms = float(np.diff(r_peaks).mean() * ms) if len(r_peaks) > 1 else None

            # PR interval: P onset → R peak
            p_onsets = to_arr("ECG_P_Onsets")
            r_onsets = to_arr("ECG_R_Onsets")
            pr_ms = self._mean_interval(p_onsets, r_onsets, ms)

            # QRS duration: R onset → R offset
            r_offsets = to_arr("ECG_R_Offsets")
            qrs_ms = self._mean_interval(r_onsets, r_offsets, ms)

            # QT interval: Q → T offset
            q_peaks = to_arr("ECG_Q_Peaks")
            t_offsets = to_arr("ECG_T_Offsets")
            qt_ms = self._mean_interval(q_peaks, t_offsets, ms)

            # QTc (Bazett): QT / sqrt(RR in seconds)
            qtc_ms = None
            if qt_ms and rr_ms:
                rr_sec = rr_ms / 1000
                qtc_ms = round(qt_ms / np.sqrt(rr_sec), 1)

        except Exception as e:
            notes.append(f"Interval extraction error: {e}")
            rr_ms = pr_ms = qrs_ms = qt_ms = qtc_ms = None

        return WaveIntervals(
            pr_interval_ms=pr_ms,
            qrs_duration_ms=qrs_ms,
            qt_interval_ms=qt_ms,
            qtc_ms=qtc_ms,
            rr_interval_ms=rr_ms,
        )

    def _extract_amplitudes(self, signals, info, clean_signal) -> WaveAmplitudes:
        try:
            def peak_amplitude(peak_key):
                peaks = info.get(peak_key, np.array([]))
                peaks = peaks[peaks < len(clean_signal)]
                if len(peaks) == 0:
                    return None
                return round(float(np.mean(clean_signal[peaks])), 4)

            r_amp = peak_amplitude("ECG_R_Peaks")
            p_amp = peak_amplitude("ECG_P_Peaks")
            q_amp = peak_amplitude("ECG_Q_Peaks")
            s_amp = peak_amplitude("ECG_S_Peaks")
            t_amp = peak_amplitude("ECG_T_Peaks")

            # ST elevation: measure signal 60–80ms after R peak
            r_peaks = info.get("ECG_R_Peaks", np.array([]))
            st_vals = []
            fs = 360  # fallback
            if len(r_peaks) > 0:
                j_point_offset = int(0.06 * 360)
                for rp in r_peaks:
                    idx = rp + j_point_offset
                    if idx < len(clean_signal):
                        st_vals.append(clean_signal[idx])
            st_elev = round(float(np.mean(st_vals)), 4) if st_vals else None

        except Exception:
            return WaveAmplitudes(None, None, None, None, None)

        return WaveAmplitudes(
            p_amplitude_mv=p_amp,
            r_amplitude_mv=r_amp,
            s_amplitude_mv=s_amp,
            t_amplitude_mv=t_amp,
            st_elevation_mv=st_elev,
        )

    def _extract_hrv(self, info, fs, notes) -> HRVFeatures:
        try:
            r_peaks = info.get("ECG_R_Peaks", np.array([]))
            if len(r_peaks) < 3:
                notes.append("Insufficient R-peaks for HRV analysis (need ≥3).")
                return HRVFeatures(None, None, None, None)

            rr_intervals = np.diff(r_peaks) / fs * 1000  # ms
            mean_rr = float(rr_intervals.mean())

            successive_diffs = np.diff(rr_intervals)
            rmssd = float(np.sqrt(np.mean(successive_diffs ** 2)))
            sdnn = float(rr_intervals.std())
            pnn50 = float(np.sum(np.abs(successive_diffs) > 50) / len(successive_diffs) * 100)

            return HRVFeatures(
                rmssd_ms=round(rmssd, 2),
                sdnn_ms=round(sdnn, 2),
                pnn50=round(pnn50, 2),
                mean_rr_ms=round(mean_rr, 2),
            )
        except Exception as e:
            notes.append(f"HRV extraction failed: {e}")
            return HRVFeatures(None, None, None, None)

    # ── Classification & Detection ────────────────────────────────────────────

    def _classify_hr(self, mean_bpm: float) -> str:
        if mean_bpm >= self.HR_TACHY_THRESHOLD:
            return "tachycardia"
        elif mean_bpm <= self.HR_BRADY_THRESHOLD:
            return "bradycardia"
        return "normal"

    def _classify_rhythm(self, hr: HeartRateFeatures, r_peaks: np.ndarray, fs: int) -> str:
        if len(r_peaks) < 2:
            return "Unclassified (insufficient data)"

        rr_intervals = np.diff(r_peaks) / fs * 1000
        rr_cv = rr_intervals.std() / rr_intervals.mean() if rr_intervals.mean() > 0 else 0

        irregular = rr_cv > 0.15

        if irregular:
            return "Irregular rhythm — possible atrial fibrillation or ectopic beats"
        elif hr.classification == "tachycardia":
            return "Sinus Tachycardia"
        elif hr.classification == "bradycardia":
            return "Sinus Bradycardia"
        else:
            return "Normal Sinus Rhythm"

    def _detect_anomalies(
        self,
        hr: HeartRateFeatures,
        intervals: WaveIntervals,
        amplitudes: WaveAmplitudes,
        quality: float,
    ) -> AnomalyFlags:
        flags = AnomalyFlags()

        flags.tachycardia = hr.classification == "tachycardia"
        flags.bradycardia = hr.classification == "bradycardia"

        if intervals.qtc_ms:
            flags.qt_prolonged = intervals.qtc_ms > self.QTC_LONG_THRESHOLD
            flags.qt_short = intervals.qtc_ms < self.QTC_SHORT_THRESHOLD

        if intervals.qrs_duration_ms:
            flags.wide_qrs = intervals.qrs_duration_ms > self.QRS_WIDE_THRESHOLD

        if intervals.pr_interval_ms:
            flags.pr_prolonged = intervals.pr_interval_ms > self.PR_LONG_THRESHOLD

        if amplitudes.st_elevation_mv is not None:
            flags.st_elevation = amplitudes.st_elevation_mv > self.ST_ELEVATION_THRESHOLD
            flags.st_depression = amplitudes.st_elevation_mv < self.ST_DEPRESSION_THRESHOLD

        flags.poor_signal_quality = quality < 0.5

        return flags

    def _build_summary(
        self,
        hr: HeartRateFeatures,
        intervals: WaveIntervals,
        anomalies: AnomalyFlags,
        rhythm: str,
    ) -> str:
        flags = anomalies.active_flags()
        hr_str = f"{hr.mean_bpm:.0f} bpm"

        if not flags:
            return f"ECG within normal limits. {rhythm} at {hr_str}."

        flag_labels = {
            "tachycardia": "sinus tachycardia",
            "bradycardia": "sinus bradycardia",
            "qt_prolonged": "prolonged QTc",
            "qt_short": "short QTc",
            "wide_qrs": "wide QRS complex",
            "pr_prolonged": "prolonged PR interval",
            "st_elevation": "ST elevation",
            "st_depression": "ST depression",
            "irregular_rhythm": "irregular rhythm",
            "poor_signal_quality": "poor signal quality",
        }

        findings = [flag_labels.get(f, f) for f in flags]
        return f"{rhythm} at {hr_str}. Findings: {', '.join(findings)}."

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _mean_interval(
        start_peaks: np.ndarray, end_peaks: np.ndarray, ms_per_sample: float
    ) -> Optional[float]:
        if len(start_peaks) == 0 or len(end_peaks) == 0:
            return None
        n = min(len(start_peaks), len(end_peaks))
        intervals = (end_peaks[:n] - start_peaks[:n]) * ms_per_sample
        intervals = intervals[intervals > 0]
        return round(float(intervals.mean()), 1) if len(intervals) > 0 else None

    def _empty_features(self, record_name, fs, n_samples, notes) -> ECGFeatures:
        hr = HeartRateFeatures(0, 0, 0, 0, "unknown")
        return ECGFeatures(
            record_name=record_name,
            sampling_rate=fs,
            duration_seconds=n_samples / fs,
            signal_quality=0.0,
            heart_rate=hr,
            hrv=HRVFeatures(None, None, None, None),
            intervals=WaveIntervals(None, None, None, None, None),
            amplitudes=WaveAmplitudes(None, None, None, None, None),
            anomalies=AnomalyFlags(),
            rhythm_classification="Unknown",
            summary="Feature extraction failed. See processing notes.",
            r_peak_count=0,
            processing_notes=notes,
        )
