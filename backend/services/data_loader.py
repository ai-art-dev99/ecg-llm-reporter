"""
MIT-BIH ECG Data Loader
-----------------------
Supports:
  - Real MIT-BIH records via wfdb (when PhysioNet is accessible)
  - Synthetic ECG generation for development/testing
  - CSV upload from user
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import neurokit2 as nk
import numpy as np
import pandas as pd

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False


# ─── Data Classes ────────────────────────────────────────────────────────────

@dataclass
class ECGRecord:
    record_name: str
    signal: np.ndarray          # shape: (n_samples,) — Lead II
    signal_all: np.ndarray      # shape: (n_samples, n_leads)
    sampling_rate: int
    leads: list[str]
    duration_seconds: float
    source: str                 # 'mitdb' | 'synthetic' | 'upload'
    metadata: dict


# ─── Loader ──────────────────────────────────────────────────────────────────

class ECGDataLoader:
    """Load ECG records from MIT-BIH, CSV files, or generate synthetic data."""

    MITDB_SAMPLING_RATE = 360
    DEFAULT_LEAD = "MLII"

    # Rhythm labels used by MIT-BIH annotators
    RHYTHM_MAP = {
        "N": "Normal sinus rhythm",
        "L": "Left bundle branch block",
        "R": "Right bundle branch block",
        "A": "Atrial premature beat",
        "V": "Premature ventricular contraction",
        "F": "Fusion beat",
        "/": "Paced beat",
        "~": "Signal quality change",
    }

    def __init__(self, data_dir: str = "data/mitdb"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def load_record(self, record_name: str) -> ECGRecord:
        """Load a record by name. Falls back to synthetic if not found."""
        csv_path = self.data_dir / f"{record_name}.csv"
        meta_path = self.data_dir / f"{record_name}_meta.json"

        if csv_path.exists() and meta_path.exists():
            return self._load_from_csv(record_name, csv_path, meta_path)

        if WFDB_AVAILABLE:
            try:
                return self._load_from_wfdb(record_name)
            except Exception as e:
                print(f"[DataLoader] wfdb failed ({e}), using synthetic fallback")

        return self.generate_synthetic(record_name)

    def load_from_array(
        self,
        signal: np.ndarray,
        sampling_rate: int,
        record_name: str = "user_upload",
    ) -> ECGRecord:
        """Wrap a raw numpy array (single lead) into an ECGRecord."""
        return ECGRecord(
            record_name=record_name,
            signal=signal,
            signal_all=signal.reshape(-1, 1),
            sampling_rate=sampling_rate,
            leads=["MLII"],
            duration_seconds=len(signal) / sampling_rate,
            source="upload",
            metadata={"n_samples": len(signal)},
        )

    def load_from_csv(self, filepath: str, sampling_rate: int = 360) -> ECGRecord:
        """Load ECG from a user-uploaded CSV. First numeric column is used."""
        df = pd.read_csv(filepath)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in CSV.")

        signal = df[numeric_cols[0]].dropna().values
        all_signals = df[numeric_cols].dropna().values

        return ECGRecord(
            record_name=Path(filepath).stem,
            signal=signal,
            signal_all=all_signals,
            sampling_rate=sampling_rate,
            leads=numeric_cols,
            duration_seconds=len(signal) / sampling_rate,
            source="upload",
            metadata={"original_columns": numeric_cols},
        )

    def generate_synthetic(
        self,
        record_name: str = "synthetic",
        heart_rate: int = 72,
        duration: int = 10,
        rhythm: str = "normal",
        noise: float = 0.01,
    ) -> ECGRecord:
        """
        Generate a realistic synthetic ECG record.

        rhythm options: 'normal' | 'tachycardia' | 'bradycardia' | 'afib'
        """
        hr_map = {
            "normal": heart_rate,
            "tachycardia": max(heart_rate, 110),
            "bradycardia": min(heart_rate, 48),
            "afib": heart_rate,
        }
        actual_hr = hr_map.get(rhythm, heart_rate)

        if rhythm == "afib":
            # Simulate AFib: irregular R-R intervals
            signal = self._simulate_afib(duration, self.MITDB_SAMPLING_RATE)
        else:
            signal = nk.ecg_simulate(
                duration=duration,
                sampling_rate=self.MITDB_SAMPLING_RATE,
                heart_rate=actual_hr,
                noise=noise,
            )

        signal_v5 = signal * 0.75 + np.random.normal(0, noise * 0.5, len(signal))

        return ECGRecord(
            record_name=record_name,
            signal=signal,
            signal_all=np.column_stack([signal, signal_v5]),
            sampling_rate=self.MITDB_SAMPLING_RATE,
            leads=["MLII", "V5"],
            duration_seconds=duration,
            source="synthetic",
            metadata={
                "rhythm": rhythm,
                "target_heart_rate": actual_hr,
                "noise_level": noise,
            },
        )

    def list_available_records(self) -> list[str]:
        """Return all locally available record names."""
        records = []
        for f in self.data_dir.glob("*.csv"):
            if "_meta" not in f.stem:
                records.append(f.stem)
        return sorted(records)

    # ── Private Helpers ───────────────────────────────────────────────────────

    def _load_from_csv(self, record_name: str, csv_path: Path, meta_path: Path) -> ECGRecord:
        df = pd.read_csv(csv_path)
        with open(meta_path) as f:
            meta = json.load(f)

        leads = meta.get("leads", df.columns.tolist())
        signal = df[leads[0]].values

        return ECGRecord(
            record_name=record_name,
            signal=signal,
            signal_all=df[leads].values,
            sampling_rate=meta.get("sampling_rate", self.MITDB_SAMPLING_RATE),
            leads=leads,
            duration_seconds=meta.get("duration_seconds", len(signal) / self.MITDB_SAMPLING_RATE),
            source="mitdb",
            metadata=meta,
        )

    def _load_from_wfdb(self, record_name: str) -> ECGRecord:
        record_path = str(self.data_dir / record_name)
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal[:, 0]  # Lead I / MLII

        return ECGRecord(
            record_name=record_name,
            signal=signal,
            signal_all=record.p_signal,
            sampling_rate=record.fs,
            leads=record.sig_name,
            duration_seconds=len(signal) / record.fs,
            source="mitdb",
            metadata={
                "comments": record.comments,
                "units": record.units,
                "n_leads": record.n_sig,
            },
        )

    @staticmethod
    def _simulate_afib(duration: int, fs: int) -> np.ndarray:
        """Approximate AFib by varying R-R intervals irregularly."""
        n_samples = duration * fs
        signal = np.zeros(n_samples)

        t = 0
        while t < n_samples:
            # Irregular RR intervals: 400-1000ms
            rr = int(np.random.uniform(0.4, 1.0) * fs)
            if t + 50 < n_samples:
                # Add a simplified QRS complex
                qrs = nk.ecg_simulate(
                    duration=0.12, sampling_rate=fs, heart_rate=60, noise=0.005
                )
                end = min(t + len(qrs), n_samples)
                signal[t:end] += qrs[: end - t]
            t += rr

        signal += np.random.normal(0, 0.01, n_samples)
        return signal
