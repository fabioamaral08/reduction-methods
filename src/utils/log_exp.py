from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class RunRecord:
    # Identity
    exp_id: str
    geom: str
    constitutive_model: str
    kernel: str  
    task: str    # "recon", "rom", "sindy", "rom+sindy", etc.

    # Parameters
    Re: Optional[float] = None
    Wi: Optional[float] = None
    beta: Optional[float] = None
    theta: Optional[float] = None

    # Reduction / SINDy
    r: Optional[int] = None # number of modes
    poly_deg: Optional[int] = None # sindy lib

    # SR3 
    relax_coeff_nu: Optional[float] = None
    reg_weight_lam: Optional[float] = None

    # Metrics
    frob_mean: Optional[float] = None
    frob_max: Optional[float] = None
    energy_mean: Optional[float] = None
    energy_max: Optional[float] = None
    stable: Optional[bool] = None

    # SINDy summary
    n_terms: Optional[int] = None
    coef_norm: Optional[float] = None

    # Tracking
    run_dir: Optional[str] = None
    metrics_path: Optional[str] = None
    fig_recon_path: Optional[str] = None
    fig_rollout_path: Optional[str] = None
    sindy_model_path: Optional[str] = None

    # Extras
    timestamp_utc: Optional[str] = None
    notes: Optional[str] = None


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # atomic on most OS/filesystems


def upsert_manifest(record: RunRecord, manifest_path: str | Path) -> pd.DataFrame:
    """
    Upsert by exp_id in manifest.csv.
    - If exp_id already exists: updates record columns (does not overwrite with None).
    - If it does not exist: adds a new row.
    Returns updated DataFrame.
    """
    manifest_path = Path(manifest_path)

    rec = asdict(record)
    if rec.get("timestamp_utc") is None:
        rec["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
    else:
        df = pd.DataFrame()

    # If DF is empty, create with columns from record
    if df.empty:
        df = pd.DataFrame([rec])
        _atomic_write_csv(df, manifest_path)
        return df

    if "exp_id" not in df.columns:
        raise ValueError("manifest.csv exists, but does not have 'exp_id' column.")

    # Ensure all columns
    for col in rec.keys():
        if col not in df.columns:
            df[col] = pd.NA

    mask = df["exp_id"].astype(str) == str(rec["exp_id"])
    if mask.any():
        idx = df.index[mask][0]
        # Update only non-None fields
        for k, v in rec.items():
            if v is not None:
                df.at[idx, k] = v
    else:
        # New row
        df = pd.concat([df, pd.DataFrame([rec])], ignore_index=True)

    _atomic_write_csv(df, manifest_path)
    return df
