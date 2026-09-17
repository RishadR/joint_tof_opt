"""
DuckDB-backed cache for generate_tof's output .npz bytes, keyed by a hash of ppath file identity,
ToFConfig, and pulse flags.

Note: inner_moment_orders is deliberately excluded from the key even though it changes generate_tof's
output (adds inner_moment_* arrays) - no call site varies it today. If a caller ever requests different
orders for the same ppath/config, it'll silently collide with whatever got cached first.
"""

import hashlib
import json
import threading
from pathlib import Path

import duckdb
import numpy as np

from joint_tof_opt.config_loader import ToFConfig

CACHE_DB_PATH = Path("data/tof_cache.duckdb")

# ToFConfig fields generate_tof() never reads - they only drive ppath_gen.py's simulation/sweep setup.
# Excluded so editing e.g. dermis_thicknesses doesn't invalidate every cached entry's key.
_CACHE_IRRELEVANT_CONFIG_FIELDS = {
    "total_photon_count",
    "epidermis_thickness",
    "donut_half_thickness",
    "sdd_distances",
    "dermis_thicknesses",
}

_key_locks: dict[str, threading.Lock] = {}
_key_locks_mutex = threading.Lock()
_db_mutex = threading.Lock()


def lock_for(key: str) -> threading.Lock:
    with _key_locks_mutex:
        return _key_locks.setdefault(key, threading.Lock())


def _json_default(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def cache_key(
    ppath_dataset_filename: Path,
    gen_config: ToFConfig,
    pulse_maternal: bool,
    pulse_fetal: bool,
    _inner_moment_orders: list[float],
) -> str:
    """Hash of ppath file identity, ToFConfig, and pulse flags. See module docstring re: inner_moment_orders."""
    resolved = ppath_dataset_filename.resolve()
    stat = resolved.stat()
    payload = {
        "ppath_path": str(resolved),
        "ppath_size": stat.st_size,
        "ppath_mtime": stat.st_mtime,
        "gen_config": gen_config.model_dump(exclude=_CACHE_IRRELEVANT_CONFIG_FIELDS),
        "pulse_maternal": pulse_maternal,
        "pulse_fetal": pulse_fetal,
    }
    canonical = json.dumps(payload, sort_keys=True, default=_json_default)
    return hashlib.sha256(canonical.encode()).hexdigest()


def _connect() -> duckdb.DuckDBPyConnection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(CACHE_DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS tof_cache (
            cache_key VARCHAR PRIMARY KEY,
            npz_bytes BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT current_timestamp
        )
        """
    )
    return conn


def get_cached_npz_bytes(key: str) -> bytes | None:
    with _db_mutex:
        conn = _connect()
        try:
            row = conn.execute(
                "SELECT npz_bytes FROM tof_cache WHERE cache_key = ?", [key]
            ).fetchone()
        finally:
            conn.close()
    return bytes(row[0]) if row is not None else None


def store_npz_bytes(key: str, npz_bytes: bytes) -> None:
    with _db_mutex:
        conn = _connect()
        try:
            conn.execute(
                "INSERT OR REPLACE INTO tof_cache (cache_key, npz_bytes) VALUES (?, ?)",
                [key, npz_bytes],
            )
        finally:
            conn.close()
