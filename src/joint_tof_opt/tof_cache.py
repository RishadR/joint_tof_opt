"""
DuckDB-backed cache for generate_tof's output .npz bytes, keyed by a hash of everything that
affects the result (ppath file identity, ToFConfig, pulse flags, inner_moment_orders).
"""

import hashlib
import json
import threading
from pathlib import Path

import duckdb
import numpy as np

from joint_tof_opt.config_loader import ToFConfig

CACHE_DB_PATH = Path("data/tof_cache.duckdb")

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
    inner_moment_orders: list[float],
) -> str:
    """Hash of every input that affects generate_tof's output."""
    resolved = ppath_dataset_filename.resolve()
    stat = resolved.stat()
    payload = {
        "ppath_path": str(resolved),
        "ppath_size": stat.st_size,
        "ppath_mtime": stat.st_mtime,
        "gen_config": gen_config.model_dump(),
        "pulse_maternal": pulse_maternal,
        "pulse_fetal": pulse_fetal,
        "inner_moment_orders": sorted(inner_moment_orders),
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
