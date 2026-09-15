"""
Unit tests for the DuckDB-backed tof_cache module.
"""

import tempfile
import unittest
from pathlib import Path

from joint_tof_opt import tof_cache
from joint_tof_opt.config_loader import ToFConfig

# Minimal but complete ToFConfig, built inline rather than via test_config.yaml (that fixture predates
# several now-required ToFConfig fields and is used elsewhere with a raw dict, not load_tof_config).
TEST_GEN_CONFIG_KWARGS = {
    "total_photon_count": 1000,
    "wavelength": 735.0,
    "epidermis_thickness": 2,
    "donut_half_thickness": 1.0,
    "datapoint_count": 60,
    "maternal_f": 1.0,
    "fetal_f": 2.05,
    "end_sec": 4.0,
    "sampling_rate": 15,
    "maternal_hb_base": 15.0,
    "fetal_hb_base": 15.0,
    "maternal_saturation": 1.0,
    "fetal_saturation": 0.60,
    "light_speeds": [2.14e8, 2.14e8, 2.14e8, 2.14e8],
    "epi_thickness_mm": 2,
    "derm_thickness_mm": 4,
    "selected_sdd_index": 1,
    "bin_count": 20,
    "time_limit_or_threshold": "weightthreshold",
    "time_limit": [0.0, 5.0],
    "weight_threshold_fraction": 0.98,
    "sdd_distances": [10, 20, 30],
    "dermis_thicknesses": [2, 4, 6],
}


class TestCacheKey(unittest.TestCase):
    """_cache_key should be deterministic and discriminate on every input that affects the output."""

    def setUp(self):
        self.gen_config = ToFConfig.model_validate(TEST_GEN_CONFIG_KWARGS)
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.ppath_file = Path(self.tmp_dir.name) / "ppath.npz"
        self.ppath_file.write_bytes(b"fake ppath bytes")

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_same_inputs_same_key(self):
        key1 = tof_cache.cache_key(self.ppath_file, self.gen_config, True, True, [])
        key2 = tof_cache.cache_key(self.ppath_file, self.gen_config, True, True, [])
        self.assertEqual(key1, key2)

    def test_different_config_different_key(self):
        other_config = self.gen_config.model_copy(update={"wavelength": self.gen_config.wavelength + 1})
        key1 = tof_cache.cache_key(self.ppath_file, self.gen_config, True, True, [])
        key2 = tof_cache.cache_key(self.ppath_file, other_config, True, True, [])
        self.assertNotEqual(key1, key2)

    def test_different_pulse_flags_different_key(self):
        key1 = tof_cache.cache_key(self.ppath_file, self.gen_config, True, True, [])
        key2 = tof_cache.cache_key(self.ppath_file, self.gen_config, False, True, [])
        self.assertNotEqual(key1, key2)

    def test_different_ppath_file_different_key(self):
        other_ppath = Path(self.tmp_dir.name) / "other.npz"
        other_ppath.write_bytes(b"different fake bytes here")
        key1 = tof_cache.cache_key(self.ppath_file, self.gen_config, True, True, [])
        key2 = tof_cache.cache_key(other_ppath, self.gen_config, True, True, [])
        self.assertNotEqual(key1, key2)

    def test_moment_order_list_is_order_independent(self):
        key1 = tof_cache.cache_key(self.ppath_file, self.gen_config, True, True, [1.0, 2.0])
        key2 = tof_cache.cache_key(self.ppath_file, self.gen_config, True, True, [2.0, 1.0])
        self.assertEqual(key1, key2)


class TestStoreAndFetch(unittest.TestCase):
    """store_npz_bytes/get_cached_npz_bytes should round-trip through a real DuckDB file."""

    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.original_db_path = tof_cache.CACHE_DB_PATH
        tof_cache.CACHE_DB_PATH = Path(self.tmp_dir.name) / "test_cache.duckdb"

    def tearDown(self):
        tof_cache.CACHE_DB_PATH = self.original_db_path
        self.tmp_dir.cleanup()

    def test_miss_then_store_then_hit(self):
        key = "some-test-key"
        self.assertIsNone(tof_cache.get_cached_npz_bytes(key))
        payload = b"pretend npz file contents"
        tof_cache.store_npz_bytes(key, payload)
        self.assertEqual(tof_cache.get_cached_npz_bytes(key), payload)

    def test_store_overwrites_existing_key(self):
        key = "overwrite-key"
        tof_cache.store_npz_bytes(key, b"first")
        tof_cache.store_npz_bytes(key, b"second")
        self.assertEqual(tof_cache.get_cached_npz_bytes(key), b"second")


if __name__ == "__main__":
    unittest.main()
