"""
Unit tests for the parameter_mapping module.
"""

import unittest
from pathlib import Path

from pydantic import ValidationError

from joint_tof_opt.parameter_mapping import ParameterMappingFile, load_parameter_mapping

DATA_PATH = Path(__file__).parent.parent / "data" / "parameter_mapping.json"


class TestLoadParameterMapping(unittest.TestCase):
    def test_loads_real_file(self):
        mapping = load_parameter_mapping(DATA_PATH)
        self.assertEqual(mapping["experiment_0000.npz"], {"derm_thickness": 4})
        self.assertEqual(mapping["experiment_0007.npz"], {"derm_thickness": 18})
        self.assertEqual(mapping["experiment_0012.npz"], {"derm_thickness": 28})
        self.assertEqual(len(mapping), 13)

    def test_rejects_malformed_data(self):
        with self.assertRaises(ValidationError):
            ParameterMappingFile.model_validate({"experiments": "not a list"})


if __name__ == "__main__":
    unittest.main()
