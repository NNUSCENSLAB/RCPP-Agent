import os
import unittest
from unittest.mock import patch

from rcpp_core.model_registry import get_deployment


class ModelRegistryTests(unittest.TestCase):
    def test_current_models_remain_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RCPP_SCENE_MODEL_KEY", None)
            os.environ.pop("RCPP_SEMANTIC_MODEL_KEY", None)
            self.assertEqual("scene_qwen25_current", get_deployment("scene").key)
            self.assertEqual("semantic_qwen25_current", get_deployment("semantic").key)

    def test_qwen3_can_be_selected_by_environment(self):
        with patch.dict(os.environ, {"RCPP_SCENE_MODEL_KEY": "scene_qwen3_vl_8b_qlora"}):
            self.assertEqual("qwen3_vl", get_deployment("scene").family)

    def test_cross_kind_route_is_rejected(self):
        with self.assertRaises(ValueError):
            get_deployment("semantic", "scene_qwen3_vl_8b_qlora")


if __name__ == "__main__":
    unittest.main()
