import os
import unittest
from unittest.mock import patch

from configs.config import get_provider_api_key


class ProviderCredentialTests(unittest.TestCase):
    def test_provider_keys_are_isolated(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds", "OPENAI_API_KEY": "oa"}, clear=False):
            self.assertEqual("ds", get_provider_api_key("deepseek"))
            self.assertEqual("oa", get_provider_api_key("openai"))

    def test_openai_never_falls_back_to_deepseek_key(self):
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "ds"}, clear=False):
            os.environ.pop("OPENAI_API_KEY", None)
            self.assertIsNone(get_provider_api_key("openai"))


if __name__ == "__main__":
    unittest.main()
