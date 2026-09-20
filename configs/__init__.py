# -*- coding: utf-8 -*-
"""
This module contains all configuration settings for the project.
"""
from __future__ import annotations
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # Pure rule/evaluation scripts do not require dotenv.
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
