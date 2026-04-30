# -*- coding: utf-8 -*-
"""
This module contains all configuration settings for the project.
"""
from __future__ import annotations
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
