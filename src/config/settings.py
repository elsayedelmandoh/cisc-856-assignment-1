"""
Settings for GRIDWORLD ENVIRONMENT PARAMETERS

Loads configuration from environment variables (.env file).
Use this module to access GridWorld parameters consistently across the project.
"""

import os
import ast
from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

GOAL = int(os.getenv("GOAL"))
DANGER = ast.literal_eval(os.getenv("DANGER"))
BLOCKED = ast.literal_eval(os.getenv("BLOCKED"))

GRID_HEIGHT = int(os.getenv("GRID_HEIGHT"))
GRID_WIDTH = int(os.getenv("GRID_WIDTH"))

DISCOUNT_FACTORS = ast.literal_eval(os.getenv("DISCOUNT_FACTORS"))
NOISE_LEVELS = ast.literal_eval(os.getenv("NOISE_LEVELS"))
CONVERGENCE_TOLERANCE = float(os.getenv("CONVERGENCE_TOLERANCE"))

RESULTS_DIR = os.getenv("RESULTS_DIR")
RESULTS_DPI = int(os.getenv("RESULTS_DPI"))


