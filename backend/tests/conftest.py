"""Test configuration and shared fixtures."""
import os
import sys
from pathlib import Path

import pytest

# Ensure the project root is in the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Set test environment variables
os.environ.setdefault("EPAM_OUTPUT__OUTPUT_DIR", "/tmp/veritas_test_output")


@pytest.fixture(autouse=True)
def clean_output_dir(tmp_path):
    """Ensure test output directory exists and is clean."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir
