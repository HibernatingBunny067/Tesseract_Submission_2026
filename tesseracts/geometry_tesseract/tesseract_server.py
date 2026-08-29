import sys
import os
import logging

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import src.fem.petsc_compat

# Redirect Tesseract run_<uuid> output folders to /tmp
os.environ["TESSERACT_OUTPUT_PATH"] = "/tmp/tesseract_runs"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"

import jax
jax.config.update('jax_enable_x64', True)

import uvicorn
from tesseract_core.runtime.serve import create_rest_api
import tesseracts.geometry_tesseract.tesseract_api as api

app = create_rest_api(api)

if __name__ == "__main__":
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
