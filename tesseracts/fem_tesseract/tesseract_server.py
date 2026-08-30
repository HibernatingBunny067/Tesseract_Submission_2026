import sys
import os
import logging

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
import src.fem.petsc_compat

# Suppress jax_fem logging in the server process
logging.getLogger("jax_fem").setLevel(logging.ERROR)
os.environ["JAX_FEM_LOG_LEVEL"] = "ERROR"

# Redirect Tesseract run_<uuid> output folders to /tmp
os.environ["TESSERACT_OUTPUT_PATH"] = "/tmp/tesseract_runs"
os.environ["MLFLOW_TRACKING_URI"] = "file:///tmp/mlruns"

import jax
jax.config.update('jax_enable_x64', True)

import uvicorn
from tesseract_core.runtime.serve import create_rest_api
import tesseracts.fem_tesseract.tesseract_api as api

app = create_rest_api(api)

# Server startup warm-up: JIT-compiles JAX forward and VJP kernels before serving requests
def warmup_server():
    try:
        print("[fem_tesseract] Warming up JAX-FEM forward and adjoint kernels at boot...")
        dummy_input = api.InputSchema()
        _ = api.apply(dummy_input)
        # Warm up VJP
        _ = api.vector_jacobian_product(
            dummy_input,
            vjp_inputs={"cell_size", "tau_bridge"},
            vjp_outputs={"compliance", "fracture_displacement"},
            cotangent_vector={"compliance": 1.0, "fracture_displacement": 1.0}
        )
        print("[fem_tesseract] JAX-FEM warm-up complete! Ready to serve instantaneous requests.")
    except Exception as e:
        print(f"[fem_tesseract] Warm-up notice: {e}")

if __name__ == "__main__":
    warmup_server()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
