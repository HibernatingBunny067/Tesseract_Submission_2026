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

# Server startup warm-up: JIT-compiles Geometry forward and VJP kernels before serving requests
def warmup_server():
    try:
        print("[geometry_tesseract] Warming up Metamaterial TPMS forward and adjoint kernels at boot...")
        dummy_input = api.InputSchema()
        _ = api.apply(dummy_input)
        # Warm up VJP
        _ = api.vector_jacobian_product(
            dummy_input,
            vjp_inputs={"cell_size", "tau_bridge"},
            vjp_outputs={"mean_porosity", "bridge_porosity"},
            cotangent_vector={"mean_porosity": 1.0, "bridge_porosity": 1.0}
        )
        print("[geometry_tesseract] Geometry warm-up complete! Ready to serve instantaneous requests.")
    except Exception as e:
        print(f"[geometry_tesseract] Warm-up notice: {e}")

if __name__ == "__main__":
    warmup_server()
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(app, host=host, port=port)
