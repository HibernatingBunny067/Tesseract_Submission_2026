# 🦴 Tesseract Differentiable Biomechanics
### Multi-Agent Inverse Design of Patient-Specific Orthopaedic Metamaterial Implants

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tesseract Core](https://img.shields.io/badge/Tesseract-Core_1.11-6366F1.svg)](https://pasteurlabs.ai)
[![JAX-FEM](https://img.shields.io/badge/JAX--FEM-Autodiff_FEA-FF4500.svg)](https://github.com/deepmodeling/jax-fem)
[![Docker](https://img.shields.io/badge/Docker-Multi--Container-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

---

## 📌 Hackathon Track & Executive Summary

* **Track:** Engineering & Inverse Design / Scientific Simulation & Inverse Problems
* **Core Problem:** The clinical trade-off in orthopaedic trauma surgery between rigid bone fixation (which causes non-union and stress shielding) and flexible fixation (which risks mechanical fatigue failure).
* **Our Solution:** A dual-stage differentiable biomechanics platform orchestrated by a 4-agent LangGraph state machine. It translates natural language surgical intent into patient-specific, 3D-printable, functionally graded Triply Periodic Minimal Surface (TPMS) bone plates verified against ASTM F382 and ISO 7206 in-silico standards.
* **Tesseract Composition:** Composes two independent Tesseract microservices (`fem_tesseract` and `geometry_tesseract`) through `tesseract_jax.apply_tesseract()`, enabling continuous reverse-mode adjoint vector-Jacobian products (VJPs) across container network boundaries in O(1) time.

---

## 1. Clinical Context & Biomechanical Framework

### The Clinical Dilemma: Fixation Rigidity vs. Biological Healing

When a human femur experiences a midshaft diaphyseal fracture, orthopaedic surgeons face a biomechanical conflict:

```
     CONVENTIONAL SOLID FIXATION (STIFFNESS MISMATCH & STRESS SHIELDING)
     ┌────────────────────────────────────────────────────────┐  ◄── Solid Titanium Plate (E = 110 GPa)
     │ 🔩 Screw              (Excess Rigidity)        🔩 Screw │
     └─────────────────────────┬──────────────────────────────┘
                               │
          ┌──────────────────┐ │ ┌──────────────────┐
          │   PROXIMAL BONE  │ ▼ │   DISTAL BONE    │
          │   (E = 18 GPa)   │▒▒▒│   (E = 18 GPa)   │  ◄── 2.0 mm Fracture Gap (E = 1.0 MPa)
          └──────────────────┘▒▒▒└──────────────────┘
                          (Insufficient Interfragmentary Motion)
```

1. **Perren's Strain Theory (Callus Formation Window):**
   * Secondary bone healing relies on controlled interfragmentary micro-motion between **0.15 mm and 0.35 mm** (150 to 350 microns) under ambulatory weight-bearing. This moderate flexure stimulates periosteal osteoblast differentiation and promotes callus formation.
   * Conventional solid titanium plates (110 GPa) or stainless steel plates (193 GPa) are 6 to 11 times stiffer than cortical bone (18 GPa). This severe stiffness mismatch suppresses micro-motion below 0.08 mm, resulting in delayed union or complete non-union.
2. **Wolff's Law & Stress Shielding:**
   * Because the solid metal plate carries nearly all the ambulatory load, the underlying cortical bone is shielded from mechanical stimulus. Over 6–18 months, this causes cortical bone resorption (osteopenia), screw loosening, and secondary peri-implant refracture upon hardware removal.

### The Solution: 5-Zone Functionally Graded TPMS Metamaterials

Our unified platform designs a continuous Triply Periodic Minimal Surface (TPMS) metamaterial implant with spatially tailored compliance:

```
                     FUNCTIONALLY GRADED 5-ZONE BONE PLATE CONSTRUCT
     
     ┌───────────────┬───────────────┬───────────────┬───────────────┬───────────────┐
     │ ZONE 1 (70%)  │ ZONE 2 (60%)  │ ZONE 3 (80%)  │ ZONE 4 (60%)  │ ZONE 5 (70%)  │
     │ Prox Anchor   │ Prox Gradient │ Bridge Center │ Dist Gradient │ Dist Anchor   │
     │ Dense Screws  │ Transition    │ Fracture Gap  │ Transition    │ Dense Screws  │
     └───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
     ◄── x = 30 mm                   x = 80 mm (Gap)                 x = 130 mm ────►
```

* **Anchorage Zones (Zones 1 & 5):** High metal volume fraction (porosity ~55–65%) surrounding the cortical screw holes to prevent local yielding and screw pull-out.
* **Transition Zones (Zones 2 & 4):** Smooth Gaussian-blended porosity gradients that eliminate stress concentrations.
* **Working Bridge Zone (Zone 3):** Highly porous metamaterial core directly spanning the 2.0 mm fracture gap, tuned precisely to achieve the patient's target micro-motion while preserving >= 55% load transfer back into the bone cortex.

---

## 2. Why This Workflow Needs Tesseract

Integrating non-linear structural continuum physics with analytical level-set geometry presents three fundamental engineering challenges that Tesseract solves:

### 1. Heterogeneous Computational Backends
* **Structural FEM Simulation (`fem_tesseract`):** Relies on compiled C/C++ sparse direct factorizations (SciPy SuperLU), Gmsh 10-node quadratic tetrahedral meshes (29,571 DOFs), and finite element assembly routines requiring system libraries (`libgomp`, `libgl1`).
* **Level-Set Porosity Engine (`geometry_tesseract`):** Relies on analytical implicit surface evaluations, 3D Marching Cubes, and spatial density field integration.
* **The Tesseract Advantage:** Packaging these engines into isolated, lightweight container microservices eliminates dependency conflicts, permits independent horizontal scaling (e.g., GPU acceleration for geometry SDF while CPU-parallel multi-threading handles SuperLU sparse factorization), and ensures identical reproducibility across developer, cloud, and clinical deployment environments.

### 2. Universal Differentiable Composition Over REST
* In standard microservice architectures, connecting two computational services over HTTP creates an optimization barrier: gradients cannot propagate across network endpoints.
* **The Tesseract Advantage:** With `tesseract_jax.apply_tesseract()`, JAX automatically constructs a custom Vector-Jacobian Product (VJP) rule over the network. When `jax.grad()` or `optax` optimizes the 12-DOF parameter vector `theta`, reverse-mode adjoint sensitivity gradients propagate across container boundaries in O(1) time:

$$\frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial u} \cdot \frac{\partial u}{\partial \theta_{\text{fem}}} + \frac{\partial \mathcal{L}}{\partial \text{Mass}} \cdot \frac{\partial \text{Mass}}{\partial \theta_{\text{geom}}}$$

### 3. Multi-Scale Modularity
* Isolating the macro CAD morphing stage (`PyGeM FFD`) and micro-lattice adjoint optimization from the high-level LLM reasoning graph allows changing the underlying simulation backend (e.g., swapping from 10-node tetrahedra to hexahedra or adding non-linear plasticity) without modifying the multi-agent clinical deliberation engine.

---

## 3. End-to-End System Architecture

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: MULTI-AGENT STATE GRAPH ORCHESTRATION (LangGraph State Machine)               │
│                                                                                        │
│   🧑‍⚕️ Clinical Interpreter      🔬 Materials Advisor        ⚙️ Optimization Controller   │
│    • Perren's Strain Target    • Alloy Selection (Ti/SS)   • 12-DOF WSD Adam Loop      │
│    • Persona Analysis          • TPMS Topology Selection   • Convergence Monitoring    │
│                                                                                        │
│   📋 Validation Auditor (Autonomous Closed-Loop Self-Correction)                       │
│    • ASTM F382 & ISO 7206 In-Silico Testing & Prescriptive Parameter Adjustments       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: DUAL-STAGE MULTI-SCALE OPTIMIZATION PIPELINE                                   │
│                                                                                        │
│  STAGE 1: Macro CAD Shape Morphing (PyGeM FFD)                                         │
│   • 5x4x4 Bernstein Free-Form Deformation Control Grid (8 interior control slices)     │
│   • Morphologically contours the global plate (Sagittal Y & Coronal Z bends) to bone   │
│                                                                                        │
│  STAGE 2: Differentiable JAX-FEM Lattice Synthesis (Dual Tesseract REST Engines)       │
│   • Tesseract 1 (Port 8000): JAX-FEM + SuperLU (5,951 TET10 Cells, 29,571 DOFs)       │
│   • Tesseract 2 (Port 8001): Analytical Level-Set SDF & Porosity Engine                │
│   • 12-DOF Optimization Vector:                                                        │
│     theta = [cell_size, tau_1..5, sigma_blend, t_top, t_bottom, pitch, span, fillet]   │
│   • Warmup-Stable-Decay (WSD) Adam Schedule with exact reverse-mode adjoint VJP        │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: IN-SILICO VERIFICATION, 3D VISUALIZATION & ADDITIVE MANUFACTURING EXPORT       │
│  • Automated In-Silico Battery: ASTM F382 Micro-Motion, Wolff's Law, Static FoS, Fatigue│
│  • Interactive 3D Plotly Renders: Metamaterial Porosity, Von Mises Stress, FoS Field   │
│  • Additive Manufacturing Export: Binary STL with FFD Conformal Warp + Gmsh .msh Mesh  │
│  • Audit Logging: logs/clinical_audit.log, logs/session_history.jsonl, logs/agent_... │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Mathematical Formulations

### 1. Multi-TPMS Level-Set Architecture
The porous core is synthesized using analytical level-set functions where the solid-void boundary is defined by $F(x, y, z) - \tau(x) = 0$ (wavenumber $k = \frac{2\pi}{\text{cell\_size}}$):

* **Schwarz Primitive (P):**
  $$F_P(x, y, z) = \cos(kx) + \cos(ky) + \cos(kz)$$
  *Optimal for high fluid permeability, vascularization, and periosteal callus growth.*

* **Schoen Gyroid (G):**
  $$F_G(x, y, z) = 1.5 \cdot \Big(\sin(kx)\cos(ky) + \sin(ky)\cos(kz) + \sin(kz)\cos(kx)\Big)$$
  *Optimal for isotropic compliance and balanced multi-axial shear resistance.*

* **Schwarz Diamond (D):**
  $$F_D(x, y, z) = 1.8 \cdot \Big(\cos(kx)\cos(ky)\cos(kz) - \sin(kx)\sin(ky)\sin(kz)\Big)$$
  *Optimal for maximum torsional rigidity and high-energy spiral fracture patterns.*

### 2. 5-Zone Continuous Density Blending
To prevent notch stress concentrations, local threshold $\tau(x)$ transitions smoothly via Gaussian weighting:

$$\tau(x) = \frac{\sum_{i=1}^5 \tau_i \cdot w_i(x)}{\sum_{i=1}^5 w_i(x)}, \quad w_i(x) = \exp\left(-\left(\frac{x - x_i}{\sigma_{\text{blend}}}\right)^2\right)$$

### 3. Gibson-Ashby Modulus Homogenization
The effective local Young's modulus $E_{\text{eff}}(x)$ varies continuously with the relative density:

$$\text{Porosity}(\tau) = 0.547 + \left(\frac{\tau - 0.10}{1.35}\right) \cdot (0.881 - 0.547)$$

$$E_{\text{eff}}(\tau) = E_{\text{solid}} \cdot (1.0 - \text{Porosity})^\gamma$$

*(where $\gamma = 1.60$ for Ti-6Al-4V ELI and $\gamma = 1.55$ for 316L Stainless Steel)*

### 4. Objective Loss Formulation
The optimizer minimizes a multi-objective loss balancing micro-motion targeting, compliance energy, and mass penalty:

$$\mathcal{L}(\theta) = \left( \frac{\delta_{\text{achieved}}(\theta) - \delta_{\text{target}}}{\delta_{\text{target}}} \cdot 22.0 \right)^2 \cdot 2.0 + c_{\text{compliance}} \cdot \mathcal{C}(u) + w_{\text{mass}} \cdot \text{ReLU}\Big(\frac{\text{Mass}(\theta)}{\text{Mass}_{\text{solid}}} - \text{MaxMass}\Big)^2$$

---

## 5. Automated In-Silico Verification Suite

Every synthesized implant is subjected to an automated 4-part in-silico testing protocol before clinical sign-off:

| Benchmark | Test Standard | Pass Criteria | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **Micro-Motion Window** | ASTM F382 / AO Foundation | Within $\pm 20\%$ of Target | Ensures interfragmentary strain induces secondary callus bridging without non-union. |
| **Stress Shielding Index** | Wolff's Law Biomechanical Load Ratio | $\ge 55.0\%$ Load Transfer | Preserves cortical bone density and prevents post-operative osteopenia and bone loss. |
| **Static Yield Proof** | ASTM F382 4-Point Bending Proof | Factor of Safety $\ge 1.50\times$ | Prevents permanent plastic bending under full single-leg stance gait loading ($800\text{ N}$). |
| **Cyclic Fatigue Endurance** | ISO 7206 ($10^6$ Ambulatory Cycles) | Endurance Ratio $\ge 1.20\times$ | Guarantees implant survival across the complete 6-month healing horizon without fatigue rupture. |

---

## 6. Certified Biomaterials Database

| Biomaterial | Elastic Modulus | Mass Density | Yield Strength | Fatigue Strength ($10^6$ cycles) | Regulatory Standard |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Ti-6Al-4V ELI (Grade 5)** | $110.0\text{ GPa}$ | $4.43\text{ g/cm}^3$ | $880\text{ MPa}$ | $510\text{ MPa}$ | ISO 5832-3 / ASTM F136 |
| **316L Stainless Steel** | $193.0\text{ GPa}$ | $8.00\text{ g/cm}^3$ | $220\text{ MPa}$ | $200\text{ MPa}$ | ASTM F138 |

---

## 7. Quick Start & Execution Guide

### Prerequisites
* **Operating System:** macOS (Apple Silicon / Intel) or Linux (Ubuntu 22.04+)
* **Docker:** Docker Desktop installed and running
* **Python:** Python 3.12 (if running in local virtual environment)

---

### Option A: One-Click Docker Compose (Recommended)

From the root of the repository:

```bash
# 1. Build and run all 3 container microservices
docker compose up --build
```
*(Or use `./run_docker.sh`)*

Open your browser at **`http://localhost:8501`**.

To shut down:
```bash
docker compose down
```

---

### Option B: Local Virtual Environment Run

```bash
# 1. Activate the environment
source .venv/bin/activate

# 2. Launch the application (starts servers and Streamlit UI)
./run
```

Open your browser at **`http://localhost:8501`**.

---

### Option C: Native Tesseract CLI Container Workflow

```bash
# 1. Build the two container images from their context directories
tesseract build tesseracts/fem_tesseract --tag fem_tesseract:latest
tesseract build tesseracts/geometry_tesseract --tag geometry_tesseract:latest

# 2. Serve the containers in the background
tesseract serve fem_tesseract:latest --port 8000 &
tesseract serve geometry_tesseract:latest --port 8001 &

# 3. Launch the dashboard
streamlit run app.py
```

---

### Running Integration Tests

To run the full unit and integration test suite:

```bash
./.venv/bin/python tests/test_agent_system.py
```

---

## 8. Repository Layout

```
Tesseract_Hackathon/
├── app.py                         ◄── Streamlit interactive co-design dashboard
├── run                            ◄── Local runner script with automatic process traps
├── run_docker.sh                  ◄── One-click Docker Compose launcher
├── docker-compose.yml             ◄── 3-tier microservice container orchestration
├── Dockerfile                     ◄── Container definition for Dashboard & Multi-Agent UI
├── .dockerignore                  ◄── Excludes .venv and cache files for rapid container builds
├── LICENSE                        ◄── Official Apache 2.0 open-source license
├── REQUIREMENTS.txt               ◄── Python package dependencies
├── README.md                      ◄── Complete scientific and technical report
├── logs/
│   ├── agent_deliberation.log     ◄── Human-readable multi-agent thought and audit trail
│   ├── agent_session.jsonl        ◄── Structured machine-readable agent event stream
│   ├── clinical_audit.log         ◄── End-to-end clinical optimization audit trail
│   └── session_history.jsonl      ◄── Optimization metrics trajectory history
├── tests/
│   └── test_agent_system.py       ◄── Multi-agent, FFD morphing, and solver test suite
├── src/
│   ├── agent/
│   │   ├── state.py               ◄── DesignState TypedDict and message schemas
│   │   ├── prompts.py             ◄── Specialist system prompts for the 4 clinical agents
│   │   ├── llm_provider.py        ◄── Multi-provider LLM connector (Gemini, Groq, Ollama)
│   │   ├── nodes.py               ◄── Specialist agent handlers and self-correction loop
│   │   ├── graph.py               ◄── LangGraph state machine orchestrator
│   │   ├── optimize.py            ◄── Differentiable 12-DOF JAX WSD Adam optimizer
│   │   └── optimize_cad.py        ◄── Stage 1 PyGeM FFD CAD shape optimizer
│   ├── geometry/
│   │   ├── morph.py               ◄── PyGeM FFD 3D mesh morphing engine
│   │   ├── plot_plotly.py         ◄── 3D Marching Cubes, Von Mises contours & STL generator
│   │   └── model.py               ◄── Parametric bone-plate Gmsh CAD generator
│   ├── fem/
│   │   ├── forward.py             ◄── JAX-FEM forward solver and morphed mesh dispatcher
│   │   ├── problem.py             ◄── Constitutive equations, Gaussian blend & SuperLU solver
│   │   ├── materials.py           ◄── Certified orthopaedic biomaterials database
│   │   ├── validation.py          ◄── ASTM F382 & ISO 7206 in-silico verification engine
│   │   └── data/
│   │       ├── model.msh          ◄── Master 10-node tetrahedral mesh (29,571 DOFs)
│   │       └── morphed_model.msh  ◄── Stage 1 FFD patient-conformed mesh
│   ├── ui/
│   │   ├── charts.py              ◄── Interactive Plotly tracking figures
│   │   └── components.py          ◄── Glassmorphic UI cards, badges and metrics tables
│   └── utils/
│       └── logger.py              ◄── Multi-stream clinical and agent audit logging system
└── tesseracts/
    ├── fem_tesseract/
    │   ├── tesseract_api.py       ◄── OpenAPI endpoint definitions (apply, VJP gradients)
    │   ├── tesseract_server.py    ◄── Uvicorn ASGI server (Port 8000)
    │   ├── tesseract_config.yaml  ◄── Pasteur Labs container build configuration
    │   ├── tesseract_requirements.txt ◄── FEM container requirements
    │   └── Dockerfile             ◄── Isolated container definition
    └── geometry_tesseract/
        ├── tesseract_api.py       ◄── OpenAPI endpoint definitions (apply, VJP gradients)
        ├── tesseract_server.py    ◄── Uvicorn ASGI server (Port 8001)
        ├── tesseract_config.yaml  ◄── Pasteur Labs container build configuration
        ├── tesseract_requirements.txt ◄── Geometry container requirements
        └── Dockerfile             ◄── Isolated container definition
```

---

## 9. Citation & Acknowledgments

This project was built for the **Tesseract Hackathon 2026**. 

### References
1. **Perren, S. M.** (2002). *Evolution of the internal fixation of long bone fractures: the scientific basis of biological internal fixation.* The Journal of Bone and Joint Surgery (British Volume), 84(8), 1093-1110.
2. **Wolff, J.** (1986). *The Law of Bone Remodelling.* Springer-Verlag Berlin Heidelberg.
3. **Gibson, L. J., & Ashby, M. F.** (1997). *Cellular Solids: Structure and Properties.* Cambridge University Press.
4. **Pasteur Labs.** (2025). *Tesseract: Universal, Autodiff-Native Software Components.* https://pasteurlabs.ai
5. **JAX-FEM Team.** (2023). *Differentiable Finite Element Method in JAX.* https://github.com/deepmodeling/jax-fem