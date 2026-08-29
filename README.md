# 🦴 Tesseract Differentiable Biomechanics
### Multi-Agent Inverse Design of Patient-Specific Orthopaedic Metamaterial Implants

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Tesseract Core](https://img.shields.io/badge/Tesseract-Core_1.11-6366F1.svg)](https://pasteurlabs.ai)
[![JAX-FEM](https://img.shields.io/badge/JAX--FEM-Autodiff_FEA-FF4500.svg)](https://github.com/deepmodeling/jax-fem)
[![Docker](https://img.shields.io/badge/Docker-Multi--Container-2496ED.svg?logo=docker&logoColor=white)](https://www.docker.com/)

> **Track:** Engineering & Inverse Design / Scientific Simulation & Inverse Problems

![Tesseract Biomechanics Demo](artifacts/demo.png)

---

## 🌟 The Elevator Pitch

Current orthopaedic trauma implants face a biomechanical paradox: rigid titanium plates cause stress shielding and bone resorption, while flexible plates risk mechanical fatigue failure. 

**Tesseract Differentiable Biomechanics** solves this by orchestrating a multi-agent AI system with dual Tesseract microservices to inversely design patient-specific, 3D-printable Triply Periodic Minimal Surface (TPMS) bone plates. By leveraging JAX-based differentiable Finite Element Analysis (FEA) across container boundaries, we optimize 12-DOF metamaterial parameters in real-time to guarantee optimal interfragmentary micro-motion for callus formation, automatically verified against ASTM F382 and ISO 7206 standards.

---

## 🚀 Key Innovations

* 🧠 **Agentic Inverse Design:** A 4-agent LangGraph state machine interprets natural language surgical intent (e.g., *"Elderly osteoporotic patient needing 0.3mm micro-motion"*) and autonomously drives the physics optimization loop with closed-loop self-correction.
* 🔗 **Differentiable Cross-Container Gradients:** Breaks traditional microservice barriers. By using Tesseract's `apply_tesseract()`, we propagate reverse-mode adjoint sensitivities (VJPs) between a compiled C++ FEM solver and a Python geometry engine via REST APIs in **O(1) time**.
* 🦴 **Functionally Graded Metamaterials:** Generates continuous 5-zone porosity gradients (Schwarz P, Gyroid, Diamond) that perfectly match cortical bone stiffness, eliminating notch stress concentrations.
* 🏥 **Automated In-Silico Verification:** Every synthesized implant automatically undergoes rigorous virtual testing for micro-motion windows, Wolff's Law load transfer, static yield, and cyclic fatigue before export.

---

## 🏥 1. The Clinical Dilemma: Fixation Rigidity vs. Biological Healing

When a human femur experiences a midshaft diaphyseal fracture, orthopaedic surgeons face a biomechanical conflict:

```text
     CONVENTIONAL SOLID FIXATION (STIFFNESS MISMATCH & STRESS SHIELDING)
     ┌────────────────────────────────────────────────────────┐  ◄── Solid Titanium Plate (E = 110 GPa)
     │ 🔩 Screw              (Excess Rigidity)        🔩 Screw │
     └─────────────────────────┬──────────────────────────────┘
                               │
          ┌──────────────────┐ │ ┌──────────────────┐
          │   PROXIMAL BONE  │ ▼ │   DISTAL BONE    │
          │   (E = 18 GPa)   │▒▒▒│   (E = 18 GPa)   │  ◄── 2.0 mm Fracture Gap 
          └──────────────────┘▒▒▒└──────────────────┘
                          (Insufficient Interfragmentary Motion)
```

* **Perren's Strain Theory:** Secondary bone healing requires controlled micro-motion between **0.15 mm and 0.35 mm**. Conventional plates suppress this below 0.08 mm, causing non-union.
* **Wolff's Law & Stress Shielding:** Solid metal carries the ambulatory load, shielding the bone. Over 6–18 months, this causes bone resorption and secondary refracture upon hardware removal.

### The Solution: 5-Zone Functionally Graded TPMS

```text
                     FUNCTIONALLY GRADED 5-ZONE BONE PLATE CONSTRUCT
     
     ┌───────────────┬───────────────┬───────────────┬───────────────┬───────────────┐
     │ ZONE 1 (70%)  │ ZONE 2 (60%)  │ ZONE 3 (80%)  │ ZONE 4 (60%)  │ ZONE 5 (70%)  │
     │ Prox Anchor   │ Prox Gradient │ Bridge Center │ Dist Gradient │ Dist Anchor   │
     └───────────────┴───────────────┴───────────────┴───────────────┴───────────────┘
```

---

## ⚙️ 2. Why This Needs Tesseract

Integrating non-linear structural continuum physics with analytical level-set geometry presents three fundamental engineering challenges that Tesseract elegantly solves:

1. **Heterogeneous Computational Backends:** Structural FEM relies on compiled C/C++ sparse direct factorizations (SciPy SuperLU) and Gmsh tetrahedral meshes. Geometry relies on analytical implicit surface evaluations. Tesseract packages these into isolated microservices, eliminating dependency conflicts and permitting independent horizontal scaling.
2. **Universal Differentiable Composition Over REST:** In standard architectures, connecting two services over HTTP creates an optimization barrier: gradients cannot propagate across network endpoints. Tesseract constructs a custom Vector-Jacobian Product (VJP) rule over the network:

$$ \frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial u} \cdot \frac{\partial u}{\partial \theta_{\text{fem}}} + \frac{\partial \mathcal{L}}{\partial \text{Mass}} \cdot \frac{\partial \text{Mass}}{\partial \theta_{\text{geom}}} $$

3. **Multi-Scale Modularity:** Isolating the macro CAD morphing stage (`PyGeM FFD`) and micro-lattice adjoint optimization from the high-level LLM reasoning graph allows swapping the underlying simulation backend without modifying the multi-agent clinical deliberation engine.

---

## 🏛️ 3. End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 1: MULTI-AGENT STATE GRAPH ORCHESTRATION (LangGraph State Machine)               │
│                                                                                        │
│   🧑⚕️ Clinical Interpreter      🔬 Materials Advisor        ⚙️ Optimization Controller   │
│    • Perren's Strain Target    • Alloy Selection (Ti/SS)   • 12-DOF WSD Adam Loop      │
│                                                                                        │
│   📋 Validation Auditor (Autonomous Closed-Loop Self-Correction)                       │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 2: DUAL-STAGE MULTI-SCALE OPTIMIZATION PIPELINE                                   │
│                                                                                        │
│  STAGE 1: Macro CAD Shape Morphing (PyGeM FFD)                                         │
│   • 5x4x4 Bernstein Free-Form Deformation Control Grid (morphs to patient anatomy)     │
│                                                                                        │
│  STAGE 2: Differentiable JAX-FEM Lattice Synthesis (Dual Tesseract REST Engines)       │
│   • Tesseract 1 (Port 8000): JAX-FEM + SuperLU (5,951 TET10 Cells, 29,571 DOFs)       │
│   • Tesseract 2 (Port 8001): Analytical Level-Set SDF & Porosity Engine                │
└───────────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ TIER 3: IN-SILICO VERIFICATION, 3D VISUALIZATION & ADDITIVE MANUFACTURING EXPORT       │
│  • Automated In-Silico Battery: ASTM F382 Micro-Motion, Static FoS, Fatigue            │
│  • Interactive 3D Plotly Renders: Metamaterial Porosity, Von Mises Stress, FoS Field   │
│  • Additive Manufacturing Export: Binary STL with FFD Conformal Warp                   │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧮 4. Mathematical & Physical Formulations

### Multi-TPMS Level-Set Architecture
The porous core is synthesized using analytical level-set functions where the solid-void boundary is defined by $F(x, y, z) - \tau(x) = 0$:

* **Schwarz Primitive (P):** $F_P = \cos(kx) + \cos(ky) + \cos(kz)$ *(High fluid permeability for vascularization)*
* **Schoen Gyroid (G):** $F_G = 1.5 (\sin(kx)\cos(ky) + \dots)$ *(Isotropic compliance and shear resistance)*

### 5-Zone Continuous Density Blending
To prevent notch stress concentrations, local threshold $\tau(x)$ transitions smoothly via Gaussian weighting:

$$ \tau(x) = \frac{\sum_{i=1}^5 \tau_i \cdot w_i(x)}{\sum_{i=1}^5 w_i(x)}, \quad w_i(x) = \exp\left(-\left(\frac{x - x_i}{\sigma_{\text{blend}}}\right)^2\right) $$

### Objective Loss Formulation
The optimizer minimizes a multi-objective loss balancing micro-motion targeting, compliance energy, and mass penalty:

$$ \mathcal{L}(\theta) = \left( \frac{\delta_{\text{achieved}}(\theta) - \delta_{\text{target}}}{\delta_{\text{target}}} \cdot 22.0 \right)^2 \cdot 2.0 + c_{\text{compliance}} \cdot \mathcal{C}(u) + w_{\text{mass}} \cdot \text{ReLU}\Big(\frac{\text{Mass}(\theta)}{\text{Mass}_{\text{solid}}} - \text{MaxMass}\Big)^2 $$

---

## 🛡️ 5. Automated In-Silico Verification Suite

Every synthesized implant is subjected to an automated 4-part in-silico testing protocol before clinical sign-off:

| Benchmark | Test Standard | Pass Criteria | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **Micro-Motion Window** | ASTM F382 / AO Foundation | Within $\pm 20\%$ of Target | Ensures interfragmentary strain induces secondary callus bridging. |
| **Stress Shielding Index** | Wolff's Law Load Ratio | $\ge 55.0\%$ Load Transfer | Preserves cortical bone density and prevents post-operative osteopenia. |
| **Static Yield Proof** | ASTM F382 4-Point Bending | Factor of Safety $\ge 1.50\times$ | Prevents permanent plastic bending under full single-leg stance gait. |
| **Cyclic Fatigue Endurance** | ISO 7206 ($10^6$ Cycles) | Endurance Ratio $\ge 1.20\times$ | Guarantees implant survival across the 6-month healing horizon. |

---

## 💻 6. Quick Start & Deployment

### Prerequisites
* **OS:** macOS (Apple Silicon / Intel) or Linux (Ubuntu 22.04+)
* **Docker:** Docker Desktop installed and running
* **Python:** 3.12 (if running in local virtual environment)

### Option A: One-Click Docker Compose (Recommended)

From the root of the repository, launch the full 3-tier microservice stack:

```bash
# Build and run all container microservices
docker compose up --build

# Alternatively, use the helper script:
./run_docker.sh
```
Open your browser at **`http://localhost:8501`**.

### Option B: Native Tesseract CLI Workflow

```bash
# 1. Build the isolated container images
tesseract build tesseracts/fem_tesseract --tag fem_tesseract:latest
tesseract build tesseracts/geometry_tesseract --tag geometry_tesseract:latest

# 2. Serve the containers in the background
tesseract serve fem_tesseract:latest --port 8000 &
tesseract serve geometry_tesseract:latest --port 8001 &

# 3. Launch the Streamlit dashboard
streamlit run app.py
```

### Running the Test Suite
To verify the agent system, FFD morphing, and solvers:
```bash
python tests/test_agent_system.py
```

---

## 📂 7. Repository Layout

```text
Tesseract_Submission_2026/
├── app.py                         ◄── Streamlit interactive co-design dashboard
├── run_docker.sh                  ◄── One-click Docker Compose launcher
├── docker-compose.yml             ◄── 3-tier microservice container orchestration
├── REQUIREMENTS.txt               ◄── Python package dependencies
├── src/
│   ├── agent/                     ◄── LangGraph multi-agent state machine & prompts
│   ├── geometry/                  ◄── PyGeM FFD morphing & Plotly 3D STL generator
│   ├── fem/                       ◄── JAX-FEM forward solver & ASTM validation engine
│   └── ui/                        ◄── Glassmorphic UI components and charting
├── tesseracts/                    ◄── Tesseract Core microservices
│   ├── fem_tesseract/             ◄── Containerized JAX-FEM + SuperLU solver
│   └── geometry_tesseract/        ◄── Containerized Level-Set SDF engine
└── logs/                          ◄── Clinical audit trails and optimization history
```

---

## 📚 8. Citation & Acknowledgments

This project was built for the **Tesseract Hackathon 2026**. 

**References:**
1. Perren, S. M. (2002). *Evolution of the internal fixation of long bone fractures.* The Journal of Bone and Joint Surgery.
2. Gibson, L. J., & Ashby, M. F. (1997). *Cellular Solids: Structure and Properties.* Cambridge University Press.
3. Pasteur Labs. (2025). *Tesseract: Universal, Autodiff-Native Software Components.* https://pasteurlabs.ai
4. JAX-FEM Team. (2023). *Differentiable Finite Element Method in JAX.* https://github.com/deepmodeling/jax-fem