# 🦴 Tesseract Differentiable Biomechanics: Agentic Patient-Specific Implant Optimization

An end-to-end differentiable biomechanics framework that translates natural language surgeon requirements into patient-specific, 3D-printable, functionally graded TPMS orthopaedic implants. Powered by **JAX-FEM**, **Dual Tesseract REST Microservices**, **SciPy SuperLU Accelerated Solvers**, and **Groq LPU Inference**.

---

## 1. Executive Summary & Clinical Problem

### The Clinical Dilemma: Rigid Fixation vs. Secondary Healing
In orthopaedic trauma surgery, fixing femur fractures with conventional solid metal plates presents a major biomechanical paradox:

```
     CONVENTIONAL SOLID TITANIUM PLATE (STIFFNESS MISMATCH)
     ┌────────────────────────────────────────────────────────┐  ◄── Solid Titanium Plate (E = 110 GPa)
     │ 🔩 Screw             (Too Stiff!)             🔩 Screw │
     └─────────────────────────┬──────────────────────────────┘
                               │
          ┌──────────────────┐ │ ┌──────────────────┐
          │   PROXIMAL BONE  │ ▼ │   DISTAL BONE    │
          │   (E = 18 GPa)   │▒▒▒│   (E = 18 GPa)   │  ◄── 2.0 mm Fracture Gap (E = 1.0 MPa)
          └──────────────────┘▒▒▒└──────────────────┘
                          (No Motion = Delayed Union)
```

1. **Perren's Strain Theory (Callus Stimulation)**:
   - Secondary bone healing requires a physiological interfragmentary micro-motion window between **0.15 mm and 0.35 mm** (150 to 350 microns) under normal walking gait. This controlled micro-flexure triggers periosteal osteoblast differentiation and forms a strong bridging bone callus.
   - Completely rigid plates prevent this motion, resulting in delayed bone union or non-union failure.
2. **Wolff's Law & Stress Shielding**:
   - Solid titanium (110 GPa) and stainless steel (193 GPa) are 6 to 10 times stiffer than cortical bone (18 GPa).
   - Solid metal plates carry almost all ambulatory loads, "shielding" underlying bone from mechanical stress and causing **cortical bone resorption, plate loosening, and refracture**.

### The Solution: 5-Zone Functionally Graded Metamaterial Plates
Our platform synthesizes a continuous Triply Periodic Minimal Surface (TPMS) metamaterial plate:
- **Rigid Solid Ends (Proximal & Distal)**: Thick struts and high metal density to provide solid anchorage for bicortical locking screws.
- **Compliant Porous Bridge**: Tuned microscopic lattice directly over the fracture gap to deliver the exact target micro-motion (e.g., 0.20 mm) while preserving $\ge 55\%$ of natural cortical load transfer.

---

## 2. End-to-End System Architecture

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. SURGEON NATURAL LANGUAGE INTERFACE                                       │
  │    "A 22-year-old athlete with spiral femur fracture needing high torsion"  │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 2. REASONING AGENT (Groq LPU / Llama 3.3 70B & 120B / Local Biomechanical NLP)│
  │    • Clinical Objective: Rigid Athletic Fixation                            │
  │    • Target Micro-Motion: 0.12 mm                                           │
  │    • Material Choice: Ti-6Al-4V ELI (Grade 5)                               │
  │    • Metamaterial Topology: Schwarz Diamond (D)                             │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 3. DUAL TESSERACT REST MICROSERVICES ENGINES                                │
  │    ┌───────────────────────────────────┐ ┌────────────────────────────────┐ │
  │    │ TESSERACT 1: FEM ADJOINT ENGINE   │ │ TESSERACT 2: GEOMETRY ENGINE   │ │
  │    │ (Port 8000 · JAX-FEM + SuperLU)   │ │ (Port 8001 · Differentiable SDF)│ │
  │    │ • 5,951 TET10 Cells (29,571 DOFs) │ │ • Continuous 3D Volume Integral│ │
  │    │ • Exact VJP Gradients in O(1) time│ │ • True Physical Lattice Density│ │
  │    └───────────────────────────────────┘ └────────────────────────────────┘ │
  │    • Warmup-Stable-Decay (WSD) Adam Optimizer converges in 10-15 steps      │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 4. AUTOMATED IN-SILICO VERIFICATION BATTERY                                 │
  │    • ASTM F382: Fracture micro-motion target verification (±15% band)       │
  │    • Wolff's Law Index: Cortical stress shielding preservation ≥ 55.0%      │
  │    • ASTM F382 Static Proof Test: Yield Safety Factor S_f ≥ 1.50            │
  │    • ISO 7206: Cyclic fatigue endurance margin at 10^6 gait cycles          │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 5. CAD / CAM 3D SLICER EXPORT & FULL CONTINUUM VISUALIZATION                │
  │    • Marked 3D Coordinate Axes (X: Length, Y: Depth, Z: Width in mm)        │
  │    • Full 3D Von Mises Stress Heatmap & Factor of Safety (FoS) Field        │
  │    • Marching Cubes isocontouring generates 3.10 MB - 5.59 MB Binary .STL   │
  │    • Permanent audit trail written to logs/clinical_audit.log               │
  └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Geometric & Anatomical Specifications

```
                              ANATOMICAL CONSTRUCT LAYOUT
                              
    ◄──────────────────────── Total Bone Length: 160 mm (16 cm) ────────────────────────►
    ┌───────────────────┬───────────────────────────────────────────┬───────────────────┐
    │ Proximal Cortex   │          2.0 mm Fracture Gap              │ Distal Cortex     │
    │ Outer Diam: 24 mm │          (Located at x = 80 mm)           │ Outer Diam: 24 mm │
    │ Inner Diam: 16 mm │                                           │ Inner Diam: 16 mm │
    └───────────────────┴───────────────────────────────────────────┴───────────────────┘
              ▲                                                               ▲
              │  ◄──────────── Plate Length: 100 mm (10 cm) ──────────────►  │
              └─────────┌───────────────────────────────────────────┐─────────┘
                        │   Implant Plate: 100 mm x 16 mm x 6 mm    │
                        └───────────────────────────────────────────┘
```

### Detailed Component Dimensions:
1. **Femoral Diaphysis Segment**:
   - Total Length: **160.0 mm** (16.0 cm).
   - Outer Cortical Diameter: **24.0 mm** (Radius = 12.0 mm).
   - Inner Marrow Diameter: **16.0 mm** (Radius = 8.0 mm).
   - Cortical Wall Thickness: **4.0 mm** of solid cortical bone ($E = 18.0\,\text{GPa}$).
   - Fracture Gap Width: **2.0 mm** ($x = 79.0\,\text{mm} \to 81.0\,\text{mm}$, $E = 1.0\,\text{MPa}$).
2. **Implant Fixation Plate & Sandwich Architecture**:
   - Total Length: **100.0 mm** (Standard 6-hole anatomical locking plate spanning $x = 30.0\,\text{mm} \to 130.0\,\text{mm}$).
   - Plate Width: **16.0 mm** ($z = -8.0\,\text{mm} \to +8.0\,\text{mm}$).
   - Plate Thickness: **6.0 mm** ($y = 11.0\,\text{mm} \to 17.0\,\text{mm}$).
   - **Solid Cortex Skin Shell (0.20 - 1.0 mm)**: Outer top, bottom, and lateral surfaces are maintained as solid metal to prevent muscle/soft tissue adhesion and ensure smooth cortical bone seating.
   - **Internal TPMS Porous Core**: Houses the continuous minimal surface lattice directly inside the sandwich bridge over the fracture gap.
   - **6-Hole AO LCP Fixation Screws**: Six 4.5 mm cylindrical through-holes (radius = 2.25 mm) positioned at $x = [35.0, 50.0, 65.0, 95.0, 110.0, 125.0]\,\text{mm}$ with top countersink chamfers.
   - Metamaterial Unit Cell Size: **3.5 mm – 7.5 mm** (default 5.0 mm).
3. **Finite Element Discretization**:
   - Total Elements: **5,951 cells** (quadratic 10-node tetrahedrons / TET10).
   - Total Degrees of Freedom: **9,857 nodes $\times$ 3 = 29,571 DOFs**.
   - Mesh Conformance: Shared interface nodes representing fixed-angle bicortical locking screws (Locking Compression Plate / LCP principle).
4. **Boundary Conditions & Joint Loading**:
   - Proximal End (Hip): Rigidly clamped (cantilever fixity).
   - Distal End (Knee): Subjected to vertical downward ground reaction traction ($-1.0\,\text{MPa}$, simulating single-leg stance during walking gait).

---

## 4. Dual Tesseract REST Microservices Architecture

The system modularizes physics simulation and geometry synthesis into two independent microservices conforming to the official Tesseract OpenAPI protocol:

| Microservice | Port | Implementation | Core Capabilities |
| :--- | :--- | :--- | :--- |
| **`fem_tesseract`** | **8000** | JAX-FEM + SciPy SuperLU | Forward FEM solve, nodal displacement extraction, 3D relative osteotomy micro-motion, compliance energy, and reverse-mode adjoint VJP $\partial L / \partial \theta$. |
| **`geometry_tesseract`** | **8001** | NumPy + Marching Cubes SDF | Differentiable 3D SDF synthesis, analytical unit-cell level-set porosity evaluation, volume-integrated mass fraction, and binary STL CAD export. |

---

## 5. Solver Benchmark & 2.43x Performance Acceleration

To maximize optimization speed while guaranteeing gradient stability, we benchmarked multiple linear equation solver backends for forward and adjoint passes:

| Solver Backend | Forward Step Time | Adjoint VJP Step Time | Total Step Time | Stability / Convergence |
| :--- | :--- | :--- | :--- | :--- |
| **PETSc Direct LU (MUMPS)** | ~7.2s | ~6.8s | **14.0s** | Stable |
| **PETSc Iterative CG + Jacobi** | ~1.8s | Failed (Diverged) | N/A | Unstable on adjoint backward pass |
| **SciPy SuperLU (`spsolve_solver`)** | **~2.9s** | **~2.8s** | **5.7s (2.43x Faster)** | **100% Unconditionally Stable** |

By configuring `SOLVER_OPTIONS` and `ADJOINT_SOLVER_OPTIONS` to use native C SuperLU (`spsolve_solver`), each gradient descent step runs in **5.7s**, allowing full 15-step optimization to complete in **under 90 seconds**.

---

## 6. Multi-TPMS Metamaterial Formulations

The implant lattice is governed by Triply Periodic Minimal Surface (TPMS) level sets $F(x,y,z) - \tau(x) = 0$:

```
                          THREE TPMS ARCHITECTURES
                          
   Schwarz Primitive (P)           Schoen Gyroid (G)           Schwarz Diamond (D)
  ┌─────────────────────────┐    ┌─────────────────────────┐  ┌─────────────────────────┐
  │ cos(kx)+cos(ky)+cos(kz) │    │ sin(x)cos(y)+sin(y)cos(z│  │ cos(x)cos(y)cos(z) -   │
  │                         │    │ +sin(z)cos(x)           │  │ sin(x)sin(y)sin(z)      │
  │ • Max Fluid Permeability│    │ • High Shear Strength   │  │ • High Torsion Grip     │
  │ • Nutrient Transport    │    │ • Isotropic Compliance  │  │ • Multi-Axial Bending   │
  └─────────────────────────┘    └─────────────────────────┘  └─────────────────────────┘
```

### Mathematical Level-Set Equations ($k = 2\pi / \text{cell\_size}$):
1. **Schwarz Primitive (P)** *(High Vascularization & Fluid Permeability)*:
   $$F_P(x,y,z) = \cos(kx) + \cos(ky) + \cos(kz)$$
2. **Schoen Gyroid (G)** *(Superior Shear Strength & Energy Absorption)*:
   $$F_G(x,y,z) = 1.5 \cdot (\sin(kx)\cos(ky) + \sin(ky)\cos(kz) + \sin(kz)\cos(kx))$$
3. **Schwarz Diamond (D)** *(Maximum Torsional Stiffness & Multi-Axial Rigidity)*:
   $$F_D(x,y,z) = 1.8 \cdot (\cos(kx)\cos(ky)\cos(kz) - \sin(kx)\sin(ky)\sin(kz))$$

### Continuous 5-Zone Partition of Unity:
The local density threshold $\tau(x)$ is blended continuously across 5 anatomical zones:
$$\tau(x) = \frac{\sum_{i=1}^5 \tau_i w_i(x)}{\sum_{i=1}^5 w_i(x)}$$
- **Zone 1 ($x = 35\,\text{mm}$)**: Far Proximal Screw Anchor (Dense metal).
- **Zone 2 ($x = 57\,\text{mm}$)**: Proximal Stress Transition (Graded compliance).
- **Zone 3 ($x = 80\,\text{mm}$)**: Fracture Bridge Gap Center (Porous & compliant).
- **Zone 4 ($x = 103\,\text{mm}$)**: Distal Stress Transition (Graded compliance).
- **Zone 5 ($x = 125\,\text{mm}$)**: Far Distal Screw Anchor (Dense metal).
- **Gaussian Kernel**: $w_i(x) = \exp\left( - \left(\frac{x - x_i}{\sigma_{\text{blend}}}\right)^2 \right)$ where $\sigma_{\text{blend}}$ is optimized dynamically.

### Physical Level-Set Porosity & GA Modulus Scaling:
For Schwarz-P surfaces, the level set threshold $\tau \in [0.10, 1.45]$ maps to true unit-cell lattice porosity:
$$\text{Porosity}(\tau) = 54.7\% + \left(\frac{\tau - 0.10}{1.35}\right) \cdot (88.1\% - 54.7\%)$$

Using material-specific Gibson-Ashby scaling:
$$E_{\text{eff}}(\tau) = E_{\text{solid}} \cdot (1.0 - \text{Porosity})^{\gamma}$$
where $\gamma = 1.60$ for Ti-6Al-4V and $\gamma = 1.55$ for 316L Stainless Steel.

---

## 7. Optimization Engine: Warmup-Stable-Decay (WSD)

To prevent premature learning rate decay while ensuring precision target convergence, the optimization uses a **Warmup-Stable-Decay (WSD)** learning rate schedule:

```
Learning Rate (eta)
  0.09 ┼           ┌────────────────────────┐
       │          /                          \
  0.03 ┼─────────/                            \───────► 0.015
       └─────────┴────────────────────────────┴────────► Step
       0        Warmup (3 steps)    Stable (75%)     Precision Decay (25%)
```

- **Warmup Phase (Steps 1–3)**: Ramps $\eta$ from $0.03 \to 0.09$ to build momentum without gradient shocks.
- **Stable Plateau Phase (Steps 4–75% Max)**: Holds at peak $\eta = 0.09$ with fast-adapting Adam ($\beta_1=0.85, \beta_2=0.98, \text{clip}=0.12$).
- **Precision Anneal Phase (Final 25% Steps)**: Cosine-decays to $\eta_{\text{min}} = 0.015$ to lock onto the exact target micro-motion within $\pm 0.005\,\text{mm}$.

---

## 8. Certified Biomaterials Database

| Biomaterial | Elastic Modulus (GPa) | Density (g/cm³) | Yield Strength (MPa) | Fatigue Limit ($10^6$ cycles) | Gibson-Ashby Exponent ($\gamma$) | Clinical Indication |
|---|---|---|---|---|---|---|
| **Ti-6Al-4V ELI (Grade 5 Titanium)** | 110.0 | 4.43 | 880 | 510 | 1.60 | Standard gold-standard trauma plate with superior osseointegration |
| **316L Stainless Steel** | 193.0 | 8.00 | 220 | 200 | 1.55 | Cost-effective, high-ductility clinical standard trauma fixation |

---

## 9. Automated In-Silico Verification Battery

Every generated implant is automatically validated against international standards before clinical signoff:

| Verification Test | Testing Standard | Passing Criteria | Clinical Rationale |
|---|---|---|---|
| **Micro-Motion Target Test** | ASTM F382 / AO Foundation | Within $\pm 15\%$ of target | Verifies optimal strain window for periosteal callus formation |
| **Stress Shielding Mitigation** | Wolff's Law Biomechanical Index | $\ge 55.0\%$ cortical load preservation | Prevents cortical bone resorption and osteopenia |
| **Static 4-Point Bending Proof** | ASTM F382 Static Bending | Safety factor $S_f \ge 1.50$ | Prevents permanent plate plastic deformation under single-leg stance |
| **Cyclic Fatigue Endurance** | ISO 7206 Dynamic Fatigue | $\ge 1.20\text{x}$ Endurance Margin | Guarantees survival over 1,000,000 ambulatory gait cycles |

---

## 10. Marked 3D Coordinate Visualizer

The 3D interactive viewport displays full continuum meshes with marked, scaled coordinate axes in physical millimeters:

- 🔵 **$X$-Axis (`#38bdf8` Cyan)**: **`X: Length / Shaft Axis (mm)`** ($0\,\text{mm} \to 160\,\text{mm}$, with osteotomy gap marked at $X = 80\,\text{mm}$).
- 🟢 **$Y$-Axis (`#4ade80` Green)**: **`Y: Plate Depth (mm)`** ($0\,\text{mm} \to 20\,\text{mm}$, indicating the $6.0\,\text{mm}$ plate sandwich thickness).
- 🟣 **$Z$-Axis (`#a78bfa` Purple)**: **`Z: Width (mm)`** ($-12\,\text{mm} \to +12\,\text{mm}$, indicating the $16\,\text{mm}$ transverse plate width).
- **Dual Visual Modes**: Seamlessly toggle between **3D Von Mises Stress Heatmap (MPa)** and **3D Factor of Safety (FoS) Field** with ASTM proof safety badges.

---

## 11. Installation & Running Locally

### Prerequisites:
- macOS (Apple Silicon / Intel) or Linux (Ubuntu 22.04+)
- Python 3.12 (virtual environment recommended)

### Quick Start:
```bash
# 1. Clone the repository
git clone https://github.com/HibernatingBunny067/Tesseract_Submission_2026.git
cd Tesseract_Submission_2026

# 2. Run the all-in-one launcher script
./run
```

The script will automatically start both Tesseract microservice servers (Ports 8000 & 8001) in the background and launch the Streamlit dashboard at:
`http://localhost:8501`

### Clean Process Shutdown:
To terminate all background workers cleanly, press `Ctrl + C` in the terminal, or run:
```bash
pkill -f "tesseract_server.py" ; pkill -f "streamlit run"
```

---

## 12. Project File Structure

```
Tesseract_Hackathon/
├── app.py                         ◄── Streamlit user interface and 3D visualization dashboard
├── run                            ◄── Auto-cleanup launch script with process traps
├── REQUIREMENTS.txt               ◄── Python package dependencies
├── README.md                      ◄── Complete project specification & verification report
├── logs/
│   ├── clinical_audit.log         ◄── Structured clinical audit trail
│   └── session_history.jsonl      ◄── JSONL convergence trajectory log
├── src/
│   ├── agent/
│   │   ├── agent.py               ◄── Groq LLM + Local Biomechanical NLP parser
│   │   └── optimize.py            ◄── Differentiable WSD Adam optimization engine
│   ├── fem/
│   │   ├── forward.py             ◄── JAX-FEM forward solve & micro-motion calculation
│   │   ├── problem.py             ◄── Constitutive equations, Gaussian blend, SuperLU dispatch
│   │   ├── materials.py           ◄── Orthopaedic biomaterials database (Ti-64 & SS316L)
│   │   ├── validation.py          ◄── ASTM F382 & ISO 7206 in-silico verification suite
│   │   └── data/
│   │       └── model.msh          ◄── Master 10-node tetrahedral mesh (29,571 DOFs)
│   ├── geometry/
│   │   ├── plot_plotly.py         ◄── 3D Marching Cubes, Marked axes, Multi-TPMS STL generator
│   │   └── model.py               ◄── Conforming bone-plate Gmsh generator script
│   ├── ui/
│   │   ├── charts.py              ◄── Interactive Plotly tracking figures with dotted target lines
│   │   ├── components.py          ◄── Glassmorphic HTML UI cards, badges, and comparison table
│   │   └── __init__.py            ◄── Premium dark-mode CSS design system
│   └── utils/
│       └── logger.py              ◄── Dual-stream clinical audit logger
└── tesseracts/
    ├── fem_tesseract/
    │   ├── tesseract_api.py       ◄── Official Tesseract FEM REST endpoints (apply, VJP)
    │   ├── tesseract_server.py    ◄── Uvicorn microservice runner (Port 8000)
    │   ├── tesseract_config.yaml  ◄── Official Tesseract container build manifest
    │   └── tesseract_requirements.txt ◄── Official Tesseract FEM dependencies
    └── geometry_tesseract/
        ├── tesseract_api.py       ◄── Official Tesseract Geometry & Porosity REST endpoints
        ├── tesseract_server.py    ◄── Uvicorn microservice runner (Port 8001)
        ├── tesseract_config.yaml  ◄── Official Tesseract container build manifest
        └── tesseract_requirements.txt ◄── Official Tesseract Geometry dependencies
```