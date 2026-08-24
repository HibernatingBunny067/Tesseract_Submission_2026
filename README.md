# Tesseract Differentiable Biomechanics: Agentic Patient-Specific Implant Optimization

An end-to-end framework that translates natural language surgeon requirements into patient-specific, 3D-printable, functionally graded TPMS orthopaedic implants. Powered by **JAX-FEM**, **Tesseract Core REST Microservices**, and **Groq LPU Inference**.

---

## 1. Executive Summary & Clinical Problem

### The Clinical Dilemma: Rigid Fixation vs. Secondary Healing
In orthopaedic trauma surgery, fixing femur fractures with conventional solid titanium plates presents a major biomechanical paradox:

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
   - Solid titanium (110 GPa) is approximately 6 times stiffer than cortical bone (18 GPa).
   - The metal plate carries almost all ambulatory loads, "shielding" the underlying bone from mechanical stress. Over months, this causes **cortical bone resorption, plate loosening, and refracture**.

### The Solution: 5-Zone Functionally Graded Metamaterial Plates
Our platform designs a continuous Triply Periodic Minimal Surface (TPMS) metamaterial plate:
- **Rigid Solid Ends (Proximal & Distal)**: Thick struts and high metal density to provide solid anchorage for bicortical locking screws.
- **Compliant Porous Bridge**: Tuned microscopic lattice directly over the fracture gap to deliver the exact target micro-motion (e.g., 0.20 mm) while preserving >= 55% of natural cortical load transfer.

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
  │ 3. TESSERACT ADJOINT FEM ENGINE (JAX-FEM + PETSc over HTTP REST)            │
  │    • 5,951 TET10 Quadratic Elements (29,571 DOFs)                           │
  │    • Continuous 5-Zone Gaussian Partition-of-Unity Level Set                │
  │    • Reverse-Mode Adjoint VJP: Exact Gradients dLoss/d(tau) in O(1) time    │
  │    • Adam Optimizer converges in 10 to 15 steps (< 20 seconds)              │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 4. AUTOMATED IN-SILICO VERIFICATION BATTERY                                 │
  │    • ASTM F382: Fracture micro-motion target verification (+/- 15% band)   │
  │    • Wolff's Law Index: Cortical stress shielding preservation >= 55.0%     │
  │    • ASTM F382 Static Proof Test: Yield Safety Factor Sf >= 1.50            │
  │    • ISO 7206: Cyclic fatigue endurance margin at 10^6 gait cycles          │
  └──────────────────────────────────────┬──────────────────────────────────────┘
                                         │
                                         ▼
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 5. CAD / CAM 3D SLICER EXPORT & FULL CONTINUUM VISUALIZATION                │
  │    • Full 3D Von Mises Stress Heatmap (in MPa) across Bone and Plate        │
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
                        │   Implant Plate: 100 mm x 16 mm x 7 mm    │
                        └───────────────────────────────────────────┘
```

### Detailed Component Dimensions:
1. **Femoral Diaphysis Segment**:
   - Total Length: **160.0 mm** (16.0 cm).
   - Outer Cortical Diameter: **24.0 mm** (Radius = 12.0 mm).
   - Inner Marrow Diameter: **16.0 mm** (Radius = 8.0 mm).
   - Cortical Wall Thickness: **4.0 mm** of solid cortical bone.
   - Fracture Gap Width: **2.0 mm** (x = 79.0 mm to 81.0 mm).
2. **Implant Fixation Plate & Sandwich Architecture**:
   - Total Length: **100.0 mm** (10.0 cm, standard 6 to 8-hole anatomical locking plate spanning x = 30.0 mm to 130.0 mm).
   - Plate Width: **16.0 mm** (z = -8.0 mm to +8.0 mm).
   - Plate Thickness: **7.0 mm** (y = 11.0 mm to 18.0 mm).
   - **Solid Cortex Skin Shell (1.0 mm)**: Surrounding outer top, bottom, and lateral surfaces are maintained as 100% solid Ti-6Al-4V to prevent muscle/soft tissue adhesion and ensure smooth cortical bone seating.
   - **Internal TPMS Porous Core**: Houses the continuous minimal surface lattice directly inside the sandwich bridge over the fracture gap (x = 70.0 mm to 90.0 mm).
   - **6-Hole AO LCP Fixation Screws**: Six 4.5 mm cylindrical through-holes (radius = 2.25 mm) positioned at x = [35.0, 50.0, 65.0, 95.0, 110.0, 125.0] mm. These holes accurately account for cross-sectional area reduction and Kirsch stress concentrations (Kt ≈ 2.5x) in the FEA continuum.
   - Metamaterial Unit Cell Size: **5.0 mm** (provides 20 unit cells along the plate length).
3. **Finite Element Discretization**:
   - Total Elements: **5,951 cells** (quadratic 10-node tetrahedrons / TET10).
   - Total Degrees of Freedom: **9,857 nodes x 3 = 29,571 DOFs**.
   - Mesh Conformance: Shared nodes at bone-plate interface representing fixed-angle bicortical locking screws (Locking Compression Plate / LCP principle).
4. **Boundary Conditions & Joint Loading**:
   - Proximal End (Hip): Rigidly clamped (cantilever fixity).
   - Distal End (Knee): Subjected to vertical downward ground reaction traction (-1.0 MPa, simulating heel-strike during walking gait).

---

## 4. Sandwich Metamaterial & Screw Hole Architecture

```
                      CLINICAL SANDWICH PLATE ARCHITECTURE
                      
    ┌────────────────────────────────────────────────────────────────────────┐  ◄── Top Solid Skin (1.0 mm)
    │  ( O )    ( O )    ( O )  │   TPMS INTERNAL CORE   │  ( O )    ( O )   │      (Soft-tissue protection)
    │ 35 mm    50 mm    65 mm   │(Tuned 0.20 mm Motion!) │  95 mm   110 mm   │  ◄── 6x ∅4.5 mm AO Screw Holes
    └───────────────────────────┴────────────────────────┴───────────────────┘  ◄── Bottom Solid Skin (1.0 mm)
      ◄── Solid Proximal End ──► ◄── Porous Bridge Core ──► ◄── Solid Distal End ──►
```

### Key Engineering Features:
1. **Clinical Soft-Tissue Safety**:
   - Exposed lattice struts can cause tendon adhesion, inflammation, and bursitis. The **1.0 mm solid outer skin** provides a continuous, polished boundary envelope.
2. **Notch-Free Bridge**:
   - Traditional plates left empty middle screw holes over the gap to create flexibility, but the sharp circular holes caused fatigue crack failure ($K_t \approx 2.5$).
   - Our design replaces empty bridge holes with a **continuous 3D TPMS core**, distributing bending stresses smoothly across thousands of microscopic struts without sharp geometric stress notches.
3. **Seamless Continuum Representation**:
   - Both the solid skin shell and the 6 screw holes are embedded directly into the differentiable JAX-FEM density formulation via smooth sigmoid spatial operators, preserving exact $O(1)$ adjoint gradient backpropagation.

---

## 5. Multi-TPMS Metamaterial Formulations

The implant lattice is governed by Triply Periodic Minimal Surface (TPMS) level sets `F(x,y,z) = tau(x)`:

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

### Mathematical Level-Set Equations (k = 2 * pi / cell_size):
1. **Schwarz Primitive (P)** *(High Vascularization & Fluid Permeability)*:
   `F_P(x,y,z) = cos(kx) + cos(ky) + cos(kz)`
2. **Schoen Gyroid (G)** *(Superior Shear Strength & Energy Absorption)*:
   `F_G(x,y,z) = 1.5 * (sin(kx)*cos(ky) + sin(ky)*cos(kz) + sin(kz)*cos(kx))`
3. **Schwarz Diamond (D)** *(Maximum Torsional Stiffness & Multi-Axial Rigidity)*:
   `F_D(x,y,z) = 1.8 * (cos(kx)*cos(ky)*cos(kz) - sin(kx)*sin(ky)*sin(kz))`

### Continuous 5-Zone Partition of Unity:
The local density threshold `tau(x)` is blended continuously across 5 anatomical zones:
`tau(x) = [ tau_1*w_1(x) + tau_2*w_2(x) + tau_3*w_3(x) + tau_4*w_4(x) + tau_5*w_5(x) ] / SUM(w_i(x))`
- **Zone 1 (x = 35 mm)**: Far Proximal Screw Anchor (Dense metal).
- **Zone 2 (x = 57 mm)**: Proximal Stress Transition (Graded compliance).
- **Zone 3 (x = 80 mm)**: Fracture Bridge Gap Center (Porous & compliant).
- **Zone 4 (x = 103 mm)**: Distal Stress Transition (Graded compliance).
- **Zone 5 (x = 125 mm)**: Far Distal Screw Anchor (Dense metal).
- **Gaussian Kernel**: `w_i(x) = exp( - ((x - x_i) / sigma_blend)^2 )` where `sigma_blend` is optimized dynamically between 10 mm and 28 mm.

---

## 6. Differentiable Physics & Loss Function

The multi-objective loss function balances three competing clinical requirements:

```
Total Loss = (Micro-Motion Penalty) + (Foreign Mass Penalty) + (Structural Compliance Term)
```

### 1. Target Micro-Motion Tracking Term:
`diff_mm = (achieved_displacement - target_displacement) * 1000`
- If motion > target (too loose -> instability risk): `Penalty = (diff_mm ^ 2) * 22,000`
- If motion < target (too stiff -> slow healing): `Penalty = (diff_mm ^ 2) * 2,500`

### 2. Foreign Implant Mass Minimization:
`Mass Penalty = w_mass * SUM(1.50 - tau_i)` for all 5 zones.
- Pushes `tau` toward its safe upper limit (1.45), opening up bone marrow channels and reducing implant weight by 20% to 35%.

### 3. Structural Compliance (Strain Energy):
`Compliance = Integral of (Traction * Surface_Displacement) dA`
- Included in rigid / high-impact scenarios (`3.0 * Compliance`) to prevent gross plate buckling under peak ambulatory loads.

### 4. Zero-Failure Mode Mathematical Safeguards:
- **Strict Invertibility**: `tau` is hard-clamped to `[0.10, 1.45]`, guaranteeing minimum Young's modulus `E >= 1.0 MPa` (`det(K) > 0`, zero singular matrix errors).
- **Nyquist Safety**: Lattice unit cell (5.0 mm) is strictly larger than element size (3.0 mm), preventing spatial aliasing.

---

## 7. Certified Biomaterials Database

| Biomaterial | Elastic Modulus (GPa) | Density (g/cm³) | Yield Strength (MPa) | Fatigue Limit (10^6 cycles) | Clinical Indication |
|---|---|---|---|---|---|
| **Ti-6Al-4V ELI (Grade 5 Titanium)** | 110.0 | 4.43 | 880 | 510 | Standard gold-standard trauma plate with superior osseointegration |
| **316L Stainless Steel** | 193.0 | 8.00 | 220 | 200 | Cost-effective, high-ductility clinical standard trauma fixation |

---

## 8. Automated In-Silico Verification Battery

Every generated implant is automatically validated against international standards before clinical signoff:

| Verification Test | Testing Standard | Passing Criteria | Clinical Rationale |
|---|---|---|---|
| **Micro-Motion Target Test** | ASTM F382 / AO Foundation | Within +/- 15% of target | Verifies optimal strain window for periosteal callus formation |
| **Stress Shielding Mitigation** | Wolff's Law Biomechanical Index | >= 55.0% cortical load preservation | Prevents cortical bone resorption and osteopenia |
| **Static 4-Point Bending Proof** | ASTM F382 Static Bending | Safety factor Sf >= 1.50 | Prevents permanent plate plastic deformation under single-leg stance |
| **Cyclic Fatigue Endurance** | ISO 7206 Dynamic Fatigue | >= 1.20x Endurance Margin | Guarantees survival over 1,000,000 ambulatory gait cycles |

### Factor of Safety (FoS) Field Inspection:
The dashboard provides a dedicated **3D Factor of Safety contour viewer**:
```
Nodal Factor of Safety (FoS) = Material_Yield_Strength (MPa) / Local_Von_Mises_Stress (MPa)
```
- **Green (> 2.0x)**: Safe elastic regime.
- **Yellow (1.0x - 2.0x)**: Elevated stress transition zone.
- **Red (< 1.0x)**: Plastic yield risk (ASTM F382 non-compliant).
- The construct-wide minimum Factor of Safety badge is displayed directly above the 3D viewport.

---

## 9. Clinical Audit Trail & Data Logging

Every surgeon prompt, LLM reasoning trace, optimization convergence step, and in-silico test report is automatically recorded into persistent audit logs:

- **`logs/clinical_audit.log`**: Human-readable ASCII clinical summary report for hospital recordkeeping.
- **`logs/session_history.jsonl`**: Machine-readable JSON-Lines dataset capturing full convergence trajectories and parameter states.

---

## 10. Installation & Running Locally

### Prerequisites:
- macOS (Apple Silicon / Intel) or Linux (Ubuntu 22.04+)
- Python 3.12 (virtual environment recommended)

### Quick Start:
```bash
# 1. Clone the repository
git clone https://github.com/your-repo/Tesseract_Hackathon.git
cd Tesseract_Hackathon

# 2. Run the all-in-one launcher script
./run
```

The app will launch the Tesseract REST server in the background and open the dashboard at:
`http://localhost:8501`

### Clean Process Shutdown:
To terminate all background workers (Streamlit and Tesseract server) cleanly, press:
`Ctrl + C` in the terminal, or run:
```bash
pkill -f "tesseract_server.py" ; pkill -f "streamlit run"
```

---

## 11. Project File Structure

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
│   │   └── optimize.py            ◄── Differentiable Adam optimization engine (5 zones + sigma)
│   ├── fem/
│   │   ├── forward.py             ◄── JAX-FEM forward solve & micro-motion calculation
│   │   ├── problem.py             ◄── Constitutive equations, Gaussian blend, Von Mises stress
│   │   ├── materials.py           ◄── Orthopaedic biomaterials database
│   │   ├── validation.py          ◄── ASTM F382 & ISO 7206 in-silico verification suite
│   │   └── data/
│   │       └── model.msh          ◄── Master 10-node tetrahedral mesh (29,571 DOFs)
│   ├── geometry/
│   │   ├── plot_plotly.py         ◄── 3D Marching Cubes, Multi-TPMS STL generator, Plotly scenes
│   │   └── model.py               ◄── Conforming bone-plate Gmsh generator script
│   ├── ui/
│   │   ├── charts.py              ◄── Interactive Plotly tracking figures with dotted target lines
│   │   ├── components.py          ◄── Glassmorphic HTML UI cards, badges, and comparison table
│   │   └── __init__.py            ◄── Premium dark-mode CSS design system
│   └── utils/
│       └── logger.py              ◄── Dual-stream clinical audit logger
└── tesseracts/
    └── fem_tesseract/
        ├── tesseract_api.py       ◄── Official Tesseract REST endpoints (apply, VJP, abstract_eval)
        ├── tesseract_server.py    ◄── Uvicorn microservice runner (Port 8000)
        ├── tesseract_config.yaml  ◄── Official Tesseract container build manifest
        └── tesseract_requirements.txt ◄── Official Tesseract dependencies
```