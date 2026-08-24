"""
Biomedical Implant Materials Database.

Provides physical properties, elastic moduli, densities, and clinical notes
for standard orthopaedic fixation materials.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class Biomaterial:
    name: str
    code: str
    youngs_modulus_gpa: float  # GPa
    density_g_cm3: float       # g/cm^3
    yield_strength_mpa: float  # MPa
    poissons_ratio: float
    tpms_ga_exponent: float    # Gibson-Ashby scaling exponent for TPMS
    clinical_note: str
    biocompatibility: str

    @property
    def youngs_modulus_pa(self) -> float:
        return self.youngs_modulus_gpa * 1e9


BIOMATERIALS: Dict[str, Biomaterial] = {
    "Ti-6Al-4V (Grade 5 Titanium)": Biomaterial(
        name="Ti-6Al-4V ELI (Grade 5)",
        code="Ti64",
        youngs_modulus_gpa=110.0,
        density_g_cm3=4.43,
        yield_strength_mpa=880.0,
        poissons_ratio=0.32,
        tpms_ga_exponent=1.60,
        clinical_note="Gold standard for load-bearing trauma plates with superior osseointegration.",
        biocompatibility="ISO 5832-3 / ASTM F136 Certified"
    ),
    "316L Stainless Steel": Biomaterial(
        name="Medical Grade 316L Stainless Steel",
        code="SS316L",
        youngs_modulus_gpa=193.0,
        density_g_cm3=8.00,
        yield_strength_mpa=220.0,
        poissons_ratio=0.30,
        tpms_ga_exponent=1.55,
        clinical_note="Cost-effective classical fixation with high ductility; high stress shielding.",
        biocompatibility="ASTM F138 Certified"
    )
}
