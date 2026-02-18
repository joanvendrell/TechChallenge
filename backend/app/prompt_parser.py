import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import spacy
from spacy.matcher import PhraseMatcher

from pint import UnitRegistry

from .pde_spec import PDESpec, GeometrySpec, MaterialSpec, MeshSpec, BCSpeс 
from copy import deepcopy
from typing import Any, Dict



# =======================
# NLP + Units setup
# =======================

# Load once at import time (fast enough for MVP; in prod load at app startup)
# Install: python -m spacy download en_core_web_sm
_NLP = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # we mostly need tokenization + matcher

_ureg = UnitRegistry()
_Q_ = _ureg.Quantity

# Add a few common aliases for robustness
_ureg.define("um = micrometer = micrometre")
_ureg.define("nm = nanometer = nanometre")
_ureg.define("kN = kilonewton")
_ureg.define("K = kelvin")

# =======================
# Domain/material vocab
# =======================

DOMAIN_KEYWORDS = {
    "heat_transfer": ["heat", "thermal", "temperature", "conduction", "diffusion"],
    "solid_mechanics": ["stress", "strain", "deformation", "elastic", "cantilever", "beam", "young", "poisson"],
    "fluid": ["fluid", "navier", "stokes", "flow", "velocity", "pressure"],
}

MATERIALS = [
    "copper",
    "aluminum",
    "aluminium",
    "silicon",
    "steel",
    "gold",
    "silver",
]

GEOM_KEYWORDS = {
    "thin_film": ["thin film", "film", "thin layer", "membrane"],
    "beam": ["beam", "cantilever"],
    "box": ["block", "box", "slab"],
    "sphere": ["sphere", "ball"],
}

# Phrase matcher for materials + some geometry hints
_material_matcher = PhraseMatcher(_NLP.vocab, attr="LOWER")
_material_matcher.add("MATERIAL", [ _NLP.make_doc(m) for m in MATERIALS ])

_geom_matcher = PhraseMatcher(_NLP.vocab, attr="LOWER")
for k, phrases in GEOM_KEYWORDS.items():
    _geom_matcher.add(f"GEOM_{k.upper()}", [_NLP.make_doc(p) for p in phrases])

# =======================
# Data model for extraction
# =======================

@dataclass
class PromptEntities:
    domain: str
    material: Optional[str] = None

    # geometry
    thickness_m: Optional[float] = None
    lengths_m: List[float] = field(default_factory=list)  # all length mentions in meters
    geom_hint: Optional[str] = None  # "thin_film", "beam", "box", ...

    # boundary conditions / loads
    temperatures_K: List[float] = field(default_factory=list)
    force_N: Optional[float] = None

    # raw + debug
    tokens: List[str] = field(default_factory=list)
    matches: Dict[str, Any] = field(default_factory=dict)


# =======================
# Unit extraction helpers
# =======================

# This regex extracts number + unit strings like:
# "30 nm", "1e-6 m", "400K", "5 N", "2kN"
# Then pint normalizes them.
_QUANTITY_RE = re.compile(
    r"([+-]?\d+(?:\.\d+)?(?:e[+-]?\d+)?)\s*([a-zA-Zµ]+)\b"
)

def _parse_quantities(text: str):
    out = []
    for m in _QUANTITY_RE.finditer(text):
        val_str, unit_str = m.group(1), m.group(2)
        raw = m.group(0)  # e.g. "5 nm"
        unit_norm = unit_str.replace("µ", "u")

        try:
            q = _Q_(float(val_str), unit_norm)
        except Exception:
            continue

        try:
            if q.check("[length]"):
                out.append((q.to("m").magnitude, "length", raw, unit_str))
            elif q.check("[force]"):
                out.append((q.to("N").magnitude, "force", raw, unit_str))
            elif str(q.units).lower() in ("kelvin", "k"):
                out.append((q.to("K").magnitude, "temperature", raw, unit_str))
        except Exception:
            continue

    return out



# =======================
# Domain inference (spaCy tokens)
# =======================

def _infer_domain(prompt: str) -> str:
    doc = _NLP(prompt.lower())
    text = " ".join([t.text for t in doc])
    for domain, keys in DOMAIN_KEYWORDS.items():
        if any(k in text for k in keys):
            return domain
    return "other"


def _infer_material(doc) -> Optional[str]:
    matches = _material_matcher(doc)
    if not matches:
        return None
    # pick first matched material span
    _, start, end = matches[0]
    return doc[start:end].text.lower()


def _infer_geom_hint(doc) -> Optional[str]:
    matches = _geom_matcher(doc)
    if not matches:
        return None
    # pick the “most specific” by priority (thin_film > beam > box)
    # you can tune this
    priorities = priorities = {
                                "GEOM_THIN_FILM": 3,
                                "GEOM_BEAM": 2,
                                "GEOM_SPHERE": 2,   # same level as beam (tune if you want)
                                "GEOM_BOX": 1
                            }
    best = None
    best_p = -1
    for match_id, start, end in matches:
        label = doc.vocab.strings[match_id]
        p = priorities.get(label, 0)
        if p > best_p:
            best_p = p
            best = label
    if best == "GEOM_THIN_FILM":
        return "thin_film"
    if best == "GEOM_BEAM":
        return "beam"
    if best == "GEOM_BOX":
        return "box"
    if best == "GEOM_SPHERE":
        return "sphere"
    return None


# =======================
# Extraction
# =======================

def extract_entities(prompt: str, domain_hint: Optional[str] = None) -> PromptEntities:
    doc = _NLP(prompt)

    domain = domain_hint or _infer_domain(prompt)
    ent = PromptEntities(domain=domain)
    ent.tokens = [t.text for t in doc]

    ent.material = _infer_material(doc)
    ent.geom_hint = _infer_geom_hint(doc)

    quantities = _parse_quantities(prompt)
    lengths = [v for (v, qtype, raw, unit) in quantities if qtype == "length"]
    temps   = [v for (v, qtype, raw, unit) in quantities if qtype == "temperature"]
    forces  = [v for (v, qtype, raw, unit) in quantities if qtype == "force"]

    ent.lengths_m = sorted(lengths)
    ent.temperatures_K = temps

    # thickness heuristic: smallest mentioned length
    if lengths:
        ent.thickness_m = min(lengths)

    # force: take first mention (you can improve later)
    if forces:
        ent.force_N = forces[0]

    ent.matches = {
        "quantities": quantities,
        "material": ent.material,
        "geom_hint": ent.geom_hint,
    }
    print(ent.matches["quantities"])
    return ent


# =======================
# Template builders (entities -> PDESpec)
# =======================

def _heat_from_entities(ent: PromptEntities) -> PDESpec:
    extracted: Dict[str, Any] = {"entities": ent.matches}
    warnings: List[str] = []
    questions: List[str] = []

    # thickness is usually critical for nanoscale conduction
    if ent.thickness_m is None:
        questions.append("What is the thickness/characteristic length (e.g., 30 nm)?")

    material_name = ent.material
    mat_props: Dict[str, Any] = {}

    # Material properties (placeholders)
    if material_name == "copper":
        mat_props["k"] = 400.0
    elif material_name in ("aluminum", "aluminium"):
        mat_props["k"] = 205.0
    elif material_name == "silicon":
        mat_props["k"] = 148.0
    elif material_name:
        warnings.append(f"Material '{material_name}' recognized but no default properties set; using k=1.0 W/mK.")
        mat_props["k"] = 1.0
    else:
        # In a “chatty” flow, this should be a question (critical), not a silent default
        questions.append("What material is it (e.g., copper, silicon, steel)?")

    # geometry type
    #geom_type = ent.geom_hint or ("thin_film" if ent.thickness_m and ent.thickness_m < 1e-6 else "box")
    # -------------------------
    # Dimensions (map extracted lengths)
    # -------------------------
    #dims: Dict[str, float] = {"Lx": 1e-6, "Ly": 1e-6, "Lz": 1e-9}
    geom_type = ent.geom_hint or "beam"
    if geom_type == "sphere":
        # interpret the smallest mentioned length as radius if available; else default
        r = (ent.thickness_m or (min(ent.lengths_m) if ent.lengths_m else None) or 0.05)
        dims = {"R": float(r)}
        warnings.append(f"Assuming sphere geometry with radius R={dims['R']} m (override recommended).")
    else:
        # default beam
        dims = {"L": 0.1, "W": 0.01, "H": 0.01}
        warnings.append("Assuming beam geometry L=0.1m, W=0.01m, H=0.01m (override recommended).")


    Ls = sorted(ent.lengths_m) if getattr(ent, "lengths_m", None) else []

    # Heuristic: if user gave 3+ lengths, treat them as box dims
    # largest -> Lx, middle -> Ly, smallest -> Lz (thickness)
    if len(Ls) >= 3:
        dims["Lz"] = float(Ls[0])
        dims["Ly"] = float(Ls[-2])
        dims["Lx"] = float(Ls[-1])
        extracted["mapped_lengths_m"] = Ls
        extracted["thickness_m"] = float(Ls[0])
        warnings.append("Mapped 3+ length values to (Lx, Ly, Lz) = (largest, middle, smallest).")

    # If exactly 2 lengths, assume (in-plane length, thickness)
    elif len(Ls) == 2:
        dims["Lz"] = float(Ls[0])
        dims["Lx"] = float(Ls[1])
        # keep Ly default but warn
        extracted["mapped_lengths_m"] = Ls
        extracted["thickness_m"] = float(Ls[0])
        warnings.append("Mapped 2 length values to (Lx, Lz) = (largest, smallest); using default Ly=1e-6 m.")

    # If only 1 length, only treat as thickness if prompt suggests it
    elif len(Ls) == 1:
        p = " ".join(ent.tokens).lower() if getattr(ent, "tokens", None) else ""
        thickness_words = ["thickness", "thick", "thin", "film", "layer", "membrane"]

        if any(w in p for w in thickness_words):
            dims["Lz"] = float(Ls[0])
            extracted["thickness_m"] = float(Ls[0])
            warnings.append("Single length value treated as thickness (based on prompt keywords). Using default Lx=Ly=1e-6 m.")
        else:
            # Don't guess: ask
            questions.append("You provided one length. Is it the thickness (Lz) or an in-plane dimension (Lx/Ly)?")
            warnings.append("Keeping default geometry until thickness/in-plane dimension is clarified.")

    else:
        dims = {"Lx": 5, "Ly": 5, "Lz": 1}
        warnings.append("Assuming default geometry 1 x 5 x 5 m (override recommended).")

    # Boundary conditions: if 2 temperatures are provided, use them; else ask
    if len(ent.temperatures_K) >= 2:
        T_left, T_right = ent.temperatures_K[0], ent.temperatures_K[1]
        bc = {"type": "dirichlet_lr", "T_left": float(T_left), "T_right": float(T_right)}
    else:
        questions.append("What boundary temperatures should I use (e.g., 400 K on left and 300 K on right)?")
        bc = {"type": "dirichlet_lr", "T_left": None, "T_right": None}

    # mesh: safe default (quality-related)
    mesh = MeshSpec(resolution="auto", params={"nx": 40, "ny": 40, "nz": 2})
    if ent.thickness_m is not None and ent.thickness_m < 1e-7:
        mesh.params["nz"] = 2
        warnings.append("Thin dimension detected; using nz=2 (increase if you need through-thickness gradients).")

    return PDESpec(
        domain="heat_transfer",
        pde="steady_heat",
        geometry=GeometrySpec(type=geom_type, dims=dims),
        material=MaterialSpec(name=material_name, props=mat_props),
        bc=BCSpeс(kind=bc),
        mesh=mesh,
        outputs=["temperature"],
        units="SI",
        extracted=extracted,
        warnings=warnings,
        clarifying_questions=questions,
    )


def _solid_from_entities(ent: PromptEntities) -> PDESpec:
    extracted: Dict[str, Any] = {"entities": ent.matches}
    warnings: List[str] = []
    questions: List[str] = []

    # Force/load is critical
    if ent.force_N is None:
        questions.append("What load should be applied (e.g., 5 N tip load)?")

    # geometry hint
    geom_type = "beam" if (ent.geom_hint == "beam") else "beam"
    dims = {"L": 0.1, "W": 0.01, "H": 0.01}
    warnings.append("Assuming beam geometry L=0.1m, W=0.01m, H=0.01m (override recommended).")

    material_name = ent.material
    mat_props: Dict[str, Any] = {}

    if material_name == "steel":
        mat_props.update({"E": 200e9, "nu": 0.30})
    elif material_name:
        warnings.append(f"Material '{material_name}' recognized but no default elastic properties set.")
        questions.append("What are the material properties (Young’s modulus E and Poisson ratio nu), or should I assume E=1e9 Pa, nu=0.30?")
        # keep placeholders if user insists on defaults
        mat_props.update({"E": 1e9, "nu": 0.30})
    else:
        questions.append("What material is the beam (e.g., steel, aluminum)?")

    bc = {
        "type": "cantilever_tip_load",
        "fixed_end": "x=0",
        "load_end": "x=L",
        "F": float(ent.force_N) if ent.force_N is not None else None,
        "direction": "down",
    }

    mesh = MeshSpec(resolution="auto", params={"nx": 60, "ny": 10, "nz": 10})

    return PDESpec(
        domain="solid_mechanics",
        pde="linear_elasticity",
        geometry=GeometrySpec(type=geom_type, dims=dims),
        material=MaterialSpec(name=material_name, props=mat_props),
        bc=BCSpeс(kind=bc),
        mesh=mesh,
        outputs=["displacement", "von_mises"],
        units="SI",
        extracted=extracted,
        warnings=warnings,
        clarifying_questions=questions,
    )

def _fluid_from_entities(ent: PromptEntities) -> PDESpec:
    extracted = {"entities": ent.matches}
    warnings, questions = [], []

    # Geometry defaults (channel)
    # Use extracted Lx/Ly/Lz if your entities provide them; otherwise keep safe defaults.
    Lx = getattr(ent, "Lx_m", None) or 1.0
    Ly = getattr(ent, "Ly_m", None) or 0.2
    Lz = getattr(ent, "Lz_m", None) or 0.2

    # Inlet velocity
    Uin = getattr(ent, "inlet_velocity_mps", None)
    if Uin is None:
        questions.append("What inlet velocity should I use (e.g., 0.2 m/s)?")
        Uin = 0.2
        warnings.append("No inlet velocity found; using default 0.2 m/s.")

    # Viscosity (dynamic mu) in Pa*s
    mu = getattr(ent, "mu_Pas", None)
    if mu is None:
        mu = 1e-3  # water-ish at room temp
        warnings.append("No viscosity provided; using mu=1e-3 Pa·s (water-like).")

    geom = GeometrySpec(type="box", dims={"Lx": float(Lx), "Ly": float(Ly), "Lz": float(Lz)})

    mat = MaterialSpec(name="fluid", props={"mu": float(mu)})

    # BCs for channel:
    # - inlet x=0: velocity = (Uin, 0, 0)
    # - walls y=0,y=Ly and z=0,z=Lz: no-slip
    # - outlet x=Lx: pressure = 0
    bc = BCSpeс(kind={
        "type": "channel_inlet_outlet",
        "U_in": float(Uin),
        "p_out": 0.0
    })

    mesh = MeshSpec(resolution="auto", params={"nx": 40, "ny": 16, "nz": 16})

    return PDESpec(
        domain="fluid",
        pde="steady_stokes",
        geometry=geom,
        material=mat,
        bc=bc,
        mesh=mesh,
        outputs=["velocity_mag", "pressure"],
        units="SI",
        extracted=extracted,
        warnings=warnings,
        clarifying_questions=questions,
    )


# =======================
# Main entry point
# =======================
def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_update(dst[k], v)
        else:
            dst[k] = v
    return dst

def _spec_from_parameters(parameters: dict) -> Optional[PDESpec]:
    # parameters might be a full PDESpec dict (domain/pde/geometry/...)
    # or might be a smaller overrides dict. Detect by keys.
    if not isinstance(parameters, dict):
        return None
    if "domain" in parameters and "geometry" in parameters and "material" in parameters:
        # looks like a full PDESpec dict
        return PDESpec.model_validate(parameters)
    return None

def _patch_from_entities(ent) -> Dict[str, Any]:
    patch: Dict[str, Any] = {}

    # -----------------
    # material
    # -----------------
    if getattr(ent, "material", None):
        patch.setdefault("material", {})["name"] = ent.material

    # -----------------
    # geometry
    # -----------------
    geom_hint = getattr(ent, "geom_hint", None)

    # If we detected a geometry hint, propagate it
    if geom_hint:
        patch.setdefault("geometry", {})["type"] = geom_hint

    dims_patch: Dict[str, Any] = {}

    if geom_hint == "sphere":
        # For sphere: use radius R, never set Lx/Ly/Lz
        # Prefer an explicit radius if you later add ent.radius_m; for now use largest length mentioned
        if getattr(ent, "lengths_m", None):
            dims_patch["R"] = float(max(ent.lengths_m))
        elif getattr(ent, "thickness_m", None) is not None:
            # fallback: if only thickness_m exists, treat it as R (not ideal, but better than Lz)
            dims_patch["R"] = float(ent.thickness_m)

    else:
        # Non-sphere geometries keep your prior behavior
        if getattr(ent, "Lx_m", None) is not None:
            dims_patch["Lx"] = float(ent.Lx_m)
        if getattr(ent, "Ly_m", None) is not None:
            dims_patch["Ly"] = float(ent.Ly_m)
        if getattr(ent, "thickness_m", None) is not None:
            dims_patch["Lz"] = float(ent.thickness_m)

    if dims_patch:
        patch.setdefault("geometry", {}).setdefault("dims", {}).update(dims_patch)

    # -----------------
    # boundary conditions (heat)
    # -----------------
    if ent.domain == "heat_transfer" and getattr(ent, "temperatures_K", None):
        temps = ent.temperatures_K
        bc_patch: Dict[str, Any] = {}
        if len(temps) >= 1:
            bc_patch["T_left"] = float(temps[0])
        if len(temps) >= 2:
            bc_patch["T_right"] = float(temps[1])

        if bc_patch:
            # Your PDESpec uses spec.bc.kind (a dict). The merge happens later via base_spec.bc.kind.update(...)
            patch.setdefault("bc", {}).update(bc_patch)

    return patch


from typing import Optional

def parse_prompt_to_spec(prompt: str,
                         domain_hint: Optional[str] = None,
                         parameters: Optional[dict] = None,
                        ) -> PDESpec:
    """
    If parameters contains a previous PDESpec dict:
      - start from that previous spec (base)
      - apply only changes found in the new prompt (patch)
    Otherwise:
      - create a fresh spec from templates as before
    """

    ent = extract_entities(prompt, domain_hint=domain_hint)
    domain = ent.domain

    # 1) If we have a previous full spec, start from it
    base_spec = _spec_from_parameters(parameters) if parameters else None
    if base_spec is not None and domain == base_spec.domain:
        # Apply patch extracted from prompt
        patch = _patch_from_entities(ent)

        # material
        if "material" in patch:
            if "name" in patch["material"]:
                base_spec.material.name = patch["material"]["name"]

        # geometry
        if "geometry" in patch:
            # update geometry type if present (e.g., sphere/box/beam)
            gtype = patch["geometry"].get("type", None)
            if gtype is not None:
                base_spec.geometry.type = gtype

            # update geometry dims
            gd = patch["geometry"].get("dims", {})
            if gd:
                base_spec.geometry.dims.update(gd)

        # bc
        if "bc" in patch:
            base_spec.bc.kind.update(patch["bc"])

        # domain override only if user explicitly changed domain
        if domain_hint and domain_hint != base_spec.domain:
            base_spec.domain = domain_hint

        base_spec.extracted = base_spec.extracted or {}
        base_spec.extracted["entities"] = ent.matches
        base_spec.warnings.append("Reused previous setup; applied updates from follow-up prompt.")

        return base_spec

    # 2) No previous spec → old behavior (fresh spec)
    if domain == "heat_transfer":
        spec = _heat_from_entities(ent)
    elif domain == "solid_mechanics":
        spec = _solid_from_entities(ent)
    elif domain == "fluid":
        spec = _fluid_from_entities(ent)
        spec.domain = "fluid"
        spec.pde = "steady_stokes"
    else:
        inferred = _infer_domain(prompt)
        if inferred == "solid_mechanics":
            spec = _solid_from_entities(ent)
        elif inferred == "fluid":
            spec = _fluid_from_entities(ent)
            spec.domain = "fluid"
            spec.pde = "steady_stokes"
        else:
            spec = _heat_from_entities(ent)
            spec.domain = "other"
            spec.warnings.append("Domain unclear; defaulting to heat template. Please specify domain.")

    # 3) If parameters is NOT a full spec dict but is an overrides dict, keep your old merge
    if parameters and base_spec is None:
        if "geometry" in parameters:
            spec.geometry.dims.update(parameters["geometry"].get("dims", {}))
            if "type" in parameters["geometry"]:
                spec.geometry.type = parameters["geometry"]["type"]

        if "material" in parameters:
            if "name" in parameters["material"]:
                spec.material.name = parameters["material"]["name"]
            spec.material.props.update(parameters["material"].get("props", {}))

        if "bc" in parameters:
            spec.bc.kind.update(parameters["bc"])

        if "mesh" in parameters and "params" in parameters["mesh"]:
            spec.mesh.params.update(parameters["mesh"]["params"])

        if "outputs" in parameters and isinstance(parameters["outputs"], list):
            spec.outputs = parameters["outputs"]

    return spec
