"""The material specification both models share, from the one photo reading (services/furniture_spec.py).

Each structural component of the reading gets one entry: its material class, colour, finish and, for woods, the
built-in wood map that fits it (frontend public/woods). The visual model keeps its own photographed textures and PBR
maps untouched; this spec is what the structural model is coloured with, and what links a structural part to the
visual one (same component id). Every entry is an estimate read from a photo, never a measured finish.
"""

from __future__ import annotations

import re
from typing import Any

CLASSES = ("solid_wood", "wood_veneer", "painted_wood", "mdf", "laminate", "glass", "mirror", "metal", "fabric", "leather", "stone", "plastic", "other")

# glTF PBR per class: roughness, metallic, alpha (glass), transmission (glass), sheen (fabric).
PRESETS: dict[str, dict[str, float]] = {
    "solid_wood": {"roughness": 0.55, "metallic": 0.0},
    "wood_veneer": {"roughness": 0.5, "metallic": 0.0},
    "painted_wood": {"roughness": 0.45, "metallic": 0.0},
    "mdf": {"roughness": 0.6, "metallic": 0.0},
    "laminate": {"roughness": 0.35, "metallic": 0.0},
    "glass": {"roughness": 0.05, "metallic": 0.0, "alpha": 0.35, "transmission": 0.9},
    "mirror": {"roughness": 0.03, "metallic": 1.0},
    "metal": {"roughness": 0.35, "metallic": 1.0},
    "fabric": {"roughness": 0.9, "metallic": 0.0, "sheen": 0.6},
    "leather": {"roughness": 0.55, "metallic": 0.0},
    "stone": {"roughness": 0.4, "metallic": 0.0},
    "plastic": {"roughness": 0.4, "metallic": 0.0},
    "other": {"roughness": 0.6, "metallic": 0.0},
}
FINISH_ROUGHNESS = {"gloss": 0.15, "glossy": 0.15, "lacquer": 0.2, "polished": 0.1, "satin": 0.35, "matte": 0.75, "brushed": 0.45}
DEFAULT_COLOR = {"glass": "#DCE8EC", "mirror": "#E8ECEF", "metal": "#A8A9AD"}

# Fallback when the reading gave no class: from the material and texture words (English and Arabic).
_WORDS: tuple[tuple[str, str], ...] = (
    ("mirror", r"mirror|مرآة|مراية"),
    ("glass", r"glass|زجاج"),
    ("metal", r"metal|steel|iron|brass|gold|chrome|alumin|معدن|حديد|ستانلس|نحاس"),
    ("leather", r"leather|جلد"),
    ("fabric", r"fabric|upholster|velvet|linen|boucle|cotton|textile|قماش|مخمل|كتان|تنجيد"),
    ("stone", r"marble|stone|granite|رخام|حجر"),
    ("laminate", r"laminate|melamine|formica|فورمايكا|ميلامين"),
    ("mdf", r"\bmdf\b|\bhdf\b|particle|chipboard|ام دي اف|خشب مضغوط"),
    ("painted_wood", r"paint|lacquer|دهان|ملون|لاكيه"),
    ("wood_veneer", r"veneer|قشرة"),
    ("solid_wood", r"wood|oak|walnut|pine|beech|ash|teak|mahogany|خشب|زان|بلوط|جوز"),
    ("plastic", r"plastic|acrylic|بلاستيك|أكريليك"),
)
# Built-in wood maps (frontend public/woods/<name>.jpg).
SPECIES = ("oak", "walnut", "beech", "ash", "pine", "teak", "mahogany")
_ENGINEERED = {"mdf": "mdf", "laminate": "laminate", "wood_veneer": "veneer"}


def material_class(material: str = "", texture: str = "", given: str | None = None) -> str:
    if given in CLASSES and given != "other":
        return given
    blob = f"{material} {texture}".lower()
    for name, pattern in _WORDS:
        if re.search(pattern, blob):
            return name
    return "other"


def wood_type(cls: str, material: str = "", texture: str = "") -> str | None:
    """The built-in wood map for a wooden part, or None (painted, not wood, or no fitting map)."""
    if cls not in ("solid_wood", "wood_veneer", "mdf", "laminate"):
        return None
    blob = f"{material} {texture}".lower()
    species = next((s for s in SPECIES if s in blob), None)
    return species or _ENGINEERED.get(cls) or ("oak" if cls == "solid_wood" else None)


def entry(component: dict[str, Any]) -> dict[str, Any]:
    cls = material_class(str(component.get("material") or ""), str(component.get("texture") or ""), component.get("materialClass"))
    finish = str(component.get("finish") or "").lower()
    preset = dict(PRESETS[cls])
    rough = next((v for k, v in FINISH_ROUGHNESS.items() if k in finish), None)
    if rough is not None and cls not in ("glass", "mirror"):
        preset["roughness"] = rough
    return {
        "class": cls,
        "color": component.get("color") or DEFAULT_COLOR.get(cls),
        "finish": finish,
        "texture": str(component.get("texture") or ""),
        "woodType": wood_type(cls, str(component.get("material") or ""), str(component.get("texture") or "")),
        **preset,
        "confidence": float(component.get("confidence") if component.get("confidence") is not None else 0.5),
        "source": "photo-estimate",
    }


def for_analysis(analysis: dict[str, Any] | None) -> dict[str, Any]:
    """{"parts": {component id: entry}}: the material of every part the photo reading lists."""
    parts = {str(c.get("id")): entry(c) for c in (analysis or {}).get("components") or [] if c.get("id")}
    return {
        "version": 1,
        "estimated": True,
        "note": "Read from the photo: material classes and colours are estimates, not measured finishes.",
        "parts": parts,
    }


def _linear(hex_color: str | None) -> list[float]:
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", str(hex_color or ""))
    if not m:
        return [0.8, 0.8, 0.8]
    rgb = [int(m.group(1)[i : i + 2], 16) / 255 for i in (0, 2, 4)]
    return [round(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4, 4) for c in rgb]


def gltf_material(spec: dict[str, Any], name: str) -> tuple[dict[str, Any], set[str]]:
    """One glTF material for a spec entry, and the extensions it uses."""
    alpha = spec.get("alpha", 1.0)
    mat: dict[str, Any] = {
        "name": name,
        "pbrMetallicRoughness": {"baseColorFactor": [*_linear(spec.get("color")), alpha], "metallicFactor": spec.get("metallic", 0.0), "roughnessFactor": spec.get("roughness", 0.6)},
        "extras": {"materialClass": spec.get("class"), "woodType": spec.get("woodType"), "estimated": True},
    }
    used: set[str] = set()
    if alpha < 1:
        mat["alphaMode"] = "BLEND"
    if spec.get("transmission"):
        mat.setdefault("extensions", {})["KHR_materials_transmission"] = {"transmissionFactor": spec["transmission"]}
        used.add("KHR_materials_transmission")
    if spec.get("sheen"):
        mat.setdefault("extensions", {})["KHR_materials_sheen"] = {"sheenColorFactor": [spec["sheen"]] * 3, "sheenRoughnessFactor": 0.8}
        used.add("KHR_materials_sheen")
    return mat, used
