"""A photographed four-door wardrobe (drawn: pink outer doors, white inner doors, oak frame, a fluted plinth drawer,
one handle per door) and the spec a vision model returns for it. Shared by the hybrid engine's tests."""

from __future__ import annotations

import io

from PIL import Image, ImageDraw

OBJ = [0.1, 0.05, 0.9, 0.95]  # the wardrobe's box in the photo
PINK, WHITE, OAK, BRASS = (217, 165, 168), (240, 236, 228), (176, 138, 92), (201, 162, 39)


def photo(w=800, h=900) -> bytes:
    img = Image.new("RGB", (w, h), (120, 130, 140))  # a blue-grey wall
    d = ImageDraw.Draw(img)
    X = lambda f: int((OBJ[0] + f * (OBJ[2] - OBJ[0])) * w)  # noqa: E731
    Y = lambda f: int((OBJ[1] + f * (OBJ[3] - OBJ[1])) * h)  # noqa: E731
    d.rectangle((X(0), Y(0), X(1), Y(1)), fill=OAK)
    for i, colour in enumerate((PINK, WHITE, WHITE, PINK)):
        d.rectangle((X(0.02 + i * 0.24), Y(0.03), X(0.02 + (i + 1) * 0.24 - 0.005), Y(0.82)), fill=colour)
    for i in range(16):  # flutes on the plinth drawer
        d.line((X(0.03 + i * 0.06), Y(0.85), X(0.03 + i * 0.06), Y(0.97)), fill=(140, 105, 66), width=3)
    for x in (0.23, 0.29, 0.71, 0.77):  # handles near the meeting stiles
        d.rectangle((X(x), Y(0.38), X(x + 0.012), Y(0.5)), fill=BRASS)
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def comp(id_, name, box, cls="painted_wood", color=None, role="structural", motion=None, relations=None, deco="none", complexity="regular", vis="visible"):
    return {"id": id_, "name": name, "nameAr": "", "box": box, "color": color, "materialClass": cls, "role": role, "visibility": vis,
            "confidence": 0.9, "motion": motion or {"type": "none"}, "relations": relations or [], "decoration": deco, "complexity": complexity,
            "finish": "satin", "material": "", "texture": ""}


def door(i, hinge):
    colour = "#D9A5A8" if i in (0, 3) else "#F0ECE4"
    return comp(f"door_{i + 1}", f"Door {i + 1}", [0.02 + i * 0.24, 0.03, 0.02 + (i + 1) * 0.24 - 0.005, 0.82], color=colour, role="functional",
                motion={"type": "rotate", "hinge": hinge, "confidence": 0.8}, relations=[{"type": "attached_to", "target": "body"}])


def spec(extra=None, size=(1.6, 2.0, 0.58), **over) -> dict:
    handles = [comp(f"handle_{n + 1}", f"Door {n + 1} handle", [x, 0.38, x + 0.012, 0.5], cls="metal", color="#C9A227", role="functional",
                    relations=[{"type": "attached_to", "target": f"door_{n + 1}"}]) for n, x in enumerate((0.23, 0.29, 0.71, 0.77))]
    comps = [comp("body", "Wardrobe body", [0, 0, 1, 1], cls="wood_veneer", color="#B08A5C"),
             *[door(i, h) for i, h in enumerate(("left", "right", "left", "right"))], *handles,
             comp("drawer", "Plinth drawer", [0.02, 0.84, 0.98, 0.98], color="#B08A5C", role="functional", deco="fluted",
                  motion={"type": "slide", "direction": "front", "confidence": 0.7}),
             *(extra or [])]
    w, h, d = size
    return {"details": {"bbox": OBJ, "width_m": w, "height_m": h, "depth_m": d, "dimension_source": over.get("source", "inferred"), "object_type": "wardrobe"},
            "analysis": {"objectType": "wardrobe", "components": comps, "movable": True, "bodyId": "body"}}
