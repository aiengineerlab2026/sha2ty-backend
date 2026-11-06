"""One-page PDF report of an implementation request (freelance_tasks row with its design spec), for the customer,
the vendors who may see it and admins: the Sha2ty logo in the header, the model's picture and size, the parts with
their materials and variants, the customer's notes and the offers. Arabic or English; Arabic is laid out right to left.

Always one A4 page: long lists and notes are cut short with "+ N more" / "…" (the full request is on the site).
"""

from __future__ import annotations

import io
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import arabic_reshaper
from bidi.algorithm import get_display
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

Lang = Literal["ar", "en"]
FONTS = Path(__file__).resolve().parent.parent / "assets" / "fonts"
pdfmetrics.registerFont(TTFont("Sans", str(FONTS / "DejaVuSans.ttf")))  # Arabic and Latin in one font
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONTS / "DejaVuSans-Bold.ttf")))

NAVY, AMBER, INK, MUTED, LINE, SOFT = (HexColor(c) for c in ("#172033", "#F2A33A", "#1E293B", "#64748B", "#E2E8F0", "#F8FAFC"))
W, H = A4
M = 40  # page margin
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "report": "Implementation request report", "ref": "Request", "created": "Created", "deadline": "Deadline", "budget": "Budget",
        "quantity": "Quantity", "location": "Location", "status": "Status", "audience": "Sent to", "open": "All approved vendors",
        "model": "Model", "size": "Size (W × H × D)", "based": "Based on", "wood": "Wood (the whole model)", "model_wood": "The model's wood",
        "file": "3D model file", "download": "Download GLB", "parts": "Parts to make", "part": "Part", "material": "Material / colour",
        "whole": "The whole model", "more": "+ {n} more", "notes": "Customer notes", "none": "None",
        "offers": "Offers", "no_offers": "No offers yet.", "accepted": "Accepted", "days": "{n} days", "generated": "Generated {at}",
        "page": "Page 1 of 1", "no_picture": "No picture", "egp": "EGP",
        "published": "Open for offers", "awarded": "Awarded", "in_progress": "In progress", "completed": "Completed", "cancelled": "Cancelled",
        "proportional": "resized proportionally", "independent": "each side resized",
    },
    "ar": {
        "report": "تقرير طلب تنفيذ", "ref": "الطلب", "created": "تاريخ الطلب", "deadline": "الموعد النهائي", "budget": "الميزانية",
        "quantity": "الكمية", "location": "الموقع", "status": "الحالة", "audience": "مرسل إلى", "open": "كل البائعين المعتمدين",
        "model": "الموديل", "size": "المقاس (عرض × ارتفاع × عمق)", "based": "مبني على", "wood": "الخشب (للموديل بالكامل)", "model_wood": "خشب الموديل",
        "file": "ملف الموديل ثلاثي الأبعاد", "download": "تنزيل GLB", "parts": "الأجزاء المطلوب تنفيذها", "part": "الجزء", "material": "الخامة / اللون",
        "whole": "الموديل بالكامل", "more": "+ {n} أخرى", "notes": "ملاحظات العميل", "none": "لا يوجد",
        "offers": "العروض", "no_offers": "لا توجد عروض بعد.", "accepted": "مقبول", "days": "{n} يوم", "generated": "أُنشئ في {at}",
        "page": "صفحة 1 من 1", "no_picture": "لا توجد صورة", "egp": "ج.م",
        "published": "متاح للعروض", "awarded": "تم الترسية", "in_progress": "قيد التنفيذ", "completed": "مكتمل", "cancelled": "ملغي",
        "proportional": "تغيير المقاس بالتناسب", "independent": "تغيير كل بُعد على حدة",
    },
}


def shaped(text: str) -> str:
    """Arabic letters joined and in display order (PDF text is drawn left to right)."""
    return get_display(arabic_reshaper.reshape(text)) if ARABIC.search(text) else text


def fit(text: str, font: str, size: float, width: float) -> str:
    """The text cut with "…" to fit `width` points (on the logical string, before shaping)."""
    if pdfmetrics.stringWidth(shaped(text), font, size) <= width:
        return text
    while text and pdfmetrics.stringWidth(shaped(text + "…"), font, size) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def wrap(text: str, font: str, size: float, width: float, lines: int) -> list[str]:
    """Words into at most `lines` lines of `width`; the last one ends with "…" when the text is longer."""
    out: list[str] = []
    for paragraph in text.splitlines() or [""]:
        line = ""
        for word in paragraph.split():
            trial = f"{line} {word}".strip()
            if pdfmetrics.stringWidth(shaped(trial), font, size) <= width:
                line = trial
            else:
                out.append(line)
                line = word
        out.append(line)
    out = [line for line in out if line.strip()]
    if len(out) > lines:
        out = out[:lines]
        out[-1] = fit(out[-1] + " …", font, size, width)
    return out


class Page:
    """Drawing in reading order: x is measured from the start edge (left in English, right in Arabic)."""

    def __init__(self, canvas: Canvas, rtl: bool):
        self.c, self.rtl = canvas, rtl

    def x(self, x: float, width: float = 0) -> float:
        return W - x - width if self.rtl else x

    def text(self, x: float, y: float, text: str, size: float = 9, bold: bool = False, color=INK, end: bool = False) -> None:
        """Text starting at `x` (or ending there, `end`), in reading order."""
        font = "Sans-Bold" if bold else "Sans"
        self.c.setFont(font, size)
        self.c.setFillColor(color)
        draw = self.c.drawString if self.rtl == end else self.c.drawRightString
        draw(self.x(x), y, shaped(text))

    def box(self, x: float, y: float, width: float, height: float, fill=SOFT, stroke=LINE, radius: float = 6) -> None:
        self.c.setFillColor(fill)
        if stroke:
            self.c.setStrokeColor(stroke)
        self.c.roundRect(self.x(x, width), y, width, height, radius, stroke=1 if stroke else 0, fill=1)


def logo(c: Canvas, x: float, y: float, size: float = 30) -> None:
    """The Sha2ty mark (Logo.tsx / platform.py LOGO_SVG: a navy house outline with the amber hammer) and the wordmark."""
    k = size / 100
    c.saveState()
    c.translate(x, y)
    c.setStrokeColor(NAVY)
    c.setLineWidth(7 * k)
    c.setLineJoin(1)
    path = c.beginPath()
    for i, (px, py) in enumerate([(16, 50), (50, 20), (84, 50), (84, 86), (16, 86)]):
        (path.moveTo if i == 0 else path.lineTo)(px * k, (100 - py) * k)
    path.close()
    c.drawPath(path, stroke=1, fill=0)
    c.translate(50 * k, 36 * k)
    c.rotate(40)
    c.setFillColor(NAVY)
    c.roundRect(-4 * k, -18 * k, 8 * k, 28 * k, 4 * k, stroke=0, fill=1)
    c.setFillColor(AMBER)
    c.roundRect(-14 * k, 9 * k, 28 * k, 11 * k, 2.5 * k, stroke=0, fill=1)
    c.restoreState()
    c.setFillColor(NAVY)
    c.setFont("Sans-Bold", size * 0.72)
    c.drawString(x + size + 6, y + size * 0.22, "sha2ty")
    c.setFillColor(AMBER)
    c.rect(x + size + 6, y + size * 0.08, pdfmetrics.stringWidth("sha2ty", "Sans-Bold", size * 0.72) * 0.9, 2.2, stroke=0, fill=1)


def _date(value: str | None) -> str:
    if not value:
        return "—"
    try:
        day = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value[:10]
    return day.strftime("%d/%m/%Y")


def _money(value: float | None, lang: Lang) -> str:
    return "—" if value is None else f"{value:,.0f} {LABELS[lang]['egp']}"


def _cm(value: float | None) -> str:
    return "—" if not value else f"{value * 100:.0f}"


def material_text(edit: dict[str, Any] | None, lang: Lang = "en") -> str:
    """The material used on a part, by name only (no amounts or prices). A wooden part is the model's one wood."""
    if not edit:
        return "—"
    material = edit.get("material") or {}
    if material.get("kind") == "wood":
        return LABELS[lang]["model_wood"]
    # The colour is drawn as a swatch beside this text (a hex code beside Arabic would be reordered).
    return material.get("name") or ("" if edit.get("color") else "—")


def build(task: dict[str, Any], bids: list[dict[str, Any]], vendors: dict[str, str], picture: bytes | None, lang: Lang = "ar", now: datetime | None = None, download: bool = True) -> bytes:
    """The PDF (bytes). `bids`: task_bids rows; `vendors`: vendor id → name; `picture`: the model's thumbnail;
    `download`: the 3D file link is included (the customer, admins, the vendor whose offer was accepted)."""
    L = LABELS[lang]
    out = io.BytesIO()
    c = Canvas(out, pagesize=A4, pageCompression=1)
    c.setTitle(f"{L['report']}: {task.get('title') or ''}")
    c.setAuthor("Sha2ty")
    p = Page(c, rtl=lang == "ar")
    spec = task.get("spec") or {}
    design = spec.get("design") or {}
    content = W - 2 * M

    # Header: the logo, the report's name and the request's reference.
    logo(c, p.x(M, 112), H - M - 30)
    p.text(W - M, H - M - 10, L["report"], 12, bold=True, color=NAVY, end=True)
    p.text(W - M, H - M - 25, f"{L['ref']} #{str(task.get('id', ''))[:8].upper()}", 8.5, color=MUTED, end=True)
    c.setStrokeColor(AMBER)
    c.setLineWidth(2)
    c.line(M, H - M - 40, W - M, H - M - 40)

    # Title and the request's facts.
    y = H - M - 70
    p.text(M, y, fit(task.get("title") or design.get("title") or "", "Sans-Bold", 17, content), 17, bold=True, color=NAVY)
    y -= 26
    audience = (spec.get("vendor") or {}).get("name") if task.get("audience") == "vendor_exclusive" else L["open"]
    facts = [
        (L["status"], L.get(str(task.get("status")), str(task.get("status") or "—"))),
        (L["created"], _date(task.get("created_at"))),
        (L["deadline"], _date(task.get("deadline"))),
        (L["budget"], _money(task.get("budget_max"), lang)),
        (L["quantity"], str(spec.get("quantity") or 1)),
        (L["audience"], audience or "—"),
    ]
    cell = content / 3
    for i, (label, value) in enumerate(facts):
        cx, cy = M + (i % 3) * cell, y - (i // 3) * 30
        p.text(cx, cy, label, 7.5, color=MUTED)
        p.text(cx, cy - 12, fit(value, "Sans-Bold", 9.5, cell - 10), 9.5, bold=True)
    y -= 72
    if task.get("location"):
        p.text(M, y, f"{L['location']}: {fit(task['location'], 'Sans', 8.5, content - 80)}", 8.5, color=MUTED)
        y -= 16

    # The model: its picture beside its size, source and wood.
    pic_w, pic_h = 200, 150
    p.box(M, y - pic_h, pic_w, pic_h, fill=HexColor("#FFFFFF"))
    if picture:
        try:
            image = ImageReader(io.BytesIO(picture))
            iw, ih = image.getSize()
            scale = min((pic_w - 12) / iw, (pic_h - 12) / ih)
            dw, dh = iw * scale, ih * scale
            c.drawImage(image, p.x(M + (pic_w - dw) / 2, dw), y - pic_h + (pic_h - dh) / 2, dw, dh, mask="auto")
        except Exception:  # noqa: BLE001 - a broken picture leaves the frame empty, the report still goes out
            picture = None
    if not picture:
        p.text(M + pic_w / 2 - 25, y - pic_h / 2, L["no_picture"], 8, color=MUTED)
    dims = design.get("dimensions") or {}
    source = design.get("source") or {}
    # One wood for the whole model (the editor puts it on every wooden part); each part's own material goes on it.
    wood = next(((m.get("material") or {}) for m in design.get("materials") or [] if (m.get("material") or {}).get("kind") == "wood"), None)
    info_x, info_w = M + pic_w + 18, content - pic_w - 18
    rows = [
        (L["model"], design.get("title") or "—"),
        (L["size"], f"{_cm(dims.get('width'))} × {_cm(dims.get('height'))} × {_cm(dims.get('depth'))} cm"
         + (f" ({L[design['resize_mode']]})" if design.get("resize_mode") in ("proportional", "independent") else "")),
        (L["based"], source.get("title") or "—"),
        *([(L["wood"], wood.get("name") or "—")] if wood else []),
    ]
    ry = y - 10
    for label, value in rows:
        p.text(info_x, ry, label, 7.5, color=MUTED)
        p.text(info_x, ry - 12, fit(value, "Sans-Bold", 9.5, info_w), 9.5, bold=True)
        ry -= 30
    if design.get("glb_url") and download:
        p.text(info_x, ry, L["file"], 7.5, color=MUTED)
        p.text(info_x, ry - 12, L["download"], 9.5, bold=True, color=HexColor("#2563EB"))
        link_w = pdfmetrics.stringWidth(shaped(L["download"]), "Sans-Bold", 9.5)
        c.linkURL(design["glb_url"], (p.x(info_x, link_w), ry - 15, p.x(info_x, link_w) + link_w, ry - 2), relative=0)
    y -= pic_h + 22

    # Parts to make, with each one's material and variant.
    p.text(M, y, L["parts"] + ("" if not design.get("whole_model") else f" — {L['whole']}"), 10.5, bold=True, color=NAVY)
    y -= 8
    edits = {m.get("part_id"): m for m in design.get("materials") or []}
    parts = [part for part in design.get("parts") or [] if part.get("requested", True)]
    cols = [(L["part"], 0.4), (L["material"], 0.6)]
    row_h, max_rows = 17, 11
    p.box(M, y - row_h, content, row_h, fill=NAVY, stroke=None, radius=4)
    cx = M
    for label, share in cols:
        p.text(cx + 7, y - 12, label, 8, bold=True, color=HexColor("#FFFFFF"))
        cx += content * share
    y -= row_h
    shown = parts[: max_rows if len(parts) <= max_rows else max_rows - 1]
    for i, part in enumerate(shown):
        if i % 2:
            p.box(M, y - row_h, content, row_h, fill=SOFT, stroke=None, radius=0)
        edit = edits.get(part.get("id")) or {}
        values = [part.get("name") or part.get("id") or "", material_text(edit or None, lang)]
        cx = M
        color = None if (edit.get("material") or {}).get("kind") == "wood" else edit.get("color")
        for col, ((_, share), value) in enumerate(zip(cols, values)):
            text = fit(value, "Sans", 8.5, content * share - (30 if col == 1 and color else 12))
            p.text(cx + 7, y - 12, text, 8.5)
            if col == 1 and isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color):
                at = cx + 7 + (pdfmetrics.stringWidth(shaped(text), "Sans", 8.5) + 5 if text else 0)
                p.box(at, y - 13.5, 10, 10, fill=HexColor(color), stroke=LINE, radius=2)
            cx += content * share
        y -= row_h
    if len(parts) > len(shown):
        p.text(M + 7, y - 12, L["more"].format(n=len(parts) - len(shown)), 8, color=MUTED)
        y -= row_h
    if not parts:
        p.text(M + 7, y - 12, L["none"], 8.5, color=MUTED)
        y -= row_h
    y -= 18

    # The customer's notes, then the offers, side by side.
    half = (content - 18) / 2
    top = y
    p.text(M, y, L["notes"], 10.5, bold=True, color=NAVY)
    note = (task.get("description") or "").strip()
    ny = y - 16
    for line in wrap(note, "Sans", 8.5, half, 7) if note else [L["none"]]:
        p.text(M, ny, line, 8.5, color=INK if note else MUTED)
        ny -= 12
    ox = M + half + 18
    p.text(ox, top, L["offers"], 10.5, bold=True, color=NAVY)
    oy = top - 16
    live = sorted((b for b in bids if b.get("status") != "withdrawn"), key=lambda b: (b.get("status") != "accepted", b.get("price") or 0))
    if not live:
        p.text(ox, oy, L["no_offers"], 8.5, color=MUTED)
    for bid in live[:5]:
        name = vendors.get(str(bid.get("vendor_id")), "—")
        tag = f"  ✓ {L['accepted']}" if bid.get("status") == "accepted" else ""
        p.text(ox, oy, fit(name, "Sans-Bold", 8.5, half * 0.5) + tag, 8.5, bold=True, color=HexColor("#047857") if tag else INK)
        p.text(ox + half, oy, f"{_money(bid.get('price'), lang)} · {L['days'].format(n=bid.get('days'))}", 8.5, end=True)
        oy -= 13
    if len(live) > 5:
        p.text(ox, oy, L["more"].format(n=len(live) - 5), 8, color=MUTED)

    # Footer.
    c.setStrokeColor(LINE)
    c.setLineWidth(0.8)
    c.line(M, M + 14, W - M, M + 14)
    stamp = (now or datetime.now(timezone.utc)).strftime("%d/%m/%Y %H:%M UTC")
    p.text(M, M, "sha2ty.net", 7.5, color=MUTED)
    p.text(W / 2 - 60, M, L["generated"].format(at=stamp), 7.5, color=MUTED)
    p.text(W - M, M, L["page"], 7.5, color=MUTED, end=True)
    c.showPage()
    c.save()
    return out.getvalue()
