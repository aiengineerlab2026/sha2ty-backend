"""One-page PDF report of an implementation request (freelance_tasks row with its design spec), for the customer,
the vendors who may see it and admins: the Sha2ty logo in the header, the model's picture and size, the parts with
their materials and sizes, and the customer's notes (no offers). Arabic or English; Arabic is laid out right to left.

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

NAVY, AMBER = HexColor("#172033"), HexColor("#F2A33A")  # the logo's own colours
GREEN, SAGE, MINT, BRONZE = (HexColor(c) for c in ("#2F5D50", "#8FA998", "#E7F0EA", "#B08A5A"))
INK, MUTED, LINE, SOFT, PAGE, WHITE = (HexColor(c) for c in ("#1F2A27", "#6B7772", "#E6E9E7", "#F3F5F4", "#F8F9FA", "#FFFFFF"))
W, H = A4
M = 30  # page margin
PAD = 16  # inside a card
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "report": "Implementation request report", "ref": "Request", "created": "Created", "deadline": "Deadline", "budget": "Budget",
        "quantity": "Quantity", "location": "Location", "status": "Status", "audience": "Sent to", "open": "All approved vendors",
        "model": "Model", "by": "Requested by", "size": "Size (W × H × D)", "based": "Based on", "wood": "Wood (the whole model)", "model_wood": "The model's wood", "part_size": "Size (W × H × D)",
        "file": "3D model file", "download": "Download GLB", "parts": "Parts to make", "part": "Part", "material": "Material / colour",
        "whole": "The whole model", "more": "+ {n} more", "notes": "Customer notes", "none": "None",
        "generated": "Generated {at}",
        "page": "Page 1 of 1", "no_picture": "No picture", "egp": "EGP",
        "published": "Open for offers", "awarded": "Awarded", "in_progress": "In progress", "completed": "Completed", "cancelled": "Cancelled",
        "proportional": "resized proportionally", "independent": "each side resized",
    },
    "ar": {
        "report": "تقرير طلب تنفيذ", "ref": "رقم الطلب", "created": "تاريخ الطلب", "deadline": "الموعد النهائي", "budget": "الميزانية",
        "quantity": "الكمية", "location": "الموقع", "status": "الحالة", "audience": "مرسل إلى", "open": "كل البائعين المعتمدين",
        "model": "الموديل", "by": "مقدم الطلب", "size": "المقاس (عرض × ارتفاع × عمق)", "based": "مبني على", "wood": "الخشب (للموديل بالكامل)", "model_wood": "خشب الموديل", "part_size": "المقاس (عرض × ارتفاع × عمق)",
        "file": "ملف الموديل ثلاثي الأبعاد", "download": "تنزيل GLB", "parts": "الأجزاء المطلوب تنفيذها", "part": "الجزء", "material": "الخامة / اللون",
        "whole": "الموديل بالكامل", "more": "+ {n} أخرى", "notes": "ملاحظات العميل", "none": "لا يوجد",
        "generated": "أُنشئ في {at}",
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
        self.c.saveState()
        if fill is not None:
            self.c.setFillColor(fill)
        if stroke:
            self.c.setStrokeColor(stroke)
            self.c.setLineWidth(0.8)
        self.c.roundRect(self.x(x, width), y, width, height, radius, stroke=1 if stroke else 0, fill=0 if fill is None else 1)
        self.c.restoreState()

    def card(self, x: float, top: float, width: float, height: float) -> None:
        """A white card with a hairline border and a soft shadow, `top` its upper edge."""
        self.box(x + 1, top - height - 1.5, width, height, fill=HexColor("#EDEFEE"), stroke=None, radius=10)
        self.box(x, top - height, width, height, fill=WHITE, stroke=LINE, radius=10)

    def heading(self, x: float, y: float, text: str) -> None:
        """A card's title with a short bronze bar before it."""
        self.box(x, y - 1, 3, 11, fill=BRONZE, stroke=None, radius=1.5)
        self.text(x + 9, y, text, 10.5, bold=True, color=GREEN)

    def icon(self, kind: str, x: float, y: float, color=SAGE) -> None:
        """A small line icon (9 pt) at `x` in reading order: a calendar or a map pin."""
        c, left = self.c, self.x(x, 9)
        c.saveState()
        c.setStrokeColor(color)
        c.setFillColor(color)
        c.setLineWidth(0.9)
        if kind == "calendar":
            c.roundRect(left, y, 9, 8, 1.5, stroke=1, fill=0)
            c.line(left, y + 5.5, left + 9, y + 5.5)
            c.line(left + 2.5, y + 7, left + 2.5, y + 9.5)
            c.line(left + 6.5, y + 7, left + 6.5, y + 9.5)
        else:
            c.circle(left + 4.5, y + 5.5, 3.2, stroke=1, fill=0)
            c.line(left + 2.2, y + 3.3, left + 4.5, y - 0.5)
            c.line(left + 6.8, y + 3.3, left + 4.5, y - 0.5)
        c.restoreState()

    def pill(self, x: float, y: float, text: str, fill=MINT, color=GREEN, end: bool = False) -> None:
        """A status badge at `x` (or ending there, `end`)."""
        width = pdfmetrics.stringWidth(shaped(text), "Sans-Bold", 7.5) + 16
        start = x - width if end else x
        self.box(start, y - 4, width, 15, fill=fill, stroke=None, radius=7.5)
        self.text(start + 8, y + 0.5, text, 7.5, bold=True, color=color)


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


def _picture(c: Canvas, p: Page, picture: bytes | None, x: float, y: float, width: float, height: float, label: str) -> None:
    """The model's picture fitted inside a light gray frame (bottom-left corner x, y in reading order)."""
    p.box(x, y, width, height, fill=SOFT, stroke=None, radius=8)
    if picture:
        try:
            image = ImageReader(io.BytesIO(picture))
            iw, ih = image.getSize()
            scale = min((width - 20) / iw, (height - 20) / ih)
            dw, dh = iw * scale, ih * scale
            c.drawImage(image, p.x(x + (width - dw) / 2, dw), y + (height - dh) / 2, dw, dh, mask="auto")
            return
        except Exception:  # noqa: BLE001 - a broken picture leaves the frame empty, the report still goes out
            pass
    p.text(x + width / 2 - 25, y + height / 2, label, 8, color=MUTED)


def build(task: dict[str, Any], picture: bytes | None, lang: Lang = "ar", now: datetime | None = None, download: bool = True, sizes: list[dict[str, Any]] | None = None, customer: str | None = None) -> bytes:
    """The PDF (bytes). `picture`: the model's thumbnail; `customer`: the name of who made the request;
    `download`: the 3D file link is included (the customer, admins, the vendor whose offer was accepted); `sizes`: each
    part's measured size (services/part_sizes.py; soft furnishings have none).

    Cards on a light gray page: the header (logo, report name, reference, status), the overview (picture beside the
    summary), the parts table, the customer's notes, and the footer."""
    L = LABELS[lang]
    out = io.BytesIO()
    c = Canvas(out, pagesize=A4, pageCompression=1)
    c.setTitle(f"{L['report']}: {task.get('title') or ''}")
    c.setAuthor("Sha2ty")
    p = Page(c, rtl=lang == "ar")
    spec = task.get("spec") or {}
    design = spec.get("design") or {}
    content = W - 2 * M
    c.setFillColor(PAGE)
    c.rect(0, 0, W, H, stroke=0, fill=1)

    # Header card: the logo at the start, the report's name, reference and status at the end.
    top = H - M
    p.card(M, top, content, 62)
    logo(c, p.x(M + PAD, 112), top - 46)
    p.text(W - M - PAD, top - 22, L["report"], 13, bold=True, color=GREEN, end=True)
    ref = f"{L['ref']} {'' if p.rtl else '#'}{str(task.get('id', ''))[:8].upper()}"  # a "#" beside Arabic is reordered
    p.text(W - M - PAD, top - 40, ref, 8.5, color=MUTED, end=True)
    status = L.get(str(task.get("status")), str(task.get("status") or "—"))
    p.pill(W - M - PAD - pdfmetrics.stringWidth(shaped(ref), "Sans", 8.5) - 10, top - 40, status, end=True)

    # Overview: the picture beside the summary card.
    top -= 62 + 12
    over_h, pic_w = 206, 214
    p.card(M, top, pic_w, over_h)
    _picture(c, p, picture, M + 8, top - over_h + 8, pic_w - 16, over_h - 16, L["no_picture"])
    sx, sw = M + pic_w + 12, content - pic_w - 12
    p.card(sx, top, sw, over_h)
    ix, iw = sx + PAD, sw - 2 * PAD
    dims = design.get("dimensions") or {}
    source = design.get("source") or {}
    wood = next(((m.get("material") or {}) for m in design.get("materials") or [] if (m.get("material") or {}).get("kind") == "wood"), None)
    y = top - 26
    p.text(ix, y, fit(task.get("title") or design.get("title") or "", "Sans-Bold", 15, iw), 15, bold=True, color=INK)
    model = design.get("title")
    # Under the title: who made the request and the model, when it is named otherwise.
    sub = [f"{L['by']}: {customer}"] if customer else []
    sub += [f"{L['model']}: {model}"] if model and model != task.get("title") else []
    if sub:
        y -= 14
        p.text(ix, y, fit(" · ".join(sub), "Sans", 8.5, iw), 8.5, color=MUTED)
    y -= 22
    size = f"{_cm(dims.get('width'))} × {_cm(dims.get('height'))} × {_cm(dims.get('depth'))} cm"
    mode = f" · {L[design['resize_mode']]}" if design.get("resize_mode") in ("proportional", "independent") else ""
    p.text(ix, y, L["size"], 7.5, color=MUTED)
    p.text(ix, y - 13, fit(size + mode, "Sans-Bold", 10, iw * 0.58), 10, bold=True)
    # The budget, the number vendors bid against, stands out.
    bx = ix + iw * 0.6
    p.box(bx, y - 20, iw * 0.4, 32, fill=MINT, stroke=None, radius=8)
    p.text(bx + 10, y + 1, L["budget"], 7.5, color=GREEN)
    p.text(bx + 10, y - 14, fit(_money(task.get("budget_max"), lang), "Sans-Bold", 12, iw * 0.4 - 20), 12, bold=True, color=GREEN)
    y -= 36
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(p.x(ix), y + 6, p.x(ix + iw), y + 6)
    audience = (spec.get("vendor") or {}).get("name") if task.get("audience") == "vendor_exclusive" else L["open"]
    facts = [
        (None, L["quantity"], str(spec.get("quantity") or 1)),
        ("calendar", L["created"], _date(task.get("created_at"))),
        ("calendar", L["deadline"], _date(task.get("deadline"))),
        ("pin", L["location"], task.get("location") or "—"),
        (None, L["audience"], audience or "—"),
        (None, L["wood"], wood.get("name") or "—") if wood else (None, L["based"], source.get("title") or "—"),
    ]
    cell = iw / 3
    for i, (kind, label, value) in enumerate(facts):
        fx, fy = ix + (i % 3) * cell, y - 10 - (i // 3) * 32
        tx = fx + (13 if kind else 0)
        if kind:
            p.icon(kind, fx, fy - 13)
        p.text(tx, fy, label, 7.5, color=MUTED)
        for n, line in enumerate(wrap(value, "Sans-Bold", 9, cell - (tx - fx) - 8, 2)):
            p.text(tx, fy - 12 - n * 11, line, 9, bold=True)
    if design.get("glb_url") and download:
        ly = top - over_h + 16
        p.text(ix, ly, f"{L['file']}:", 8, color=MUTED)
        lx = ix + pdfmetrics.stringWidth(shaped(L["file"] + ":"), "Sans", 8) + 6
        p.text(lx, ly, L["download"], 8.5, bold=True, color=GREEN)
        link_w = pdfmetrics.stringWidth(shaped(L["download"]), "Sans-Bold", 8.5)
        c.linkURL(design["glb_url"], (p.x(lx, link_w), ly - 3, p.x(lx, link_w) + link_w, ly + 10), relative=0)

    # Specifications: each part with its material and size.
    top -= over_h + 12
    edits = {m.get("part_id"): m for m in design.get("materials") or []}
    parts = [part for part in design.get("parts") or [] if part.get("requested", True)]
    row_h, max_rows = 19, 10
    shown = parts[: max_rows if len(parts) <= max_rows else max_rows - 1]
    rows = len(shown) + (len(parts) > len(shown)) + (not parts)
    spec_h = 44 + row_h * (rows + 1) + 10
    p.card(M, top, content, spec_h)
    p.heading(M + PAD, top - 24, L["parts"] + ("" if not design.get("whole_model") else f" — {L['whole']}"))
    tx, tw = M + PAD, content - 2 * PAD
    cols = [(L["part"], 0.32), (L["material"], 0.38), (L["part_size"], 0.30)]
    size_of = {s["name"]: s for s in sizes or []}
    y = top - 38
    p.box(tx, y - row_h, tw, row_h, fill=SOFT, stroke=None, radius=5)
    cx = tx
    for label, share in cols:
        p.text(cx + 8, y - 13, label, 7.5, bold=True, color=MUTED)
        cx += tw * share
    y -= row_h
    c.setStrokeColor(LINE)
    c.setLineWidth(0.5)
    for i, part in enumerate(shown):
        edit = edits.get(part.get("id")) or {}
        size = size_of.get(part.get("name") or "")
        values = [part.get("name") or part.get("id") or "", material_text(edit or None, lang),
                  f"{_cm(size['width'])} × {_cm(size['height'])} × {_cm(size['depth'])} cm" + (f" (×{size['count']})" if size["count"] > 1 else "") if size else "—"]
        color = None if (edit.get("material") or {}).get("kind") == "wood" else edit.get("color")
        cx = tx
        for col, ((_, share), value) in enumerate(zip(cols, values)):
            swatch = col == 1 and isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color)
            text = fit(value, "Sans-Bold" if col == 0 else "Sans", 8.5, tw * share - (32 if swatch else 14))
            p.text(cx + 8, y - 13, text, 8.5, bold=col == 0, color=INK)
            if swatch:
                at = cx + 8 + (pdfmetrics.stringWidth(shaped(text), "Sans", 8.5) + 6 if text else 0)
                p.box(at, y - 14.5, 10, 10, fill=HexColor(color), stroke=LINE, radius=5)
            cx += tw * share
        if i < len(shown) - 1:
            c.line(tx + 8, y - row_h, tx + tw - 8, y - row_h)
        y -= row_h
    if len(parts) > len(shown):
        p.text(tx + 8, y - 13, L["more"].format(n=len(parts) - len(shown)), 8, color=MUTED)
        y -= row_h
    if not parts:
        p.text(tx + 8, y - 13, L["none"], 8.5, color=MUTED)

    # The customer's notes, in a callout across the page.
    top -= spec_h + 12
    room = top - (M + 26) - 10
    note = (task.get("description") or "").strip()
    lines = wrap(note, "Sans", 8.5, content - 2 * PAD - 18, max(1, int((room - 70) // 12))) if note else [L["none"]]
    call_h = 16 + 12 * len(lines)
    p.card(M, top, content, 38 + call_h + PAD + 4)  # the card hugs its text
    p.heading(M + PAD, top - 24, L["notes"])
    p.box(M + PAD, top - 38 - call_h, content - 2 * PAD, call_h, fill=HexColor("#FBF7F1"), stroke=None, radius=6)
    p.box(M + PAD, top - 38 - call_h, 3, call_h, fill=BRONZE, stroke=None, radius=1.5)
    ny = top - 50
    for line in lines:
        p.text(M + PAD + 12, ny, line, 8.5, color=INK if note else MUTED)
        ny -= 12

    # Footer: the site, when it was made and the page, under a thin line.
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(M, M + 12, W - M, M + 12)
    stamp = (now or datetime.now(timezone.utc)).strftime("%d/%m/%Y %H:%M UTC")
    p.text(M, M, "sha2ty.net", 7.5, bold=True, color=GREEN)
    generated = L["generated"].format(at=stamp)
    p.text(W / 2 - pdfmetrics.stringWidth(shaped(generated), "Sans", 7.5) / 2, M, generated, 7.5, color=MUTED)
    p.text(W - M, M, L["page"], 7.5, color=MUTED, end=True)
    c.showPage()
    c.save()
    return out.getvalue()
