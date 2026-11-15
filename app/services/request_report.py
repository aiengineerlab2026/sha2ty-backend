"""PDF report of an implementation request (freelance_tasks row with its design spec), for the customer, the vendors
who may see it and admins. Page 1: the Sha2ty logo, the model's picture and size, the request's details, the 3D file
link and the customer's notes (no offers). Then "3D Model Views & Dimensions": the saved model rendered closed (its
overall size) and opened, with each part's dimensions written beside it, and close-ups two to a page. Arabic or
English; Arabic is laid out right to left. Page 1 is always one A4 page: long notes are cut short with "…".
"""

from __future__ import annotations

import io
import re
import unicodedata
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
# Cairo (the site's Arabic font, Latin letters too; SIL Open Font License) for the whole report.
_CAIRO = TTFont("Sans", str(FONTS / "Cairo-Regular.ttf"))
pdfmetrics.registerFont(_CAIRO)
pdfmetrics.registerFont(TTFont("Sans-Bold", str(FONTS / "Cairo-Bold.ttf")))
_GLYPHS = set(_CAIRO.face.charToGlyph)

NAVY, AMBER = HexColor("#172033"), HexColor("#F2A33A")  # the logo's own colours
GOLD, CREAM = HexColor("#B8731D"), HexColor("#FDF6EA")  # amber's deep tone (text, lines) and its lightest wash
GREEN, SAGE, MINT, BRONZE = (HexColor(c) for c in ("#2F7A5C", "#94A3B8", "#E8F3EE", "#B8731D"))
INK, MUTED, LINE, SOFT, PAGE, WHITE = (HexColor(c) for c in ("#172033", "#64748B", "#E2E8F0", "#F1F5F9", "#F8FAFC", "#FFFFFF"))
W, H = A4
M = 57  # page margin: 20 mm all round
PAD = 18  # inside a card
GAP = 20  # between cards and sections
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "report": "Implementation request report", "ref": "Request", "created": "Created", "deadline": "Deadline", "budget": "Budget",
        "quantity": "Quantity", "location": "Location", "status": "Status", "audience": "Sent to", "open": "All approved vendors",
        "model": "Model", "by": "Requested by", "size": "Size (W × H × D)", "based": "Based on", "wood": "Wood (the whole model)",
        "file": "3D model file", "download": "Download GLB", "notes": "Customer notes", "none": "None",
        "generated": "Generated {at}",
        "page": "Page {i} of {n}", "no_picture": "No picture", "egp": "EGP", "cm": "cm",
        "views": "3D Model Views & Dimensions", "closed_view": "Closed — overall dimensions", "open_view": "Opened — the main structure", "detail_view": "Detail {i}",
        "other_view": "Another angle — each part's dimensions (no movable parts to open)",
        "components": "Components and materials (cm)", "added": "Added", "wood_type": "Wood type", "mm": "mm", "surface": "Surface",
        "g_structure": "Structure", "g_doors": "Doors", "g_drawers": "Drawers", "g_shelves": "Shelves", "g_accessories": "Other parts", "key": "≈ Est.: measured from the 3D model · Set: by the customer · To confirm: with the customer · Sizes in cm",
        "c_name": "Component", "c_w": "W", "c_h": "H", "c_d": "D", "c_t": "T", "c_qty": "Qty", "c_status": "Dimensions", "c_material": "Material", "c_color": "Colour", "c_finish": "Finish",
        "requested": "The whole model (requested size)", "measured": "The 3D model as drawn", "model_size": "3D model", "confirmed": "Set", "estimated": "≈ Est.",
        "tbc": "To confirm", "kind_wood": "Wood", "kind_paint": "Paint", "kind_fabric": "Fabric", "kind_metal": "Metal", "kind_glass": "Glass",
        "matte": "Matte", "satin": "Satin", "gloss": "Gloss", "pictured": "As pictured",
        "no_views": "The pictures of the model with its dimensions are not ready for this request yet: they are made when the customer opens the request. Download the report again after that.",
        "published": "Open for offers", "awarded": "Awarded", "in_progress": "In progress", "completed": "Completed", "cancelled": "Cancelled",
        "proportional": "resized proportionally", "independent": "each side resized",
    },
    "ar": {
        "report": "تقرير طلب تنفيذ", "ref": "رقم الطلب", "created": "تاريخ الطلب", "deadline": "الموعد النهائي", "budget": "الميزانية",
        "quantity": "الكمية", "location": "الموقع", "status": "الحالة", "audience": "مرسل إلى", "open": "كل البائعين المعتمدين",
        "model": "الموديل", "by": "مقدم الطلب", "size": "المقاس (عرض × ارتفاع × عمق)", "based": "مبني على", "wood": "الخشب (للموديل بالكامل)",
        "file": "ملف الموديل ثلاثي الأبعاد", "download": "تنزيل GLB", "notes": "ملاحظات العميل", "none": "لا يوجد",
        "generated": "أُنشئ في {at}",
        "page": "صفحة {i} من {n}", "no_picture": "لا توجد صورة", "egp": "ج.م", "cm": "سم",
        "views": "مناظر الموديل ثلاثية الأبعاد والمقاسات", "closed_view": "مغلق — المقاسات الكلية", "open_view": "مفتوح — الهيكل الرئيسي", "detail_view": "تفاصيل {i}",
        "other_view": "زاوية أخرى — مقاسات كل جزء (لا توجد أجزاء متحركة)",
        "components": "المكونات والخامات (سم)", "added": "مضاف", "wood_type": "نوع الخشب", "mm": "مم", "surface": "السطح",
        "g_structure": "الهيكل", "g_doors": "الأبواب", "g_drawers": "الأدراج", "g_shelves": "الأرفف", "g_accessories": "أجزاء أخرى", "key": "≈ تقديري: مقاس من الموديل ثلاثي الأبعاد · مؤكد: حدده العميل · يحدد مع العميل: غير معروف بعد · المقاسات بالسنتيمتر",
        "c_name": "المكون", "c_w": "العرض", "c_h": "الارتفاع", "c_d": "العمق", "c_t": "السُمك", "c_qty": "العدد", "c_status": "المقاسات", "c_material": "الخامة", "c_color": "اللون", "c_finish": "التشطيب",
        "requested": "الموديل بالكامل (المقاس المطلوب)", "measured": "الموديل ثلاثي الأبعاد كما رُسم", "model_size": "الموديل ثلاثي الأبعاد", "confirmed": "مؤكد", "estimated": "≈ تقديري",
        "tbc": "يحدد مع العميل", "kind_wood": "خشب", "kind_paint": "دهان", "kind_fabric": "قماش", "kind_metal": "معدن", "kind_glass": "زجاج",
        "matte": "مطفي", "satin": "نصف لامع", "gloss": "لامع", "pictured": "كالصورة",
        "no_views": "صور الموديل بمقاساته غير جاهزة لهذا الطلب بعد: تُنشأ عندما يفتح العميل الطلب. نزّل التقرير مرة أخرى بعد ذلك.",
        "published": "متاح للعروض", "awarded": "تم الترسية", "in_progress": "قيد التنفيذ", "completed": "مكتمل", "cancelled": "ملغي",
        "proportional": "تغيير المقاس بالتناسب", "independent": "تغيير كل بُعد على حدة",
    },
}


def shaped(text: str) -> str:
    """Arabic letters joined and in display order (PDF text is drawn left to right). A joined form the font does not
    have (Cairo lacks a few isolated ones: ا ة د ص آ) is drawn as its plain letter, which looks the same."""
    if not ARABIC.search(text):
        return text
    out = get_display(arabic_reshaper.reshape(text))
    return "".join(ch if ord(ch) in _GLYPHS or not 0xFB50 <= ord(ch) <= 0xFEFF else unicodedata.normalize("NFKC", ch) for ch in out)


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
        self.box(x, y - 1, 3, 12, fill=AMBER, stroke=None, radius=1.5)
        self.text(x + 9, y, text, 11, bold=True, color=NAVY)

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


def _compact(image: Any, quality: int = 85) -> ImageReader:
    """A picture as JPEG (transparent ones on white): the PDF stays small; PNG pictures would be stored uncompressed."""
    from PIL import Image

    if isinstance(image, (bytes, bytearray)):
        image = Image.open(io.BytesIO(image))
    if image.mode in ("RGBA", "LA", "P"):
        image = image.convert("RGBA")
        flat = Image.new("RGB", image.size, "#FFFFFF")
        flat.paste(image, mask=image.split()[-1])
        image = flat
    out = io.BytesIO()
    image.convert("RGB").save(out, "JPEG", quality=quality, optimize=True)
    return ImageReader(io.BytesIO(out.getvalue()))


def _picture(c: Canvas, p: Page, picture: bytes | None, x: float, y: float, width: float, height: float, label: str, frame: bool = True) -> None:
    """The model's picture fitted inside a light gray frame (bottom-left corner x, y in reading order); without
    `frame`, as large as the space allows on the card's white (the annotated views)."""
    inset = 20 if frame else 0
    if frame:
        p.box(x, y, width, height, fill=SOFT, stroke=None, radius=8)
    if picture:
        try:
            image = _compact(picture, 85 if frame else 90)
            iw, ih = image.getSize()
            scale = min((width - inset) / iw, (height - inset) / ih)
            dw, dh = iw * scale, ih * scale
            c.drawImage(image, p.x(x + (width - dw) / 2, dw), y + (height - dh) / 2, dw, dh, mask="auto")
            return
        except Exception:  # noqa: BLE001 - a broken picture leaves the frame empty, the report still goes out
            pass
    p.text(x + width / 2 - 25, y + height / 2, label, 8, color=MUTED)


VIEWS_VERSION = 2  # pictures with the dimensions written beside the parts (frontend lib/designs/viewParts.ts)


def _footer(c: Canvas, p: Page, L: dict[str, str], stamp: str, page: int, pages: int) -> None:
    """The site, when it was made and the page, under a thin line."""
    c.setStrokeColor(LINE)
    c.setLineWidth(0.6)
    c.line(M, M + 14, W - M, M + 14)
    p.text(M, M, "sha2ty.net", 7.5, bold=True, color=GOLD)
    generated = L["generated"].format(at=stamp)
    p.text(W / 2 - pdfmetrics.stringWidth(shaped(generated), "Sans", 7.5) / 2, M, generated, 7.5, color=MUTED)
    p.text(W - M, M, L["page"].format(i=page, n=pages), 7.5, color=MUTED, end=True)


def _page_background(c: Canvas) -> None:
    c.setFillColor(PAGE)
    c.rect(0, 0, W, H, stroke=0, fill=1)


def _view_pages(meta: dict[str, Any] | None, lang: Lang = "ar") -> int:
    """The pictures' pages (one each: closed, open, every detail view) and the components table's; else the notice's."""
    if not meta:
        return 1
    return 2 + len(meta.get("detail_urls") or []) + len(_bom_pages(_components(meta, lang), lang, meta.get("_model_wood")))


def _views_pages(c: Canvas, p: Page, L: dict[str, str], views: dict[str, Any] | None, images: tuple[bytes, ...] | None, stamp: str, first: int, pages: int) -> None:
    """One large picture a page: closed (the overall size), open (the main structure), then each group's details."""
    content = W - 2 * M
    _page_background(c)
    top = H - M
    p.card(M, top, content, 40)
    p.heading(M + PAD, top - 25, L["views"])
    top -= 40 + GAP
    if not views or not images:
        p.card(M, top, content, 60)
        for n, line in enumerate(wrap(L["no_views"], "Sans", 9, content - 2 * PAD, 3)):
            p.text(M + PAD, top - 26 - n * 16, line, 9, color=MUTED)
        _footer(c, p, L, stamp, first, pages)
        return
    # One large picture a page: the closed model, the open one, then each group's details.
    groups = views.get("detail_groups") or []
    names = {g: L.get(f"g_{g}", "") for g in GROUP_ORDER}
    captions = [L["closed_view"], L["open_view"] if views.get("opened") else L["other_view"]] + [
        f"{L['detail_view'].format(i=i + 1)}" + (f" · {names[groups[i]]}" if i < len(groups) and names.get(groups[i]) else "") for i in range(len(images) - 2)
    ]
    page = first
    for n, (image, caption) in enumerate(zip(images, captions)):
        if n:
            _footer(c, p, L, stamp, page, pages)
            c.showPage()
            _page_background(c)
            page += 1
            top = H - M
        card_h = top - (M + 30)
        p.card(M, top, content, card_h)
        p.box(M + PAD, top - 31, 3, 13, fill=AMBER, stroke=None, radius=1.5)
        p.text(M + PAD + 10, top - 28, caption, 11, bold=True, color=NAVY)
        _picture(c, p, image, M + 8, top - card_h + 8, content - 16, card_h - 48, L["no_picture"], frame=False)
    _footer(c, p, L, stamp, page, pages)
    _bom(c, p, L, views, _components(views, "ar" if p.rtl else "en"), stamp, page + 1, pages)


# The components (bill of materials): one table of the main parts (no hardware), grouped (the body, doors, drawers,
# shelves). Alike parts (same name and material, sizes within 1.5 cm) are one row with their count and, when they
# differ a little, the range of each size. Each row: its picture, name, size (estimated from the 3D model, or confirmed:
# set by the customer), the material the customer chose, and how many. What was not chosen is "to be confirmed with
# the customer", never guessed.
GROUP_ORDER = ["structure", "doors", "drawers", "shelves", "accessories"]
WOOD_TYPES = {
    "en": {"oak": "Oak", "beech": "Beech", "pine": "Pine", "walnut": "Walnut", "teak": "Teak", "ash": "Ash", "mahogany": "Mahogany", "mdf": "MDF", "hdf": "HDF",
           "plywood": "Plywood", "particleboard": "Particleboard", "blockboard": "Blockboard", "melamine": "Melamine", "veneer": "Veneer", "laminate": "Laminate / HPL", "lacquer": "Clear lacquer"},
    "ar": {"oak": "بلوط", "beech": "زان", "pine": "صنوبر", "walnut": "جوز", "teak": "تك", "ash": "دردار", "mahogany": "ماهوجني", "mdf": "MDF", "hdf": "HDF",
           "plywood": "أبلكاش", "particleboard": "خشب حبيبي", "blockboard": "بلوك بورد", "melamine": "ميلامين", "veneer": "قشرة", "laminate": "فورمايكا / HPL", "lacquer": "ورنيش شفاف"},
}


SIZES = ("width", "height", "depth", "thickness")
SAME_WITHIN = 1.5  # cm: alike parts (measured from a 3D model, alike pieces differ by about this much)


def _components(meta: dict[str, Any] | None, lang: Lang) -> list[dict[str, Any]]:
    """The table's rows, by group: alike parts once, their count summed and each size's range (`lo` / `hi`)."""
    rows: list[dict[str, Any]] = []
    chosen = (meta or {}).get("_materials") or {}
    for r in (meta or {}).get("parts") or []:
        if not r.get("main", r.get("shown", True)):
            continue  # hardware and small fittings
        r = _with_material(r, chosen.get(r.get("id")))
        name = (r.get("main_name_ar") or r.get("name_ar") or r.get("name")) if lang == "ar" else (r.get("main_name") or r.get("name"))
        kind = (name, *(r.get(k) for k in ("kind", "material", "board", "board_mm", "surface", "paint", "texture")), bool(r.get("added")), r.get("group"))
        sizes = {k: float(r.get(k) or 0) for k in SIZES}
        same = next((x for x in rows if x["_kind"] == kind and all(abs(sizes[k] - x["lo"][k]) <= SAME_WITHIN and abs(sizes[k] - x["hi"][k]) <= SAME_WITHIN for k in SIZES)), None)
        if same:
            same["quantity"] += int(r.get("quantity") or 1)
            for k in SIZES:
                same["lo"][k], same["hi"][k] = min(same["lo"][k], sizes[k]), max(same["hi"][k], sizes[k])
        else:
            rows.append({**r, "_kind": kind, "label": name or r.get("id") or "", "quantity": int(r.get("quantity") or 1), "lo": dict(sizes), "hi": dict(sizes)})
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(rows, key=lambda r: rank.get(r.get("group") or "structure", 0))


def _with_material(row: dict[str, Any], edit: dict[str, Any] | None) -> dict[str, Any]:
    """The part with the material saved on it in the editor (the request's own, kept current by every save of the
    design), over what the pictures recorded when they were made."""
    material = (edit or {}).get("material") or {}
    if not material.get("kind"):
        return row
    finish = material.get("finish") or {}
    return {**row, "kind": material["kind"], "material": material.get("name"), "board": material.get("wood_type"), "board_mm": material.get("thickness_mm"),
            "surface": finish.get("name"), "surface_type": finish.get("wood_type"), "paint": material.get("color_name") if material["kind"] == "paint" else None,
            "texture": edit.get("mapUrl") or row.get("texture"), "color": edit.get("color") or row.get("color")}


def _wood(row: dict[str, Any], L: dict[str, str], lang: Lang, model_wood: dict[str, Any] | None = None) -> str:
    """The wood of a part, as chosen in the editor: its name and board type (MDF, plywood, oak…), the board's thickness
    and the surface on it. A part without its own takes the wood the whole model is made of; a painted or upholstered
    part says so. Nothing chosen: to be confirmed with the customer."""
    types = WOOD_TYPES[lang]
    kind = row.get("kind")
    if kind and kind != "wood":
        text = f"{L.get(f'kind_{kind}', kind)}: {row.get('material') or L['tbc']}"
        return text + (f" · {row['paint']}" if row.get("paint") else "")
    wood = {"name": row.get("material"), "wood_type": row.get("board"), "thickness_mm": row.get("board_mm"),
            "finish": {"name": row.get("surface"), "wood_type": row.get("surface_type")} if row.get("surface") else None} if kind == "wood" else (model_wood or {})
    if not wood.get("name") and not wood.get("wood_type"):
        return L["tbc"]
    board = types.get(wood.get("wood_type") or "", wood.get("wood_type") or "")
    text = wood.get("name") or board
    if board and board != text:
        text += f" ({board})"
    if wood.get("thickness_mm"):
        text += f" · {_num(wood['thickness_mm'])} {L['mm']}"
    surface = wood.get("finish") or {}
    if surface.get("name"):
        text += f" · {L['surface']}: {surface['name']}" + (f" ({types.get(surface['wood_type'], surface['wood_type'])})" if surface.get("wood_type") else "")
    return text


def _span(row: dict[str, Any], key: str) -> str:
    """A size, or its range when alike parts differ a little ("36–37.5")."""
    lo, hi = row.get("lo", {}).get(key, row.get(key) or 0), row.get("hi", {}).get(key, row.get(key) or 0)
    if not hi:
        return "—"
    return _num(lo) if abs(hi - lo) < 0.05 else f"{_num(lo)}–{_num(hi)}"


B_COLS = [("img", 44), ("name", 86), ("qty", 30), ("w", 54), ("h", 52), ("d", 52), ("t", 34), ("material", 93)]  # the table's 445 pt
B_TOP = H - M - 72  # the column titles' top (under the title and the key)
B_BOTTOM = M + 34
B_HEAD = 24
B_GROUP = 38  # a group's title, with room above and below
B_IMG = 34
B_PAD = 12  # a row's padding above and below


def _b_lines(row: dict[str, Any], L: dict[str, str], lang: Lang, model_wood: dict[str, Any] | None) -> tuple[list[str], list[str]]:
    width = dict(B_COLS)
    return wrap(row["label"], "Sans-Bold", 8.5, width["name"] - 12, 4), wrap(_wood(row, L, lang, model_wood), "Sans", 7.5, width["material"] - 24, 5)


def _b_height(row: dict[str, Any], L: dict[str, str], lang: Lang, model_wood: dict[str, Any] | None) -> float:
    name, material = _b_lines(row, L, lang, model_wood)
    return 2 * B_PAD + max(B_IMG, 15 * len(name) + 20, 13 * len(material))  # lines at about 1.7 times their size


def _bom_pages(rows: list[dict[str, Any]], lang: Lang = "ar", model_wood: dict[str, Any] | None = None) -> list[list[tuple[str, Any]]]:
    """Pages of ("group", name) and ("row", part) items: a group's title never ends a page alone; rows never split."""
    L = LABELS[lang]
    pages: list[list[tuple[str, Any]]] = [[]]
    y = B_TOP - B_HEAD - 56  # the size summary on the first page
    group = None
    for row in rows:
        h = _b_height(row, L, lang, model_wood)
        need = h + (B_GROUP if row.get("group") != group else 0)
        if y - need < B_BOTTOM and pages[-1]:
            pages.append([])
            y = B_TOP - B_HEAD
            group = None
        if row.get("group") != group:
            group = row.get("group")
            pages[-1].append(("group", group))
            y -= B_GROUP
        pages[-1].append(("row", row))
        y -= h
    return pages


_component_pages = _bom_pages  # the pages the views section counts


def _picture_of(sheet: Any, row: dict[str, Any], size: int, cols: int) -> Any:
    pic = row.get("pic")
    if sheet is None or pic is None:
        return None
    x, y = int(pic) % cols * size, int(pic) // cols * size
    if y + size > sheet.height or x + size > sheet.width:
        return None
    return _compact(sheet.crop((x, y, x + size, y + size)))


def _bom(c: Canvas, p: Page, L: dict[str, str], views: dict[str, Any], rows: list[dict[str, Any]], stamp: str, first: int, pages: int) -> None:
    from PIL import Image

    lang: Lang = "ar" if p.rtl else "en"
    sheet_bytes = views.get("_pictures")
    sheet = Image.open(io.BytesIO(sheet_bytes)).convert("RGB") if sheet_bytes else None
    size, cols = int(views.get("pic_size") or 512), int(views.get("pic_cols") or 8)
    textures = views.get("_textures") or {}
    model_wood = views.get("_model_wood")
    known = views.get("scale_known", True)
    content = W - 2 * M
    tx, tw = M + PAD, content - 2 * PAD
    groups = {g: L[f"g_{g}"] for g in GROUP_ORDER}
    heads = {"img": "", "name": L["c_name"], "qty": L["c_qty"], "w": L["c_w"], "h": L["c_h"], "d": L["c_d"], "t": L["c_t"], "material": L["c_material"]}
    for index, items in enumerate(_bom_pages(rows, lang, model_wood)):
        c.showPage()
        _page_background(c)
        top = H - M
        p.card(M, top, content, top - (M + 20))
        p.heading(M + PAD, top - 26, L["components"])
        for n, line in enumerate(wrap(L["key"], "Sans", 7, tw, 2)):
            p.text(M + PAD, top - 44 - n * 12, line, 7, color=MUTED)
        y = B_TOP
        if index == 0:
            # The whole model: the size asked for (confirmed) and the 3D model's (the parts are measured from it).
            d = views.get("_requested") or {}
            o = views.get("overall") or {}
            half = (tw - 8) / 2
            p.box(tx, y - 40, half, 40, fill=CREAM, stroke=HexColor("#F3D9AE"), radius=8)
            p.text(tx + 12, y - 15, fit(L["requested"], "Sans-Bold", 7, half - 24), 7, bold=True, color=GOLD)
            p.text(tx + 12, y - 31, f"{_cm(d.get('width'))} × {_cm(d.get('height'))} × {_cm(d.get('depth'))} {L['cm']}", 9.5, bold=True, color=NAVY)
            p.box(tx + half + 8, y - 40, half, 40, fill=PAGE, stroke=LINE, radius=8)
            p.text(tx + half + 20, y - 15, fit(L["measured"], "Sans-Bold", 7, half - 24), 7, bold=True, color=MUTED)
            p.text(tx + half + 20, y - 31, f"≈ {_num(o.get('width'))} × {_num(o.get('height'))} × {_num(o.get('depth'))} {L['cm']}" if o else L["tbc"], 9.5, bold=True, color=NAVY)
            y -= 56
        # The column titles, on every page.
        p.box(tx, y - B_HEAD, tw, B_HEAD, fill=NAVY, stroke=None, radius=5)
        cx = tx
        for key, width in B_COLS:
            if heads[key]:
                p.text(cx + 6, y - 15.5, fit(heads[key], "Sans-Bold", 7, width - 10), 7, bold=True, color=WHITE)
            cx += width
        y -= B_HEAD
        stripe = 0
        for kind, item in items:
            if kind == "group":
                count = sum(int(r.get("quantity") or 1) for r in rows if r.get("group") == item)
                # Room above (apart from the rows before) and below; an amber bar and a fine line under it.
                p.box(tx, y - 28, 3, 14, fill=AMBER, stroke=None, radius=1.5)
                p.text(tx + 10, y - 25, f"{groups.get(item or 'structure', '')} · {count}", 10, bold=True, color=NAVY)
                c.setStrokeColor(HexColor("#F3D9AE"))
                c.setLineWidth(0.6)
                c.line(p.x(tx), y - B_GROUP + 4, p.x(tx + tw), y - B_GROUP + 4)
                y -= B_GROUP
                stripe = 0
                continue
            row = item
            h = _b_height(row, L, lang, model_wood)
            added = bool(row.get("added"))
            p.box(tx, y - h, tw, h, fill=CREAM if added else (PAGE if stripe % 2 else WHITE), stroke=None, radius=4)
            stripe += 1
            name_lines, material_lines = _b_lines(row, L, lang, model_wood)
            cx = tx
            for key, width in B_COLS:
                if key == "img":
                    p.box(cx + 4, y - B_PAD - B_IMG, B_IMG, B_IMG, fill=WHITE, stroke=LINE, radius=5)
                    image = _picture_of(sheet, row, size, cols)
                    if image is not None:
                        c.drawImage(image, p.x(cx + 5, B_IMG - 2), y - B_PAD - 1 - (B_IMG - 2), B_IMG - 2, B_IMG - 2)
                elif key == "name":
                    ny = y - B_PAD - 9
                    for line in name_lines:
                        p.text(cx + 5, ny, line, 8.5, bold=True, color=NAVY)
                        ny -= 15
                    tag = (L["confirmed"] if added else L["estimated"]) if known else L["tbc"]
                    p.pill(cx + 6, ny - 3, tag, fill=MINT if added and known else SOFT, color=GREEN if added and known else MUTED)
                elif key == "qty":
                    p.pill(cx + 4, y - B_PAD - 9, f"×{int(row.get('quantity') or 1)}", fill=CREAM, color=GOLD)
                elif key in ("w", "h", "d", "t"):
                    k = {"w": "width", "h": "height", "d": "depth", "t": "thickness"}[key]
                    value = _span(row, k)
                    if key == "t" and row.get("board_mm"):
                        value = _num(row["board_mm"] / 10)  # the board chosen: its thickness
                    # Never cut: a long range ("164.9–166.3") is set a little smaller to fit its column.
                    font = 8.5
                    while font > 7 and pdfmetrics.stringWidth(value, "Sans-Bold", font) > width - 10:
                        font -= 0.25
                    p.text(cx + 6, y - B_PAD - 9, value, font, bold=True, color=NAVY if value != "—" else MUTED)
                elif key == "material":
                    swatch = textures.get(row.get("texture") or "")
                    color = row.get("color")
                    sx = cx + 5
                    if swatch:
                        try:
                            c.drawImage(ImageReader(io.BytesIO(swatch)), p.x(sx, 12), y - B_PAD - 11, 12, 12, mask="auto")
                        except Exception:  # noqa: BLE001 - a broken texture leaves the swatch out
                            pass
                    elif isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color) and (row.get("kind") or model_wood):
                        p.box(sx, y - B_PAD - 11, 12, 12, fill=HexColor(color), stroke=LINE, radius=6)
                    my = y - B_PAD - 9
                    for line in material_lines:
                        p.text(sx + 17, my, line, 7.5, color=NAVY if line != L["tbc"] else MUTED)
                        my -= 13
                cx += width
            c.setStrokeColor(LINE)
            c.setLineWidth(0.4)
            c.line(p.x(tx + 4), y - h, p.x(tx + tw - 4), y - h)
            y -= h
        _footer(c, p, L, stamp, first + index, pages)


def _num(value: Any) -> str:
    v = float(value or 0)
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"



def build(task: dict[str, Any], picture: bytes | None, lang: Lang = "ar", now: datetime | None = None, download: bool = True, customer: str | None = None, views: tuple[bytes, ...] | None = None,
          pictures: bytes | None = None, textures: dict[str, bytes] | None = None) -> bytes:
    """The PDF (bytes). `picture`: the model's thumbnail; `customer`: the name of who made the request;
    `download`: the 3D file link is included (the customer, admins, the vendor whose offer was accepted); `views`: the
    closed and opened pictures
    (design.views' files), drawn with design.views' parts table on the pages after the first.

    Cards on a light gray page: the header (logo, report name, reference, status), the overview (picture beside the
    summary), the customer's notes, and the footer."""
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

    stamp = (now or datetime.now(timezone.utc)).strftime("%d/%m/%Y %H:%M UTC")
    dims = design.get("dimensions") or {}
    source = design.get("source") or {}
    wood = next(((m.get("material") or {}) for m in design.get("materials") or [] if (m.get("material") or {}).get("kind") == "wood"), None)

    # Banner: the logo at the start; at the end the report's name and its reference, date and time on a badge.
    top = H - M
    p.box(M, top - 64, content, 64, fill=WHITE, stroke=LINE, radius=12)
    p.box(M, top - 64, content, 3, fill=AMBER, stroke=None, radius=1.5)  # the brand's amber edge
    logo(c, p.x(M + PAD, 112), top - 45)
    badge_w = 190
    bx = M + content - PAD - badge_w
    p.box(bx, top - 54, badge_w, 44, fill=PAGE, stroke=LINE, radius=8)
    p.text(bx + 12, top - 24, L["report"], 9, bold=True, color=NAVY)
    ref = f"{L['ref']} {'' if p.rtl else '#'}{str(task.get('id', ''))[:8].upper()} · {_date(task.get('created_at'))}"  # "#" beside Arabic is reordered
    p.text(bx + 12, top - 37, ref, 7.5, color=MUTED)
    p.text(bx + 12, top - 48, L["generated"].format(at=stamp), 6.5, color=MUTED)

    # The title, its status and who asked.
    top -= 64 + GAP + 16  # the title's line
    title = task.get("title") or design.get("title") or ""
    status = L.get(str(task.get("status")), str(task.get("status") or "—"))
    title_w = min(pdfmetrics.stringWidth(shaped(title), "Sans-Bold", 19), content - 120)
    p.text(M, top, fit(title, "Sans-Bold", 19, content - 120), 19, bold=True, color=NAVY)
    p.pill(M + title_w + 12, top + 2, status, fill=CREAM, color=GOLD)
    model = design.get("title")
    sub = [f"{L['by']}: {customer}"] if customer else []
    sub += [f"{L['model']}: {model}"] if model and model != task.get("title") else []
    if sub:
        p.text(M, top - 20, fit(" · ".join(sub), "Sans", 8.5, content), 8.5, color=MUTED)
    top -= 34 + GAP

    # The picture across the page, then four summary tiles: the budget, the size and quantity, the timeline and
    # place, the wood.
    pic_h = 146
    p.card(M, top, content, pic_h)
    _picture(c, p, picture, M + 10, top - pic_h + 10, content - 20, pic_h - 20, L["no_picture"])
    top -= pic_h + GAP
    gx = M
    tile_gap = 16
    tile_w, tile_h = (content - tile_gap) / 2, 86

    def tile(col: int, row: int, label: str, accent: bool = False) -> tuple[float, float, float]:
        """A summary tile (its text start x, label's y and text width), with its label written."""
        x, y = gx + col * (tile_w + tile_gap), top - row * (tile_h + tile_gap)
        p.box(x, y - tile_h, tile_w, tile_h, fill=CREAM if accent else WHITE, stroke=HexColor("#F3D9AE") if accent else LINE, radius=10)
        p.text(x + PAD, y - PAD - 6, label, 7.5, bold=True, color=GOLD if accent else MUTED)
        return x + PAD, y - PAD - 6, tile_w - 2 * PAD

    tx, ty, tw_ = tile(0, 0, L["budget"], accent=True)
    p.text(tx, ty - 30, fit(_money(task.get("budget_max"), lang), "Sans-Bold", 20, tw_), 20, bold=True, color=GOLD)
    p.text(tx, ty - 50, f"{L['quantity']}: {spec.get('quantity') or 1}", 8.5, bold=True, color=NAVY)

    tx, ty, tw_ = tile(1, 0, L["size"])
    p.text(tx, ty - 28, fit(f"{_cm(dims.get('width'))} × {_cm(dims.get('height'))} × {_cm(dims.get('depth'))} {L['cm']}", "Sans-Bold", 14, tw_), 14, bold=True, color=NAVY)
    measured = ((design.get("views") or {}).get("overall") or {}) if views else {}
    mode = L[design["resize_mode"]] if design.get("resize_mode") in ("proportional", "independent") else ""
    extra = []
    if measured and any(abs(float(measured.get(k) or 0) - float(dims.get(k) or 0) * 100) >= 1 for k in ("width", "height", "depth")):
        extra.append(f"{L['model_size']}: ≈{_num(measured.get('width'))} × {_num(measured.get('height'))} × {_num(measured.get('depth'))}")
    if mode:
        extra.append(mode)
    for n, line in enumerate([ln for part in extra for ln in wrap(part, "Sans", 7.5, tw_, 2)][:2]):
        p.text(tx, ty - 47 - n * 13, line, 7.5, color=MUTED)

    tx, ty, tw_ = tile(0, 1, f"{L['created']} · {L['deadline']} · {L['location']}")
    for n, (kind, value) in enumerate((("calendar", f"{L['created']}: {_date(task.get('created_at'))}"), ("calendar", f"{L['deadline']}: {_date(task.get('deadline'))}"), ("pin", task.get("location") or "—"))):
        p.icon(kind, tx, ty - 22 - n * 17 - 1, color=AMBER)
        p.text(tx + 16, ty - 22 - n * 17, fit(value, "Sans-Bold", 8.5, tw_ - 16), 8.5, bold=True, color=NAVY)

    tx, ty, tw_ = tile(1, 1, L["wood"])
    types = WOOD_TYPES[lang]
    if wood:
        board = types.get(wood.get("wood_type") or "", wood.get("wood_type") or "")
        p.text(tx, ty - 25, fit(wood.get("name") or board or L["tbc"], "Sans-Bold", 11, tw_), 11, bold=True, color=NAVY)
        details = " · ".join(x for x in (board if board != wood.get("name") else "", f"{_num(wood['thickness_mm'])} {L['mm']}" if wood.get("thickness_mm") else "",
                                         ((wood.get("finish") or {}).get("name") or "")) if x)
        if details:
            p.text(tx, ty - 42, fit(details, "Sans", 8, tw_), 8, color=MUTED)
    else:
        p.text(tx, ty - 25, fit(L["tbc"], "Sans-Bold", 10, tw_), 10, bold=True, color=MUTED)
    audience = (spec.get("vendor") or {}).get("name") if task.get("audience") == "vendor_exclusive" else L["open"]
    p.text(tx, ty - tile_h + PAD + 12, fit(f"{L['audience']}: {audience or '—'}", "Sans", 7, tw_), 7, color=MUTED)

    top -= 2 * tile_h + tile_gap + GAP
    if design.get("glb_url") and download:
        # The 3D file, a link.
        p.box(M, top - 34, content, 34, fill=WHITE, stroke=LINE, radius=8)
        p.text(M + PAD, top - 21, f"{L['file']}:", 8, color=MUTED)
        lx = M + PAD + pdfmetrics.stringWidth(shaped(L["file"] + ":"), "Sans", 8) + 6
        p.text(lx, top - 21, L["download"], 8.5, bold=True, color=GOLD)
        link_w = pdfmetrics.stringWidth(shaped(L["download"]), "Sans-Bold", 8.5)
        c.linkURL(design["glb_url"], (p.x(lx, link_w), top - 25, p.x(lx, link_w) + link_w, top - 11), relative=0)
        top -= 34 + GAP

    # The customer's notes, in a callout across the page.
    room = top - (M + 30) - GAP
    note = (task.get("description") or "").strip()
    lines = wrap(note, "Sans", 9, content - 2 * PAD - 24, max(1, int((room - 80) // 16))) if note else [L["none"]]
    call_h = 22 + 16 * len(lines)  # a 1.8 line height
    p.card(M, top, content, 44 + call_h + PAD)  # the card hugs its text
    p.heading(M + PAD, top - 26, L["notes"])
    p.box(M + PAD, top - 42 - call_h, content - 2 * PAD, call_h, fill=CREAM, stroke=None, radius=6)
    p.box(M + PAD, top - 42 - call_h, 3, call_h, fill=AMBER, stroke=None, radius=1.5)
    ny = top - 58
    for line in lines:
        p.text(M + PAD + 14, ny, line, 9, color=INK if note else MUTED)
        ny -= 16

    meta = design.get("views") if views else None
    if meta and meta.get("version", 1) < VIEWS_VERSION:
        meta, views = None, None  # numbered-marker pictures are never shown: the notice says they are being made again
    if meta:
        # The wood the whole model is made of (the editor's choice for its wooden parts), when there is one.
        model_wood = next(((m.get("material") or {}) for m in design.get("materials") or [] if (m.get("material") or {}).get("kind") == "wood"), None)
        materials = {m.get("part_id"): m for m in design.get("materials") or [] if m.get("part_id")}
        meta = {**meta, "_requested": design.get("dimensions") or {}, "_pictures": pictures, "_textures": textures or {}, "_model_wood": model_wood, "_materials": materials}
    pages = 1 + _view_pages(meta, lang)
    _footer(c, p, L, stamp, 1, pages)
    c.showPage()
    _views_pages(c, p, L, meta, views, stamp, 2, pages)
    c.showPage()
    c.save()
    return out.getvalue()
