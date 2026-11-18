"""PDF report of an implementation request (freelance_tasks row with its design spec), for the customer, the vendors
who may see it and admins. Page 1: the Sha2ty logo, the model's picture and size, the request's details, the 3D file
link and the customer's notes (no offers). Then the checks, then the 3D views on two pages: closed and open together,
the front/side/top sheet on its own. Component close-ups are three or four to a page when the dimension text stays
at least 7 pt, otherwise two. The cutting list ends with one status line that points to the checks.
Arabic or English; Arabic is laid out right to left. Page 1 is always one A4 page: long notes are cut short with "…".
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
M = 40  # page margin: 14 mm, enough to clear the footer without a blank band
PAD = 12  # inside a card
GAP = 10  # between cards and sections
ARABIC = re.compile(r"[؀-ۿݐ-ݿࢠ-ࣿ]")

LABELS: dict[str, dict[str, str]] = {
    "en": {
        "report": "Implementation request report", "ref": "Request", "created": "Created", "deadline": "Deadline", "budget": "Budget",
        "quantity": "Quantity", "location": "Location", "status": "Status", "audience": "Sent to", "open": "All approved vendors",
        "model": "Model", "by": "Requested by", "size": "Size (W × H × D)", "based": "Based on", "wood": "Wood (the whole model)",
        "file": "3D model file", "download": "Download GLB", "notes": "Customer notes", "none": "None",
        "generated": "Generated {at}",
        "page": "Page {i} of {n}", "no_picture": "No picture", "egp": "EGP", "cm": "cm",
        "views": "3D Model Views & Dimensions", "closed_view": "Closed — overall dimensions", "open_view": "Opened — every moving part open", "detail_view": "Detail {i}",
        "other_view": "Another angle — each part's dimensions (no movable parts to open)",
        "components": "Cutting list (cm)", "added": "Added", "wood_type": "Wood type", "mm": "mm", "surface": "Surface",
        "g_structure": "Structure", "g_doors": "Doors", "g_drawers": "Drawers", "g_shelves": "Shelves", "g_accessories": "Other parts", "key": "Length × width × thickness: the same edges as the drawings. Thickness is the 3D part; a different board is printed under it and is not substituted. Confirmed: the customer set it. 3D model: measured. Estimated: scale unknown.",
        "c_name": "Component", "c_w": "Width", "c_h": "Height", "c_d": "Depth", "c_t": "T", "c_len": "Length", "c_wid": "Width", "c_thk": "Thickness", "c_qty": "Qty", "c_status": "Dimensions", "c_material": "Material", "c_color": "Colour", "c_finish": "Finish",
        "requested": "The whole model (requested size)", "measured": "The 3D model as drawn", "model_size": "3D model", "confirmed": "Confirmed", "from_model": "3D model", "estimated": "Estimated",
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
        "views": "مناظر الموديل ثلاثية الأبعاد والمقاسات", "closed_view": "مغلق — المقاسات الكلية", "open_view": "مفتوح — كل الأجزاء المتحركة مفتوحة", "detail_view": "تفاصيل {i}",
        "other_view": "زاوية أخرى — مقاسات كل جزء (لا توجد أجزاء متحركة)",
        "components": "قائمة القص (سم)", "added": "مضاف", "wood_type": "نوع الخشب", "mm": "مم", "surface": "السطح",
        "g_structure": "الهيكل", "g_doors": "الأبواب", "g_drawers": "الأدراج", "g_shelves": "الأرفف", "g_accessories": "أجزاء أخرى", "key": "الطول × العرض × السُمك: نفس الأضلاع في الرسومات. السُمك من الموديل، ولوح مختلف يُطبع تحته ولا يُستبدل. مؤكد: حدده العميل. من الموديل: مقاس. تقديري: المقياس غير معروف.",
        "c_name": "المكون", "c_w": "العرض", "c_h": "الارتفاع", "c_d": "العمق", "c_t": "السُمك", "c_len": "الطول", "c_wid": "العرض", "c_thk": "السُمك", "c_qty": "العدد", "c_status": "المقاسات", "c_material": "الخامة", "c_color": "اللون", "c_finish": "التشطيب",
        "requested": "الموديل بالكامل (المقاس المطلوب)", "measured": "الموديل ثلاثي الأبعاد كما رُسم", "model_size": "الموديل ثلاثي الأبعاد", "confirmed": "مؤكد", "from_model": "من الموديل", "estimated": "تقديري",
        "tbc": "يحدد مع العميل", "kind_wood": "خشب", "kind_paint": "دهان", "kind_fabric": "قماش", "kind_metal": "معدن", "kind_glass": "زجاج",
        "matte": "مطفي", "satin": "نصف لامع", "gloss": "لامع", "pictured": "كالصورة",
        "no_views": "صور الموديل بمقاساته غير جاهزة لهذا الطلب بعد: تُنشأ عندما يفتح العميل الطلب. نزّل التقرير مرة أخرى بعد ذلك.",
        "published": "متاح للعروض", "awarded": "تم الترسية", "in_progress": "قيد التنفيذ", "completed": "مكتمل", "cancelled": "ملغي",
        "proportional": "تغيير المقاس بالتناسب", "independent": "تغيير كل بُعد على حدة",
    },
}


# The front / side / top views and the measurement checks page.
LABELS["en"].update({
    "ortho_view": "Front, side and top views — overall dimensions", "checks": "Dimension checks",
    "k_requested": "Requested (customer)", "k_model": "3D model (measured)", "k_diff": "Difference",
    "k_note": "Cutting sizes are measured from the same 3D model geometry as the pictures. Thickness is that model's thinnest side; a different chosen board is a critical issue below and is not substituted. Sizes in cm.",
    "k_none": "No problems found: every part fits the model and the model matches the requested size.",
    "k_old": "These pictures were made before the checks: download the report again after the customer opens the request.",
    "k_also": "Also: {list}.", "k_pair": "{a} and {b}", "k_sep": "; ",
    "k_critical": "Critical", "k_warning": "Warning", "k_info": "Info", "width": "Width", "height": "Height", "depth": "Depth",
    "k_ready": "Ready for Manufacturing: no critical issues. Check the warnings with the customer.",
    "k_not_ready": "Requires Confirmation: {n} critical issue(s) must be confirmed or fixed before manufacturing.",
    "k_status": "Manufacturing status",
    "k_status_line": "Requires Confirmation: {n} critical issue(s), listed on page 2.",
    "k_status_ready": "Ready for Manufacturing. The check is on page 2.",
    "k_status_old": "These pictures have no check yet. See page 2.",
    "c_substrate": "{a}: finish {finish} is set, and the board under it is not. Confirm the substrate.",
    "c_joint": "{a} and {b}: one goes {depth} cm into the other, as a groove or housing joint would. Confirm the joint type.",
    "c_nomaterial": "No material chosen for: {list}. Confirm with the customer.",
    "c_board": "{a}: the 3D model is {model} cm thick and the chosen board is {board} cm. Confirm which to cut. Neither is substituted.",
    "c_mirror": "{a}: {model} cm in the 3D model. Confirm whether this is the glass or mirror, or the wood panel behind it. Neither size is changed.",
    "c_mirror_glass": "{a}: {model} cm is the glass or mirror, not the wood panel behind it.",
    "c_mirror_wood": "{a}: {model} cm is the wood panel. The mirror glass is a separate piece; confirm its size.",
    "board_spec": "Board", "model_thk": "3D",
    "cat_solid": "Solid wood", "cat_board": "Engineered board", "cat_plywood": "Plywood", "cat_glass": "Glass", "cat_mirror": "Mirror", "cat_metal": "Metal", "cat_fabric": "Fabric", "cat_paint": "Painted",
    "a_suggest": "Suggested: ",
    "a_edge4": "Edge band: all 4 edges", "a_edge_front": "Edge band: front edge", "a_polished": "Polished edges",
    "a_hinges": "{n} hinges", "a_slides": "Drawer slides, box depth {n} cm",
    "a_glass": "This size is the glass or mirror, not the wood behind it",
    "a_support": "Wood panel behind the mirror; the glass is cut separately",
    "a_mirror_ask": "Confirm glass or wood before cutting; the 3D size is kept",
    "c_approx": "The model's real size is not known: every dimension is approximate (≈) and must be confirmed with the customer.",
    "c_mismatch": "{axis}: the customer asked for {requested} cm, the 3D model is {model} cm (difference {diff} cm). Confirm which one to make.",
    "c_units": "The model's overall size ({width} × {height} × {depth} cm) is unusual for furniture: check its units.",
    "c_missing": "{a}: a dimension is missing (0 cm) in the 3D model.",
    "c_exceeds": "{a}: its {axis} ({value} cm) is larger than the 3D model ({overall} cm). The part is not resized.",
    "c_over_request": "{a}: its {axis} is {value} cm, larger than the size the customer asked for ({overall} cm). The part is not resized.",
    "c_axis": "{a}: its thinnest side is not its {expected}: check how it stands in the model.",
    "c_overlap": "{a} and {b} go {depth} cm into each other: their sizes do not fit together as modelled.",
    "c_clearance": "{a} and {b}: measured gap {gap} cm. Confirm the door clearance before cutting. The gap is not changed.",
    "c_clearance_all": "Door gaps, each measured: {list}. Confirm each before cutting. No gap is changed.",
    "c_drawer": "{a} and {b}: measured gap {gap} cm. Confirm the fit before cutting. The gap is not changed.",
    "c_drawer_all": "Drawer gaps, each measured: {list}. Confirm each fit before cutting. No gap is changed.",
    "c_drawer_box": "{list}: the model has the drawer and no separate box (sides, bottom, back). Those sizes are not in the model and are not guessed.",
    "c_shelf": "{a} and {b}: gap {gap} cm, so the shelf does not meet the panel. Confirm whether it should be housed.",
    "c_shelf_all": "Shelf gaps, each measured: {list}. Confirm whether each shelf should meet the panel. No size is changed.",
})
LABELS["ar"].update({
    "ortho_view": "مناظر أمامي وجانبي وعلوي — المقاسات الكلية", "checks": "مراجعة المقاسات",
    "k_requested": "المطلوب (العميل)", "k_model": "الموديل ثلاثي الأبعاد (مقاس)", "k_diff": "الفرق",
    "k_note": "مقاسات القص من نفس هندسة الموديل ثلاثي الأبعاد مثل الصور. السُمك هو أنحف ضلع في ذلك الموديل؛ ولوح مختار بسُمك مختلف مشكلة حرجة أدناه ولا يُستبدل. المقاسات بالسنتيمتر.",
    "k_none": "لا توجد مشكلات: كل الأجزاء متوافقة مع الموديل، والموديل مطابق للمقاس المطلوب.",
    "k_old": "هذه الصور أُنشئت قبل مراجعة المقاسات: نزّل التقرير مرة أخرى بعد أن يفتح العميل الطلب.",
    "k_also": "وأيضًا: {list}.", "k_pair": "{a} و{b}", "k_sep": "؛ ",
    "k_critical": "حرج", "k_warning": "تحذير", "k_info": "معلومة", "width": "العرض", "height": "الارتفاع", "depth": "العمق",
    "k_ready": "جاهز للتصنيع: لا توجد مشكلات حرجة. راجع التحذيرات مع العميل.",
    "k_not_ready": "يتطلب التأكيد — غير جاهز للتصنيع: {n} مشكلة حرجة يجب تأكيدها أو تصحيحها أولًا.",
    "k_status": "حالة التصنيع",
    "k_status_line": "يتطلب التأكيد — غير جاهز للتصنيع: {n} مشكلة حرجة، مفصّلة في صفحة 2.",
    "k_status_ready": "جاهز للتصنيع. المراجعة في صفحة 2.",
    "k_status_old": "هذه الصور بلا مراجعة بعد. انظر صفحة 2.",
    "c_substrate": "{a}: التشطيب {finish} محدد، واللوح تحته غير محدد. أكّد الخامة الأساسية.",
    "c_joint": "{a} و{b}: أحدهما داخل الآخر {depth} سم كما في وصلة مجرى أو تجويف. أكّد نوع الوصلة.",
    "c_nomaterial": "لم تُحدد الخامة لـ: {list}. أكّدها مع العميل.",
    "c_board": "{a}: سُمك الموديل {model} سم وسُمك اللوح المختار {board} سم. أكّد أيهما يُقص. لا يُستبدل أحدهما بالآخر.",
    "c_mirror": "{a}: {model} سم في الموديل. أكّد إن كان هذا الزجاج أو المرآة، أم لوح الخشب خلفها. لا يُغيَّر أي مقاس.",
    "c_mirror_glass": "{a}: {model} سم هي الزجاج أو المرآة، وليس لوح الخشب خلفها.",
    "c_mirror_wood": "{a}: {model} سم هي لوح الخشب. زجاج المرآة قطعة مستقلة؛ أكّد مقاسه.",
    "board_spec": "اللوح", "model_thk": "3D",
    "cat_solid": "خشب طبيعي", "cat_board": "خشب مُصنَّع", "cat_plywood": "أبلكاش", "cat_glass": "زجاج", "cat_mirror": "مرآة", "cat_metal": "معدن", "cat_fabric": "قماش", "cat_paint": "دهان",
    "a_suggest": "مقترح: ",
    "a_edge4": "شريط حواف: الحواف الأربعة", "a_edge_front": "شريط حواف: الحافة الأمامية", "a_polished": "حواف مصقولة",
    "a_hinges": "{n} مفصلات", "a_slides": "مجاري أدراج، عمق الصندوق {n} سم",
    "a_glass": "هذا المقاس للزجاج أو المرآة، وليس للخشب خلفها",
    "a_support": "لوح خشب خلف المرآة؛ الزجاج يُقص قطعة مستقلة",
    "a_mirror_ask": "أكّد الزجاج أو الخشب قبل القص؛ مقاس الموديل يبقى كما هو",
    "c_approx": "المقاس الحقيقي للموديل غير معروف: كل المقاسات تقريبية (≈) ويجب تأكيدها مع العميل.",
    "c_mismatch": "{axis}: طلب العميل {requested} سم، والموديل ثلاثي الأبعاد {model} سم (الفرق {diff} سم). أكّد أيهما يُنفَّذ.",
    "c_units": "المقاس الكلي للموديل ({width} × {height} × {depth} سم) غير معتاد لقطعة أثاث: راجع وحدات القياس.",
    "c_missing": "{a}: أحد المقاسات غير موجود (0 سم) في الموديل ثلاثي الأبعاد.",
    "c_exceeds": "{a}: {axis} ({value} سم) أكبر من الموديل ثلاثي الأبعاد ({overall} سم). الجزء لا يُصغَّر.",
    "c_over_request": "{a}: {axis} {value} سم، أكبر من المقاس الذي طلبه العميل ({overall} سم). الجزء لا يُصغَّر.",
    "c_axis": "{a}: أنحف جانب فيه ليس {expected}: راجع وضعه في الموديل.",
    "c_overlap": "{a} و{b} متداخلان {depth} سم: مقاساتهما لا تتوافق كما في الموديل.",
    "c_clearance": "{a} و{b}: المسافة المقاسة {gap} سم. أكّد خلوص الباب قبل القص. المسافة لا تُغيَّر.",
    "c_clearance_all": "خلوص الأبواب، كل مسافة مقاسة: {list}. أكّد كل واحدة قبل القص. لا تُغيَّر مسافة.",
    "c_drawer": "{a} و{b}: المسافة المقاسة {gap} سم. أكّد التوافق قبل القص. المسافة لا تُغيَّر.",
    "c_drawer_all": "مسافات الأدراج، كل واحدة مقاسة: {list}. أكّد كل توافق قبل القص. لا تُغيَّر مسافة.",
    "c_drawer_box": "{list}: الموديل فيه الدرج، ولا يوجد صندوق مستقل (جوانب وقاع وظهر). هذه المقاسات ليست في الموديل ولا تُخمَّن.",
    "c_shelf": "{a} و{b}: مسافة {gap} سم، فالرف لا يصل إلى اللوح. أكّد إن كان يُركَّب في تجويف.",
    "c_shelf_all": "مسافات الأرفف، كل واحدة مقاسة: {list}. أكّد إن كان كل رف يصل إلى اللوح. لا يُغيَّر مقاس.",
})
RED, ROSE = HexColor("#B42318"), HexColor("#FEF3F2")


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
    """Centimetres from a size stored in metres, printed like every other size. A fraction is kept; it is not rounded to a whole centimetre."""
    return "—" if not value else _num(float(value) * 100)


def _requested_differs(requested_m: float, measured_cm: float) -> bool:
    """True when the 3D model and the customer's size disagree by more than 1 cm and more than 1% of the request.
    The same rule as the measurement check, so the first page and the checks page name the same conflict."""
    want = float(requested_m or 0) * 100
    got = float(measured_cm or 0)
    return bool(want and got and abs(got - want) > max(1, 0.01 * want))


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
            raw = _fitted(picture)[0]
            image = _compact(raw, 85 if frame else 90)
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


def _pixel_size(blob: bytes) -> tuple[int, int]:
    """The picture's pixel size after the blank margin is removed. A broken file is treated as a square capture."""
    try:
        return _fitted(blob)[1:]
    except Exception:  # noqa: BLE001
        return (2400, 2400)


def _fitted(blob: bytes) -> tuple[bytes, int, int]:
    """The picture with its blank margin cut off, and the size that remains. Dimension text sits on the drawing, so it
    stays. A page with one illustration then uses the page for the drawing instead of the white around it."""
    cache: dict = getattr(_fitted, "_cache", {})
    key = (len(blob), blob[:48], blob[-48:] if len(blob) > 48 else b"")
    if key in cache:
        return cache[key]
    from PIL import Image

    im = Image.open(io.BytesIO(blob)).convert("RGB")
    mask = im.convert("L").point(lambda p: 255 if p < 246 else 0)
    box = mask.getbbox()
    if box:
        x0, y0, x1, y1 = box
        pad = 16
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(im.width, x1 + pad), min(im.height, y1 + pad)
        if x1 - x0 < im.width * 0.98 or y1 - y0 < im.height * 0.98:
            im = im.crop((x0, y0, x1, y1))
    out = io.BytesIO()
    im.save(out, "PNG")
    result = (out.getvalue(), im.width, im.height)
    if len(cache) > 48:
        cache.clear()
    cache[key] = result
    _fitted._cache = cache  # type: ignore[attr-defined]
    return result


# Dimension text is drawn 54 px high on the capture (frontend captureViews TEXT.size). Below 7 pt on the page it is
# no longer something a carpenter can read, so a sheet that would go under that gets fewer pictures.
LABEL_PX = 54
MIN_LABEL_PT = 7.0
VIEW_TITLE = 24
VIEW_CAPTION = 20
VIEW_GAP = 6


def _view_width() -> float:
    return W - 2 * M - 16


def _view_room() -> float:
    """The height left for pictures under the one-line title and above the footer."""
    return H - 2 * M - VIEW_TITLE - 8 - 22


def _image_scale(sizes: list[tuple[int, int]], room: float, width: float) -> float:
    """Points per pixel for pictures stacked down the page. One scale, so every dimension label matches."""
    n = len(sizes)
    budget = room - n * VIEW_CAPTION - VIEW_GAP * (n - 1)
    if budget <= 0 or not sizes:
        return 0
    scale = budget / sum(ih for _, ih in sizes)
    return min(scale, *(width / iw for iw, _ in sizes if iw))


def _grid_scale(sizes: list[tuple[int, int]], room: float, width: float) -> float:
    """Points per pixel in two columns. Used when stacking would put the dimension text under 7 pt."""
    n = len(sizes)
    rows = (n + 1) // 2
    cell_w = (width - 8) / 2
    budget = room - rows * VIEW_CAPTION - VIEW_GAP * (rows - 1)
    cell_h = budget / rows if rows else 0
    if cell_w <= 0 or cell_h <= 0:
        return 0
    return min(min(cell_w / iw, cell_h / ih) for iw, ih in sizes if iw and ih)


def _label_ok(scale: float, sizes: list[tuple[int, int]]) -> bool:
    """The 54 px dimension text is still at least 7 pt, and each picture is large enough to use."""
    if scale <= 0 or any(ih * scale < 80 for _, ih in sizes):
        return False
    if any(max(w, h) >= 1000 for w, h in sizes) and LABEL_PX * scale < MIN_LABEL_PT:
        return False
    return True


def _placed(sizes: list[tuple[int, int]], room: float, width: float) -> tuple[str, float]:
    """'stack' down the page, or a two-column 'grid' when that is what keeps the dimension text readable."""
    stacked = _image_scale(sizes, room, width)
    if len(sizes) < 3 or _label_ok(stacked, sizes):
        return "stack", stacked
    grid = _grid_scale(sizes, room, width)
    if _label_ok(grid, sizes):
        return "grid", grid
    return "stack", stacked


def _sheet_ok(sizes: list[tuple[int, int]], room: float, width: float) -> bool:
    """True when this many pictures still leave the dimension text readable. A small fixture has no baked labels,
    so it is not used to justify more than two on a page."""
    if len(sizes) >= 3 and any(max(w, h) < 1000 for w, h in sizes):
        return False
    _, scale = _placed(sizes, room, width)
    return _label_ok(scale, sizes)


def _take(sizes: list[tuple[int, int]], room: float, width: float) -> int:
    """Four or three pictures share a page when each stays readable, otherwise two, otherwise one."""
    for n in (4, 3, 2):
        if n <= len(sizes) and _sheet_ok(sizes[:n], room, width):
            return n
    return 1


def _overview_sheets(full: int) -> list[list[int]]:
    """The whole-model pictures on two pages. Closed and open share one; front, side and top stay together on the
    other, because that sheet already holds three measured views. Two pictures stay one to a page."""
    if full <= 2:
        return [[i] for i in range(full)]
    return [[0, full - 1], [1]]


def _plan_sheets(images: tuple[bytes, ...] | list[bytes], ortho: bool) -> list[list[int]]:
    """Page groups of picture indexes. Overall views on two pages, then close-ups by category order, three or four
    to a page while the dimension text stays readable."""
    full = min(len(images), 2 + int(ortho))
    sizes = [_pixel_size(images[i]) for i in range(len(images))]
    sheets = _overview_sheets(full)
    room, width = _view_room(), _view_width()
    i = full
    while i < len(images):
        n = _take(sizes[i:], room, width)
        sheets.append(list(range(i, i + n)))
        i += n
    # A last page with one picture is mostly empty. Give it one or two from the page before when both stay readable.
    if len(sheets) >= 2 and len(sheets[-1]) == 1 and len(sheets[-2]) >= 3 and sheets[-2][0] >= full:
        prev, last = sheets[-2], sheets[-1]
        for give in (1, 2):
            if len(prev) - give < 2:
                continue
            left, right = prev[:-give], prev[-give:] + last
            if _sheet_ok([sizes[k] for k in left], room, width) and _sheet_ok([sizes[k] for k in right], room, width):
                sheets[-2], sheets[-1] = left, right
                break
    return sheets


def _view_pages(meta: dict[str, Any] | None, lang: Lang = "ar", images: tuple[bytes, ...] | None = None) -> int:
    """The picture pages and the cutting list's; else the notice's."""
    if not meta:
        return 1
    picture_pages = len(_plan_sheets(images, bool(meta.get("ortho_url")))) if images else 1
    return picture_pages + len(_bom_pages(_components(meta, lang), lang, meta.get("_model_wood"), _status_height(meta, lang), _cols(meta.get("_pictures"))))


def _views_pages(c: Canvas, p: Page, L: dict[str, str], views: dict[str, Any] | None, images: tuple[bytes, ...] | None, stamp: str, first: int, pages: int) -> None:
    """Overall views on two pages, then component pictures three or four to a page while the dimension text stays 7 pt."""
    content = W - 2 * M
    _page_background(c)
    top = H - M
    if not views or not images:
        p.card(M, top, content, 36)
        p.heading(M + PAD, top - 22, L["views"])
        top -= 36 + GAP
        p.card(M, top, content, 60)
        for n, line in enumerate(wrap(L["no_views"], "Sans", 9, content - 2 * PAD, 3)):
            p.text(M + PAD, top - 26 - n * 16, line, 9, color=MUTED)
        _footer(c, p, L, stamp, first, pages)
        return
    groups = views.get("detail_groups") or []
    names = {g: L.get(f"g_{g}", "") for g in GROUP_ORDER}
    ortho = [L["ortho_view"]] if views.get("ortho_url") else []
    captions = [L["closed_view"], *ortho, L["open_view"] if views.get("opened") else L["other_view"]] + [
        f"{L['detail_view'].format(i=i + 1)}" + (f" · {names[groups[i]]}" if i < len(groups) and names.get(groups[i]) else "") for i in range(len(images) - 2 - len(ortho))
    ]
    sheets = _plan_sheets(images, bool(ortho))
    room, width = _view_room(), _view_width()
    page = first
    for n, sheet in enumerate(sheets):
        if n:
            _footer(c, p, L, stamp, page, pages)
            c.showPage()
            _page_background(c)
            page += 1
        top = H - M
        p.text(M, top - 14, L["views"], 11, bold=True, color=NAVY)
        top -= VIEW_TITLE + 4
        sizes = [_pixel_size(images[i]) for i in sheet]
        mode, scale = _placed(sizes, room, width)
        if mode == "grid":
            cols, gap = 2, 8
            rows_n = (len(sheet) + 1) // 2
            cell_w = (content - gap) / cols
            cell_h = (room - rows_n * VIEW_CAPTION - VIEW_GAP * (rows_n - 1)) / rows_n
            for k, i in enumerate(sheet):
                row, col = divmod(k, cols)
                x = M + col * (cell_w + gap)
                y_top = top - row * (cell_h + VIEW_CAPTION + VIEW_GAP)
                p.box(x, y_top - 16, 3, 12, fill=AMBER, stroke=None, radius=1.5)
                p.text(x + 10, y_top - 14, fit(captions[i] if i < len(captions) else "", "Sans-Bold", 9, cell_w - 16), 9, bold=True, color=NAVY)
                _picture(c, p, images[i], x, y_top - VIEW_CAPTION - cell_h + 2, cell_w, cell_h, L["no_picture"], frame=False)
        else:
            for k, i in enumerate(sheet):
                iw, ih = sizes[k]
                slot = ih * scale + VIEW_CAPTION
                p.box(M, top - 16, 3, 12, fill=AMBER, stroke=None, radius=1.5)
                p.text(M + 10, top - 14, fit(captions[i] if i < len(captions) else "", "Sans-Bold", 10, content - 16), 10, bold=True, color=NAVY)
                _picture(c, p, images[i], M, top - slot + 2, content, slot - VIEW_CAPTION, L["no_picture"], frame=False)
                top -= slot + VIEW_GAP
    _footer(c, p, L, stamp, page, pages)
    _bom(c, p, L, views, _components(views, "ar" if p.rtl else "en"), stamp, page + 1, pages)


# The measurement checks (frontend lib/designs/measure.ts): the requested size beside the 3D model's, then what does
# not add up (parts larger than the model, overlapping, doors without clearance…), each in a line of its own. Sizes are
# never changed here, only reported. As many pages as the list needs.
K_ROW = 22  # a row of the requested / model table
K_LINE = 14


def _check_text(check: dict[str, Any], L: dict[str, str], names: dict[str, str]) -> str:
    items = check.get("_items") or []
    if len(items) > 1 and check.get("code") in _GAP_CODES:
        bits = []
        for ch in items:
            gap = (ch.get("params") or {}).get("gap")
            groups = [ch.get("parts") or []] + list(ch.get("_also") or [])
            for g in groups:
                pair = L["k_pair"].format(a=names.get(g[0], g[0]), b=names.get(g[1], g[1])) if len(g) > 1 else names.get(g[0], g[0]) if g else ""
                bits.append(f"{pair} ({_num(gap)} {L['cm']})" if isinstance(gap, (int, float)) else pair)
        try:
            return L[f"c_{check.get('code')}_all"].format(list=L["k_sep"].join(b for b in bits if b))
        except KeyError:
            return str(check.get("code"))
    template = L.get(f"c_{check.get('code')}", "")
    # An axis name starts a sentence ("Width: …") or sits inside one ("its width").
    word = lambda v: L.get(v, v) if template.startswith("{axis}") else L.get(v, v).lower()  # noqa: E731
    params = {k: (word(v) if k in ("axis", "expected") else _num(v) if isinstance(v, (int, float)) else v) for k, v in (check.get("params") or {}).items()}
    ids = check.get("parts") or []
    params.update(a=names.get(ids[0], ids[0]) if ids else "", b=names.get(ids[1], ids[1]) if len(ids) > 1 else "")
    try:
        text = L[f"c_{check.get('code')}"].format(**params)
    except (KeyError, IndexError):
        return str(check.get("code"))
    also = [L["k_pair"].format(a=names.get(g[0], g[0]), b=names.get(g[1], g[1])) if len(g) > 1 else names.get(g[0], g[0]) for g in check.get("_also") or []]
    return f"{text} {L['k_also'].format(list=L['k_sep'].join(also))}" if also else text


LEVELS = ("critical", "warning", "info")
# Gap warnings that say the same thing about different pairs. One line lists every measured gap.
_GAP_CODES = {"shelf", "drawer", "clearance"}


# A customer/model size conflict, a missing or impossible size, a board that does not match the part, or a mirror that
# cannot be told from the wood behind it: the report is not ready until it is confirmed. Stored as a warning on an
# older picture, these still block manufacturing.
BLOCKING = {"approx", "mismatch", "units", "missing", "exceeds", "over_request", "board", "mirror"}


def _level(check: dict[str, Any]) -> str:
    if check.get("code") in BLOCKING:
        return "critical"
    return "critical" if check.get("level") == "error" else check.get("level") or "warning"


def _findings(meta: dict[str, Any], lang: Lang) -> list[dict[str, Any]]:
    """Every finding: the measurement checks made with the pictures, then what the specification still lacks (read
    here, where the materials the customer saved since are known): parts without a material, a chosen board whose
    thickness is not the part's. Critical first."""
    L = LABELS[lang]
    rows = _components(meta, lang)
    model_wood = meta.get("_model_wood")
    found = list(meta.get("checks") or [])
    missing = [r["label"] for r in rows if _wood(r, L, lang, model_wood) == L["tbc"]]
    if missing:
        found.append({"level": "warning", "code": "nomaterial", "parts": [], "params": {"list": "، ".join(missing) if lang == "ar" else ", ".join(missing)}})
    for r in rows:
        board = float(r.get("board_mm") or 0) / 10
        model = _model_thickness(r)
        # Both thicknesses stay. A different chosen board is critical and is not written in as the cut.
        if board and model and abs(board - model) > 0.1:
            found.append({"level": "critical", "code": "board", "parts": [r.get("id")], "params": {"board": board, "model": model}})
        body, finish = _material_parts(r, L, lang, model_wood)
        if finish and body == L["tbc"]:
            found.append({"level": "warning", "code": "substrate", "parts": [r.get("id")], "params": {"finish": finish}})
        role = _mirror_role(r)
        if role == "unverified":
            found.append({"level": "critical", "code": "mirror", "parts": [r.get("id")], "params": {"model": model}})
        elif role == "glass":
            found.append({"level": "info", "code": "mirror_glass", "parts": [r.get("id")], "params": {"model": model}})
        elif role == "support":
            found.append({"level": "info", "code": "mirror_wood", "parts": [r.get("id")], "params": {"model": model}})
    drawers = [r for r in rows if r.get("group") == "drawers"]
    if drawers and not any(_drawer_box(r) for r in rows):
        names = [r["label"] for r in drawers]
        found.append({"level": "warning", "code": "drawer_box", "parts": [], "params": {"list": "، ".join(names) if lang == "ar" else ", ".join(names)}})
    return sorted(_grouped(found), key=lambda ch: LEVELS.index(_level(ch)))


def _grouped(found: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The same finding on several parts (twelve shelves let into both sides) is one line: the first, then "Also:"
    the other parts. Shelf, drawer and door gaps with different measured numbers share one warning, and every gap
    stays in that line. Nothing is dropped and no number is changed."""
    out: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}
    for ch in found:
        key = repr((ch.get("code"), _level(ch), sorted((ch.get("params") or {}).items())))
        if key in seen:
            if ch.get("parts"):
                seen[key].setdefault("_also", []).append(ch["parts"])
        else:
            seen[key] = ch = {**ch}
            out.append(ch)
    merged: list[dict[str, Any]] = []
    buckets: dict[tuple, dict[str, Any]] = {}
    for ch in out:
        if ch.get("code") not in _GAP_CODES:
            merged.append(ch)
            continue
        key = (ch.get("code"), _level(ch))
        if key not in buckets:
            buckets[key] = {**ch, "_items": [ch]}
            merged.append(buckets[key])
        else:
            buckets[key]["_items"].append(ch)
    return merged


def _check_pages(meta: dict[str, Any], lang: Lang) -> list[list[tuple[dict[str, Any], list[str]]]]:
    """The findings, wrapped, a page each list (the first page also has the verdict and the size table)."""
    L = LABELS[lang]
    names = {r.get("id"): (r.get("name_ar") or r.get("name")) if lang == "ar" else r.get("name") for r in meta.get("parts") or []}
    width = W - 2 * M - 2 * PAD - 70
    rows = [(ch, wrap(_check_text(ch, L, names), "Sans", 8.5, width, 4)) for ch in _findings(meta, lang)]
    pages: list[list[tuple[dict[str, Any], list[str]]]] = [[]]
    room = H - 2 * M - 40 - 4 * K_ROW - 170  # the first page: under the title, the verdict, the table and the note
    for ch, lines in rows:
        need = 12 + len(lines) * K_LINE
        if need > room and pages[-1]:
            pages.append([])
            room = H - 2 * M - 80
        pages[-1].append((ch, lines))
        room -= need
    return pages


def _checks(c: Canvas, p: Page, L: dict[str, str], meta: dict[str, Any], lang: Lang, stamp: str, first: int, pages: int) -> int:
    """The checks pages; returns how many."""
    content = W - 2 * M
    requested, model = meta.get("_requested") or {}, meta.get("overall") or {}
    layout = _check_pages(meta, lang)
    for index, items in enumerate(layout):
        _page_background(c)
        top = H - M
        p.card(M, top, content, top - (M + 30))
        p.heading(M + PAD, top - 26, L["checks"])
        y = top - 50
        tx, tw = M + PAD, content - 2 * PAD
        if index == 0:
            # The verdict: never "ready" while a critical finding is open (or the checks were never made).
            critical = sum(_level(ch) == "critical" for ch in _findings(meta, lang))
            ready = meta.get("checks") is not None and not critical
            if critical:
                verdict = L["k_not_ready"].format(n=critical)
            elif meta.get("checks") is None:
                verdict = L["k_old"]
            else:
                verdict = L["k_ready"]
            lines = wrap(verdict, "Sans-Bold", 10, tw - 28, 2)
            box_h = 22 + 15 * len(lines)
            p.box(tx, y - box_h, tw, box_h, fill=MINT if ready else ROSE, stroke=GREEN if ready else RED, radius=8)
            for n, line in enumerate(lines):
                p.text(tx + 14, y - 20 - n * 15, line, 10, bold=True, color=GREEN if ready else RED)
            y -= box_h + 14
            cols = [(L["size"], 0.31), (L["k_requested"], 0.23), (L["k_model"], 0.25), (L["k_diff"], 0.21)]
            p.box(tx, y - K_ROW, tw, K_ROW, fill=NAVY, stroke=None, radius=5)
            x = tx
            for title, share in cols:
                p.text(x + 8, y - 14.5, fit(title, "Sans-Bold", 7.5, share * tw - 12), 7.5, bold=True, color=WHITE)
                x += share * tw
            y -= K_ROW
            for n, axis in enumerate(("width", "height", "depth")):
                want_m = float(requested.get(axis) or 0)  # the request's size is in metres
                got = float(model.get(axis) or 0)
                off = _requested_differs(want_m, got)
                want = want_m * 100
                p.box(tx, y - K_ROW, tw, K_ROW, fill=ROSE if off else (WHITE if n % 2 else PAGE), stroke=None, radius=0)
                values = [L[axis], _cm(want_m), (("≈ " if not meta.get("scale_known", True) else "") + _num(got)) if got else "—",
                          (("+" if got - want > 0 else "") + _num(got - want)) if want and got else "—"]
                x = tx
                for (title, share), value in zip(cols, values):
                    p.text(x + 8, y - 14.5, value, 8.5, bold=True, color=RED if off and value == values[3] else NAVY)
                    x += share * tw
                y -= K_ROW
            y -= 10
            for line in wrap(L["k_note"], "Sans", 7.5, tw, 2):
                p.text(tx, y - 8, line, 7.5, color=MUTED)
                y -= 12
            y -= 16
            if meta.get("checks") is not None and not items:
                p.pill(tx, y - 10, L["k_none"], fill=MINT, color=GREEN)
        for ch, lines in items:
            level = _level(ch)
            fill, color = {"critical": (ROSE, RED), "warning": (CREAM, GOLD), "info": (SOFT, MUTED)}[level]
            p.pill(tx, y - 10, L[f"k_{level}"], fill=fill, color=color)
            for n, line in enumerate(lines):
                p.text(tx + 70, y - 10 - n * K_LINE, line, 8.5, color=INK)
            y -= 12 + len(lines) * K_LINE
        _footer(c, p, L, stamp, first + index, pages)
        c.showPage()
    return len(layout)


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
SAME_WITHIN = 0.05  # cm: the same size to the millimetre (frontend measure.ts): a part that differs at all is its own row


# Bedding, not built by the vendor: never in the parts list (frontend lib/designs/partNames.ts SOFT: the same words).
BEDDING = re.compile(r"\b(mattress(es)?|blankets?|duvets?|quilts?|comforters?|bedspreads?|throws?|pillows?|pillowcases?)\b"
                  r"|(?:^|[\s\d_])(?:ال)?(?:مرتبة|مراتب|فرشة|بطانية|بطانيات|لحاف|ألحفة|مخدة|مخدات|وسادة|وسائد|وسادات)(?=$|[\s\d_])", re.I)


def _soft(row: dict[str, Any]) -> bool:
    return any(BEDDING.search(re.sub(r"[_\-.]+", " ", row.get(k) or "")) for k in ("name", "name_ar", "main_name", "main_name_ar"))


# Hardware (handles, hinges, slides). The same idea as frontend partNames.ts FITTING: never a cut.
_FITTING = re.compile(
    r"\b(handles?|knobs?|pulls?|locks?|keys?|hinges?|screws?|bolts?|slides?|runners?|casters?|castors?)\b"
    r"|(?:^|[\s\d_])(?:ال)?(?:مقبض|مقابض|قفل|أقفال|مفصلة|مفصلات|برغي|مجرى|مجاري)(?=$|[\s\d_])",
    re.I,
)
# A drawer's own wooden box, when the model actually has those panels. A front alone is not a box.
_DRAWER_BOX = re.compile(
    r"drawer[\s_\-]*(side|bottom|back|box)|(?:side|bottom|back|box)[\s_\-]*drawer"
    r"|جانب(?:[\s_]+ال)?[\s_]*درج|قاع(?:[\s_]+ال)?[\s_]*درج|ظهر(?:[\s_]+ال)?[\s_]*درج|صندوق(?:[\s_]+ال)?[\s_]*درج",
    re.I,
)


def _part_text(row: dict[str, Any]) -> str:
    return re.sub(r"[_\-.]+", " ", " ".join(str(row.get(k) or "") for k in ("name", "name_ar", "main_name", "main_name_ar")))


def _fitting(row: dict[str, Any]) -> bool:
    return bool(_FITTING.search(_part_text(row)))


def _drawer_box(row: dict[str, Any]) -> bool:
    """A wooden drawer side, bottom, back or box that is in the model. Not a slide, and not a front."""
    return bool(_DRAWER_BOX.search(_part_text(row))) and not _fitting(row)


def _components(meta: dict[str, Any] | None, lang: Lang) -> list[dict[str, Any]]:
    """The table's rows, by group: alike parts once, their count summed and each size's range (`lo` / `hi`).
    The label keeps each part's own name (left door, right door), joined when one cut covers several.
    A drawer box panel is listed even when it was not marked main, because it is a cut. Hardware is not."""
    rows: list[dict[str, Any]] = []
    chosen = (meta or {}).get("_materials") or {}
    for r in (meta or {}).get("parts") or []:
        if _soft(r) or _fitting(r):
            continue
        if not r.get("main", r.get("shown", True)) and not _drawer_box(r):
            continue
        r = _with_material(r, chosen.get(r.get("id")))
        name = (r.get("main_name_ar") or r.get("name_ar") or r.get("name")) if lang == "ar" else (r.get("main_name") or r.get("name"))
        spec = _shop_name(r, lang)
        kind = (name, *(r.get(k) for k in ("kind", "material", "board", "board_mm", "surface", "paint", "texture")), bool(r.get("added")), r.get("group"))
        sizes = {k: float(r.get(k) or 0) for k in SIZES}
        same = next((x for x in rows if x["_kind"] == kind and all(abs(sizes[k] - x["lo"][k]) <= SAME_WITHIN and abs(sizes[k] - x["hi"][k]) <= SAME_WITHIN for k in SIZES)), None)
        if same:
            same["quantity"] += int(r.get("quantity") or 1)
            if spec not in same["_names"]:
                same["_names"].append(spec)
            for k in SIZES:
                same["lo"][k], same["hi"][k] = min(same["lo"][k], sizes[k]), max(same["hi"][k], sizes[k])
        else:
            rows.append({**r, "_kind": kind, "_names": [spec], "label": spec, "quantity": int(r.get("quantity") or 1), "lo": dict(sizes), "hi": dict(sizes)})
    for row in rows:
        row["label"] = " / ".join(n for n in row["_names"] if n) or row.get("id") or ""
    rank = {g: i for i, g in enumerate(GROUP_ORDER)}
    return sorted(rows, key=lambda r: rank.get(r.get("group") or "structure", 0))


GENERIC_NAME = re.compile(r"^(?:part|component|mesh|object|node|group|piece|primitive|geom|جزء|مكون|قطعة|عنصر)(?:[\s_\-]*\d+)?$", re.I)
SHOP_NAME = {
    "en": {"structure": "Carcass panel", "doors": "Door", "drawers": "Drawer", "shelves": "Shelf", "accessories": "Fitting"},
    "ar": {"structure": "لوح الهيكل", "doors": "باب", "drawers": "درج", "shelves": "رف", "accessories": "إكسسوار"},
}


def _shop_name(row: dict[str, Any], lang: Lang) -> str:
    """The name a carpenter uses: the part's own name, with left/right kept. "Part 3" becomes the piece it is."""
    raw = ((row.get("name_ar") if lang == "ar" else row.get("name")) or "").strip()
    fallback = ((row.get("main_name_ar") if lang == "ar" else row.get("main_name")) or "").strip()

    def generic(text: str) -> bool:
        return not text or bool(GENERIC_NAME.match(text))

    if not generic(raw):
        return raw
    if not generic(fallback):
        return fallback
    noun = SHOP_NAME[lang].get(row.get("group") or "structure", SHOP_NAME[lang]["structure"])
    num = re.search(r"\d+", raw or str(row.get("name") or ""))
    return f"{noun} {num.group(0)}" if num else noun


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


FINISH_TYPES = {"melamine", "veneer", "laminate", "lacquer"}  # a surface, never the board that is cut
SOLID = {"oak", "beech", "pine", "walnut", "teak", "ash", "mahogany"}
BOARDS = {"mdf", "hdf", "particleboard", "blockboard"}


def _type_text(wood: dict[str, Any] | None, lang: Lang) -> tuple[str | None, str]:
    """The structural board (None when the choice is only a finish) and the surface finish, as separate strings."""
    if not wood or not (wood.get("name") or wood.get("wood_type") or (wood.get("finish") or {}).get("name")):
        return None, ""
    types = WOOD_TYPES[lang]
    wtype = wood.get("wood_type") or ""
    surface = wood.get("finish") or {}
    finish = ""
    if surface.get("name"):
        kind = surface.get("wood_type") or ""
        finish = surface["name"] + (f" ({types.get(kind, kind)})" if kind else "")
    if wtype in FINISH_TYPES:
        own = wood.get("name") or types.get(wtype, wtype)
        label = types.get(wtype, "")
        if label and label not in own:
            own = f"{own} ({label})"
        finish = f"{finish} · {own}" if finish else own
        return None, finish
    if not wood.get("name") and not wtype:
        return None, finish
    board = types.get(wtype, wtype)
    text = wood.get("name") or board
    if board and board != text:
        text = f"{text} ({board})"
    return text, finish


def _material_parts(row: dict[str, Any], L: dict[str, str], lang: Lang, model_wood: dict[str, Any] | None = None) -> tuple[str, str]:
    """Substrate and finish in two columns. Paint, veneer, laminate, melamine and lacquer are the finish.
    They are never written as the board. A painted part keeps the structural wood when one was chosen."""
    kind = row.get("kind")
    if kind == "paint":
        own = _type_text({"wood_type": row.get("board")} if row.get("board") and row.get("board") not in FINISH_TYPES else None, lang)[0]
        shared = _type_text(model_wood, lang)[0]
        label = row.get("material") or L["tbc"]
        finish = f"{L['kind_paint']}: {label}" + (f" · {row['paint']}" if row.get("paint") else "")
        return own or shared or L["tbc"], finish
    if kind and kind != "wood":
        return f"{L.get(f'kind_{kind}', kind)}: {row.get('material') or L['tbc']}", ""
    if kind == "wood":
        wood = {"name": row.get("material"), "wood_type": row.get("board"),
                "finish": {"name": row.get("surface"), "wood_type": row.get("surface_type")} if row.get("surface") else None}
    else:
        wood = model_wood or {}
    body, finish = _type_text(wood, lang)
    return body or L["tbc"], finish


def _wood(row: dict[str, Any], L: dict[str, str], lang: Lang, model_wood: dict[str, Any] | None = None) -> str:
    """One line for the checks. Paint stays 'Paint: name · colour'. A finish with no board is the finish only,
    not a made-up substrate."""
    if row.get("kind") == "paint":
        text = f"{L['kind_paint']}: {row.get('material') or L['tbc']}"
        return text + (f" · {row['paint']}" if row.get("paint") else "")
    body, finish = _material_parts(row, L, lang, model_wood)
    if not finish:
        return body
    if body == L["tbc"]:
        return f"{L['surface']}: {finish}"
    return f"{body} · {L['surface']}: {finish}"


def _category(row: dict[str, Any], model_wood: dict[str, Any] | None = None) -> str | None:
    """What the part is made of, as a carpenter sorts it: solid wood, engineered board, plywood, glass, mirror…
    A wood panel that carries a mirror stays wood. Only the glass itself is a mirror."""
    role = _mirror_role(row)
    name = f"{row.get('name') or ''} {row.get('name_ar') or ''}"
    if role == "glass":
        return "mirror" if re.search(r"mirror|مرآة|مرايا|مراية", name, re.I) else "glass"
    kind = row.get("kind") or ("wood" if model_wood else None)
    board = row.get("board") if row.get("kind") == "wood" else None
    if not board or board in FINISH_TYPES:
        shared = (model_wood or {}).get("wood_type")
        board = shared if shared and shared not in FINISH_TYPES else board
    if board in FINISH_TYPES:
        board = None
    if kind in ("glass", "metal", "fabric"):
        return kind
    if kind == "paint" and not board:
        return "paint"
    if board in SOLID:
        return "solid"
    if board == "plywood":
        return "plywood"
    if board in BOARDS:
        return "board"
    if kind == "paint":
        return "paint"
    return None


_GLASS_WORD = re.compile(r"(?:^|[\s_])(?:mirrors?|mirrored|glass)(?:$|[\s_])|مرآة|مرايا|مراية|زجاج", re.I)
_SUPPORT_WORD = re.compile(
    r"(?:^|[\s_])(?:frames?|backing|backs?|doors?|panels?|sides?|shel(?:f|ves)|surrounds?|carcass)(?:$|[\s_])|(?:إطار|خلفي|باب|أبواب|لوح|ألواح|جانب|جوانب|رفوف|أرفف|هيكل)|(?:^|[\s_])رف(?:$|[\s_])",
    re.I,
)


def _mirror_role(row: dict[str, Any]) -> str | None:
    """'glass' when the part is the mirror or glass, 'support' when it is the wood behind one, 'unverified' when the
    name says mirror but the thickness is a board's and the material does not say which. None for anything else."""
    name = " ".join(str(row.get(k) or "") for k in ("name", "name_ar", "main_name", "main_name_ar"))
    if row.get("kind") != "glass" and not _GLASS_WORD.search(name):
        return None
    if row.get("kind") == "glass":
        return "glass"
    if row.get("kind") == "wood" or _SUPPORT_WORD.search(name):
        return "support"
    thick = _model_thickness(row)
    if thick and thick <= 1.0:
        return "glass"
    return "unverified"


def _assembly(row: dict[str, Any], L: dict[str, str], category: str | None) -> list[str]:
    """What the size is, when a mirror could be confused with the wood behind it. Hardware is not listed, and no
    missing size is filled in."""
    del category
    role = _mirror_role(row)
    if role == "unverified":
        return [L["a_mirror_ask"]]
    if role == "glass":
        return [L["a_glass"]]
    if role == "support":
        return [L["a_support"]]
    return []


def _span(row: dict[str, Any], key: str) -> str:
    """A size, or its range when alike parts differ a little ("36–37.5")."""
    lo, hi = row.get("lo", {}).get(key, row.get(key) or 0), row.get("hi", {}).get(key, row.get(key) or 0)
    if not hi:
        return "—"
    return _num(lo) if abs(hi - lo) < 0.05 else f"{_num(lo)}–{_num(hi)}"


def _face_axes(row: dict[str, Any]) -> tuple[str, str, str]:
    """Cutting length, cutting width, thickness: the longest side, the middle side, the thinnest side. All three come
    from the part's width, height and depth (the same geometry the drawings use). The chosen board is not an axis."""
    axes = ("width", "height", "depth")

    def size(key: str) -> float:
        return float((row.get("hi") or {}).get(key, row.get(key) or 0))

    return tuple(sorted(axes, key=size, reverse=True))  # type: ignore[return-value]


def _model_thickness(row: dict[str, Any]) -> float:
    """The thickness the cutting list prints: the thinnest side of the 3D part, in cm."""
    return float((row.get("hi") or {}).get(_face_axes(row)[2], row.get(_face_axes(row)[2]) or 0))


def _cut(row: dict[str, Any]) -> tuple[str, str, str]:
    """Cutting length, cutting width and actual thickness, as the table prints them."""
    length, width, thick = _face_axes(row)
    return _span(row, length), _span(row, width), _span(row, thick)


def _board_conflict(row: dict[str, Any]) -> float | None:
    """The chosen board's thickness (cm) when it differs from the 3D part. None when they agree or neither is set."""
    board = float(row.get("board_mm") or 0) / 10
    model = _model_thickness(row)
    if board and model and abs(board - model) > 0.1:
        return board
    return None


# Length, width and thickness: the same three edges the drawings name on a flat panel. The longest face, the other
# face, the thinnest side. Width, height and depth are not repeated beside them.
B_COLS = [("img", 32), ("name", 112), ("qty", 30), ("len", 64), ("wid", 58), ("thk", 58), ("material", 74), ("finish", 63)]  # 491 pt
B_COLS_BARE = [("name", 144), *B_COLS[2:]]  # no part pictures at all: no column of empty cells


def _cols(pictures: Any) -> list[tuple[str, int]]:
    return B_COLS if pictures else B_COLS_BARE
B_TOP = H - M - 68  # under the title and the two-line key
B_BOTTOM = M + 28
B_HEAD = 18
B_GROUP = 22  # a group's title
B_IMG = 22
B_PAD = 5  # a row's padding above and below
SUMMARY = 36  # the requested / model pair on the first cutting-list page


def _b_lines(row: dict[str, Any], L: dict[str, str], lang: Lang, model_wood: dict[str, Any] | None, cols: list[tuple[str, int]] = B_COLS) -> tuple[list[str], list[str], list[str], list[str]]:
    """Name lines, substrate lines, finish lines, and the suggested hardware (kept with the name, not in those columns)."""
    width = dict(cols)
    category = _category(row, model_wood)
    body, finish = _material_parts(row, L, lang, model_wood)
    material = wrap(body, "Sans", 7.5, width["material"] - 20, 3)
    finish_lines = wrap(finish, "Sans", 7.5, width["finish"] - 8, 3) if finish else ["—"]
    assembly = [ln for text in _assembly(row, L, category) for ln in wrap(text, "Sans", 6.5, width["name"] - 8, 3)]
    return wrap(row["label"], "Sans-Bold", 8, width["name"] - 8, 3), material, finish_lines, assembly


def _b_height(row: dict[str, Any], L: dict[str, str], lang: Lang, model_wood: dict[str, Any] | None, cols: list[tuple[str, int]] = B_COLS) -> float:
    name, material, finish, assembly = _b_lines(row, L, lang, model_wood, cols)
    # Matches the drawn baselines: a name line, the status badge, then each suggestion.
    name_block = 13 * len(name) + 16 + 9 * len(assembly)
    text = max(name_block, 11 * len(material), 11 * len(finish), B_IMG)
    extra = 11 if _board_conflict(row) else 0  # the chosen board's thickness, under the model's, when they differ
    return B_PAD + text + B_PAD + extra


def _bom_pages(rows: list[dict[str, Any]], lang: Lang = "ar", model_wood: dict[str, Any] | None = None, status_h: float = 0, cols: list[tuple[str, int]] = B_COLS) -> list[list[tuple[str, Any]]]:
    """Pages of ("group", name) and ("row", part) items: a group's title never ends a page alone; rows never split.
    The closing status is the last item, on the last page when it fits and on a page of its own when it does not."""
    L = LABELS[lang]
    pages: list[list[tuple[str, Any]]] = [[]]
    y = B_TOP - B_HEAD - SUMMARY - 4  # the size summary on the first page
    group = None
    for row in rows:
        h = _b_height(row, L, lang, model_wood, cols)
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
    if status_h:
        if y - status_h < B_BOTTOM:
            pages.append([])
        pages[-1].append(("status", status_h))
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
    heads = {"img": "", "name": L["c_name"], "qty": L["c_qty"], "len": L["c_len"], "wid": L["c_wid"], "thk": L["c_thk"], "material": L["c_material"], "finish": L["c_finish"]}
    status_h = _status_height(views, lang)
    table = _cols(sheet)
    layout = _bom_pages(rows, lang, model_wood, status_h, table)
    for index, items in enumerate(layout):
        c.showPage()
        _page_background(c)
        top = H - M
        p.card(M, top, content, top - (M + 20))
        closing = len(items) == 1 and items[0][0] == "status"
        p.heading(M + PAD, top - 26, L["k_status"] if closing else L["components"])
        if not closing:
            for n, line in enumerate(wrap(L["key"], "Sans", 7, tw, 2)):
                p.text(M + PAD, top - 44 - n * 12, line, 7, color=MUTED)
        y = B_TOP
        if index == 0 and not closing:
            # The whole model: the size asked for (confirmed) and the 3D model's (the parts are measured from it).
            d = views.get("_requested") or {}
            o = views.get("overall") or {}
            half = (tw - 8) / 2
            p.box(tx, y - SUMMARY, half, SUMMARY, fill=CREAM, stroke=HexColor("#F3D9AE"), radius=8)
            p.text(tx + 10, y - 13, fit(L["requested"], "Sans-Bold", 7, half - 20), 7, bold=True, color=GOLD)
            p.text(tx + 10, y - 28, f"{_cm(d.get('width'))} × {_cm(d.get('height'))} × {_cm(d.get('depth'))} {L['cm']}", 9, bold=True, color=NAVY)
            p.box(tx + half + 8, y - SUMMARY, half, SUMMARY, fill=PAGE, stroke=LINE, radius=8)
            p.text(tx + half + 18, y - 13, fit(L["measured"], "Sans-Bold", 7, half - 20), 7, bold=True, color=MUTED)
            p.text(tx + half + 18, y - 28, ("" if known else "≈ ") + (f"{_num(o.get('width'))} × {_num(o.get('height'))} × {_num(o.get('depth'))} {L['cm']}" if o else L["tbc"]), 9, bold=True, color=NAVY)
            y -= SUMMARY + 4
        if not closing:
            # The column titles, on every page that has rows.
            p.box(tx, y - B_HEAD, tw, B_HEAD, fill=NAVY, stroke=None, radius=5)
            cx = tx
            for key, width in table:
                if heads[key]:
                    p.text(cx + 6, y - 12, fit(heads[key], "Sans-Bold", 7, width - 10), 7, bold=True, color=WHITE)
                cx += width
            y -= B_HEAD
        stripe = 0
        for kind, item in items:
            if kind == "status":
                _paint_status(c, p, views, lang, tx, y, tw, float(item))
                y -= float(item)
                continue
            if kind == "group":
                count = sum(int(r.get("quantity") or 1) for r in rows if r.get("group") == item)
                # Room above (apart from the rows before) and below; an amber bar and a fine line under it.
                p.box(tx, y - 16, 3, 11, fill=AMBER, stroke=None, radius=1.5)
                p.text(tx + 8, y - 14, f"{groups.get(item or 'structure', '')} · {count}", 9, bold=True, color=NAVY)
                c.setStrokeColor(HexColor("#F3D9AE"))
                c.setLineWidth(0.6)
                c.line(p.x(tx), y - B_GROUP + 2, p.x(tx + tw), y - B_GROUP + 2)
                y -= B_GROUP
                stripe = 0
                continue
            row = item
            h = _b_height(row, L, lang, model_wood, table)
            added = bool(row.get("added"))
            p.box(tx, y - h, tw, h, fill=CREAM if added else (PAGE if stripe % 2 else WHITE), stroke=None, radius=4)
            stripe += 1
            name_lines, material_lines, finish_lines, assembly_lines = _b_lines(row, L, lang, model_wood, table)
            cx = tx
            for key, width in table:
                if key == "img":
                    image = _picture_of(sheet, row, size, cols)
                    if image is not None:  # a part without a picture leaves the cell empty, no frame
                        p.box(cx + 4, y - B_PAD - B_IMG, B_IMG, B_IMG, fill=WHITE, stroke=LINE, radius=5)
                        c.drawImage(image, p.x(cx + 5, B_IMG - 2), y - B_PAD - 1 - (B_IMG - 2), B_IMG - 2, B_IMG - 2)
                elif key == "name":
                    ny = y - B_PAD - 9
                    for line in name_lines:
                        p.text(cx + 5, ny, line, 8, bold=True, color=NAVY)
                        ny -= 13
                    if not known:
                        tag, fill, color = L["estimated"], CREAM, GOLD
                    elif added:
                        tag, fill, color = L["confirmed"], MINT, GREEN
                    else:
                        tag, fill, color = L["from_model"], SOFT, NAVY
                    p.pill(cx + 4, ny - 2, tag, fill=fill, color=color)
                    ny -= 16
                    for line in assembly_lines:
                        p.text(cx + 4, ny, line, 6.5, color=GOLD)
                        ny -= 9
                elif key == "qty":
                    p.pill(cx + 2, y - B_PAD - 9, f"×{int(row.get('quantity') or 1)}", fill=CREAM, color=GOLD)
                elif key in ("len", "wid", "thk"):
                    length, face, thick = _face_axes(row)
                    value = _span(row, {"len": length, "wid": face, "thk": thick}[key])
                    board = _board_conflict(row) if key == "thk" else None
                    shown = f"{L['model_thk']} {value}" if board else value
                    font = 8
                    while font > 6 and pdfmetrics.stringWidth(shown, "Sans-Bold", font) > width - 8:
                        font -= 0.25
                    # When the chosen board differs, neither number is the cut: both are labelled and the report is not ready.
                    p.text(cx + 4, y - B_PAD - 9, shown, font, bold=True, color=RED if board else (NAVY if value != "—" else MUTED))
                    if board:
                        p.text(cx + 4, y - B_PAD - 21, fit(f"{L['board_spec']} {_num(board)}", "Sans", 6.5, width - 8), 6.5, color=RED)
                elif key == "material":
                    swatch = textures.get(row.get("texture") or "")
                    color = row.get("color")
                    sx = cx + 4
                    if swatch:
                        try:
                            c.drawImage(ImageReader(io.BytesIO(swatch)), p.x(sx, 11), y - B_PAD - 11, 11, 11, mask="auto")
                        except Exception:  # noqa: BLE001 - a broken texture leaves the swatch out
                            pass
                    elif isinstance(color, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", color) and (row.get("kind") or model_wood):
                        p.box(sx, y - B_PAD - 11, 11, 11, fill=HexColor(color), stroke=LINE, radius=5)
                    my = y - B_PAD - 9
                    for line in material_lines:
                        p.text(sx + 14, my, line, 7, color=NAVY if line != L["tbc"] else MUTED)
                        my -= 11
                elif key == "finish":
                    fy = y - B_PAD - 9
                    for line in finish_lines:
                        p.text(cx + 4, fy, line, 7, color=MUTED if line == "—" else NAVY)
                        fy -= 11
                cx += width
            c.setStrokeColor(LINE)
            c.setLineWidth(0.4)
            c.line(p.x(tx + 4), y - h, p.x(tx + tw - 4), y - h)
            y -= h
        _footer(c, p, L, stamp, first + index, pages)


def _num(value: Any) -> str:
    v = float(value or 0)
    return f"{v:.0f}" if abs(v - round(v)) < 0.05 else f"{v:.1f}"


def _status_line(meta: dict[str, Any], lang: Lang) -> tuple[bool, str]:
    """One line for the end of the cutting list. The issues themselves stay on page 2."""
    L = LABELS[lang]
    critical = sum(_level(ch) == "critical" for ch in _findings(meta, lang))
    if critical:
        return False, L["k_status_line"].format(n=critical)
    if meta.get("checks") is None:
        return False, L["k_status_old"]
    return True, L["k_status_ready"]


def _status_height(meta: dict[str, Any], lang: Lang) -> float:
    """The closing line, kept on the last cutting-list page."""
    _, text = _status_line(meta, lang)
    lines = wrap(text, "Sans-Bold", 8.5, W - 2 * M - 2 * PAD - 24, 2)
    return 14 + 12 * len(lines) + 8


def _paint_status(c: Canvas, p: Page, meta: dict[str, Any], lang: Lang, x: float, y: float, width: float, height: float) -> None:
    """Ready or not, with a pointer to the check on page 2. The findings are not copied here."""
    ready, text = _status_line(meta, lang)
    p.box(x, y - height, width, height, fill=MINT if ready else ROSE, stroke=GREEN if ready else RED, radius=6)
    for n, line in enumerate(wrap(text, "Sans-Bold", 8.5, width - 20, 2)):
        p.text(x + 10, y - 14 - n * 12, line, 8.5, bold=True, color=GREEN if ready else RED)



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
    view_meta = (design.get("views") or {}) if views else {}
    measured = view_meta.get("overall") or {}
    known = view_meta.get("scale_known", True)
    mode = L[design["resize_mode"]] if design.get("resize_mode") in ("proportional", "independent") else ""
    extra = []
    # The customer's size stays the large number. The 3D size is added only when the check would call them different, and ≈ only when the scale is unknown.
    if measured and any(_requested_differs(float(dims.get(k) or 0), float(measured.get(k) or 0)) for k in ("width", "height", "depth")):
        mark = "" if known else "≈ "
        extra.append(f"{L['model_size']}: {mark}{_num(measured.get('width'))} × {_num(measured.get('height'))} × {_num(measured.get('depth'))}")
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
        details = " · ".join(x for x in (board if board != wood.get("name") else "",
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
    checks = len(_check_pages(meta, lang)) if meta else 0
    pages = 1 + checks + _view_pages(meta, lang, views)
    _footer(c, p, L, stamp, 1, pages)
    c.showPage()
    if meta:
        _checks(c, p, L, meta, lang, stamp, 2, pages)
    _views_pages(c, p, L, meta, views, stamp, 2 + checks, pages)
    c.showPage()
    c.save()
    return out.getvalue()
