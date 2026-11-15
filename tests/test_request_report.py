import re

from app.api.reports import may_see
from app.services import request_report as rr

TASK = {"id": "3f2a9c1e", "title": "سرير ورد الصباح", "status": "published", "customer_id": "c", "audience": "open", "description": "ملاحظة " * 400,
        "spec": {"design": {"title": "Bed", "parts": [{"id": f"p{i}", "name": f"Part {i}", "requested": True} for i in range(40)],
                            "materials": [{"part_id": "p0", "color": "#aabbcc", "material": {"name": "قماش"}}]}}}


def pages(pdf: bytes) -> int:
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf))


def test_the_request_page_is_always_one_page_and_a_request_without_views_says_so():
    for lang in ("ar", "en"):
        pdf = rr.build(TASK, None, lang, customer="علاء حامد")  # 40 parts and a very long note still fit the first page
        assert pdf.startswith(b"%PDF") and pages(pdf) == 2, lang  # then the views page with its notice


def _png(color: str = "#dde") -> bytes:
    import io

    from PIL import Image

    out = io.BytesIO()
    Image.new("RGB", (300, 200), color).save(out, "PNG")
    return out.getvalue()


def test_the_views_section_has_both_pictures_and_no_parts_table():
    rows = [{"n": i + 1, "id": f"p{i}", "name": f"Part {i}", "added": i % 7 == 0, "material": "MDF", "color": "#aabbcc",
             "width": 40.0, "height": 1.8, "depth": 45.0, "thickness": 1.8, "offsets": {"top": 30.0, "bottom": 50.5, "left": 0.0, "right": 0.0} if i % 7 == 0 else None}
            for i in range(90)]
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "views": {"closed_url": "x", "open_url": "y", "opened": 3, "scale_known": False, "parts": rows, "version": 2}}}}
    for lang in ("ar", "en"):
        pdf = rr.build(task, None, lang, views=(_png(), _png("#edd")))
        assert pages(pdf) == 1 + 2 + len(rr._bom_pages(rr._components(task["spec"]["design"]["views"], lang))), lang  # request, a page per picture, components
        assert pdf.count(b"/Subtype /Image") >= 2


def test_who_may_download_a_report():
    approved = {"status": "approved"}
    assert may_see(TASK, "c", None, [])  # the customer
    assert may_see(TASK, "v", approved, [])  # an open request, any approved vendor
    assert not may_see(TASK, "v", {"status": "pending"}, [])
    exclusive = {**TASK, "audience": "vendor_exclusive", "exclusive_vendor_id": "v1"}
    assert may_see(exclusive, "v1", approved, []) and not may_see(exclusive, "v2", approved, [])
    awarded = {**TASK, "status": "awarded"}
    assert may_see(awarded, "v2", approved, [{"vendor_id": "v2"}]) and not may_see(awarded, "v3", approved, [])
    assert not may_see(TASK, "stranger", None, [])


def test_close_ups_get_their_own_pages_two_to_a_page():
    rows = [{"n": 1, "id": "p", "name": "Shelf", "added": True, "width": 40.0, "height": 1.8, "depth": 45.0, "thickness": 1.8, "offsets": {"top": 30.0, "bottom": 50.0}}]
    meta = {"closed_url": "x", "open_url": "y", "detail_urls": ["a", "b", "c"], "opened": 1, "parts": rows, "version": 2}
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "views": meta}}}
    pictures = tuple(_png(c) for c in ("#dde", "#edd", "#ded", "#dee", "#eed"))
    pdf = rr.build(task, None, "en", views=pictures)
    assert pages(pdf) == 1 + 5 + 1  # the request, a page per picture (closed, open, 3 details), the components
    assert pdf.count(b"/Subtype /Image") >= 5


def test_numbered_marker_pictures_are_never_shown():
    meta = {"closed_url": "x", "open_url": "y", "opened": 1, "parts": [], "version": 1}
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "views": meta}}}
    pdf = rr.build(task, None, "en", views=(_png(), _png("#edd")))
    assert pages(pdf) == 2 and b"/Subtype /Image" not in pdf  # the notice page, no pictures


def test_components_list_main_parts_once_with_their_count_and_mark_what_is_not_known():
    def part(i, name, name_ar, main=True, **kw):
        return {"n": i, "id": f"p{i}", "name": name, "name_ar": name_ar, "main_name": name.split(" ", 1)[-1], "main_name_ar": name_ar.split(" ")[0],
                "main": main, "width": 40, "height": 196.3, "depth": 1.8, "thickness": 1.8, "quantity": 1, **kw}
    rows = [
        part(1, "Left door", "باب أيسر", color="#ffffff"),
        part(2, "Right door", "باب أيمن", color="#ffffff"),
        part(3, "Door handle", "مقبض باب", main=False),
        part(4, "Added shelf", "رف مضاف", added=True, height=1.8, depth=45, kind="wood", material="Oak", finish="matte"),
    ]
    out = rr._components({"parts": rows}, "ar")
    assert [(r["label"], r["quantity"]) for r in out] == [("باب", 2), ("رف", 1)]  # no handle; the doors once, ×2
    L = rr.LABELS["ar"]
    assert rr._wood(out[0], L, "ar") == "يحدد مع العميل" and rr._wood(out[1], L, "ar") == "Oak"
    mdf = {**out[1], "board": "mdf", "board_mm": 18, "surface": "Walnut", "surface_type": "veneer"}
    assert rr._wood(mdf, L, "ar") == "Oak (MDF) · 18 مم · السطح: Walnut (قشرة)"
    # A part without its own wood takes the whole model's; a painted part says it is painted.
    assert rr._wood(out[0], L, "ar", {"name": "Natural beech", "wood_type": "beech", "thickness_mm": 18}) == "Natural beech (زان) · 18 مم"
    assert rr._wood({**out[0], "kind": "paint", "material": "Sage", "paint": "أخضر"}, L, "ar") == "دهان: Sage · أخضر"
    meta = {"closed_url": "x", "open_url": "y", "opened": 1, "parts": rows, "version": 3, "overall": {"width": 117.9, "height": 176.9, "depth": 49.7}}
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "dimensions": {"width": 1.2, "height": 1.8, "depth": 0.48}, "views": meta}}}
    for lang in ("ar", "en"):
        pdf = rr.build(task, None, lang, views=(_png(), _png("#edd")))
        assert pages(pdf) == 1 + 2 + len(rr._bom_pages(out)), lang  # the request, a page per picture, the components


def test_a_card_has_its_parts_picture_and_none_without_one():
    import io

    from PIL import Image, ImageDraw

    rows = [{"n": i + 1, "id": f"p{i}", "name": f"Part {i}", "name_ar": f"جزء {i}", "main": True, "group": "structure", "pic": i if i < 2 else None,
             "width": 40, "height": 180, "depth": 45, "thickness": 1.8} for i in range(3)]
    sheet = Image.new("RGB", (8 * 64, 64), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    for i in range(8):
        draw.rectangle((i * 64, 0, i * 64 + 63, 63), fill=(i * 30, 90, 200 - i * 20))
    out = io.BytesIO()
    sheet.save(out, "JPEG")
    meta = {"closed_url": "x", "open_url": "y", "opened": 1, "parts": rows, "version": 4, "pic_size": 64, "pic_cols": 8}
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "views": meta}}}
    pdf = rr.build(task, None, "ar", views=(_png(), _png("#edd")), pictures=out.getvalue())
    assert pdf.count(b"/Subtype /Image") == 2 + 2  # the two views and the two parts that have a picture


def test_each_part_shows_the_wood_saved_on_it_in_the_editor_not_what_the_pictures_recorded():
    rows = [{"n": 1, "id": "door", "name": "Door", "name_ar": "باب", "main": True, "group": "doors", "width": 40, "height": 180, "depth": 1.8, "thickness": 1.8},
            {"n": 2, "id": "side", "name": "Side", "name_ar": "جانب", "main": True, "group": "structure", "width": 1.8, "height": 180, "depth": 45, "thickness": 1.8}]
    wood = {"kind": "wood", "name": "Natural beech", "wood_type": "mdf", "thickness_mm": 18, "finish": {"name": "Walnut", "wood_type": "veneer"}}
    meta = {"parts": rows, "_materials": {"door": {"part_id": "door", "material": wood, "mapUrl": "https://x/beech.jpg"}}}
    out = {r["id"]: r for r in rr._components(meta, "ar")}
    L = rr.LABELS["ar"]
    assert rr._wood(out["door"], L, "ar") == "Natural beech (MDF) · 18 مم · السطح: Walnut (قشرة)"
    assert out["door"]["texture"] == "https://x/beech.jpg"
    assert rr._wood(out["side"], L, "ar") == "يحدد مع العميل"  # nothing saved on it, and no wood for the whole model


def test_alike_parts_are_one_row_with_their_count_and_the_range_of_their_sizes():
    def shelf(i, width, depth=43.6):
        return {"n": i, "id": f"s{i}", "name": "Shelf", "name_ar": "رف", "main_name_ar": "رف", "main": True, "group": "shelves", "width": width, "height": 1.8, "depth": depth, "thickness": 1.8}
    rows = rr._components({"parts": [shelf(1, 36), shelf(2, 36), shelf(3, 37.3), shelf(4, 37.5, 43.1), shelf(5, 55.9)]}, "ar")
    assert [(r["label"], r["quantity"]) for r in rows] == [("رف", 4), ("رف", 1)]  # within a centimetre: one row; 55.9 apart
    assert rr._span(rows[0], "width") == "36–37.5" and rr._span(rows[0], "height") == "1.8" and rr._span(rows[1], "width") == "55.9"
