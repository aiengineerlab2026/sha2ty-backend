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
        assert pages(pdf) == 1 + 1 + 2 + len(rr._bom_pages(rr._components(task["spec"]["design"]["views"], lang), lang, None, rr._status_height(task["spec"]["design"]["views"], lang))), lang  # request, checks, a page per picture, components
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


def test_close_ups_are_one_to_a_page():
    rows = [{"n": 1, "id": "p", "name": "Shelf", "added": True, "width": 40.0, "height": 1.8, "depth": 45.0, "thickness": 1.8, "offsets": {"top": 30.0, "bottom": 50.0}}]
    meta = {"closed_url": "x", "open_url": "y", "detail_urls": ["a", "b", "c"], "opened": 1, "parts": rows, "version": 2}
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "views": meta}}}
    pictures = tuple(_png(c) for c in ("#dde", "#edd", "#ded", "#dee", "#eed"))
    pdf = rr.build(task, None, "en", views=pictures)
    assert pages(pdf) == 1 + 1 + 2 + 2 + 1  # the request, the checks, closed and open, 3 close-ups on 2 pages, the cutting list
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
    assert [(r["label"], r["quantity"]) for r in out] == [("باب أيسر / باب أيمن", 2), ("رف مضاف", 1)]  # no handle; one cut, both names
    L = rr.LABELS["ar"]
    assert rr._wood(out[0], L, "ar") == "يحدد مع العميل" and rr._wood(out[1], L, "ar") == "Oak"
    mdf = {**out[1], "board": "mdf", "board_mm": 18, "surface": "Walnut", "surface_type": "veneer"}
    assert rr._wood(mdf, L, "ar") == "Oak (MDF) · السطح: Walnut (قشرة)"
    # A part without its own wood takes the whole model's; a painted part says it is painted.
    assert rr._wood(out[0], L, "ar", {"name": "Natural beech", "wood_type": "beech", "thickness_mm": 18}) == "Natural beech (زان)"
    assert rr._wood({**out[0], "kind": "paint", "material": "Sage", "paint": "أخضر"}, L, "ar") == "دهان: Sage · أخضر"
    meta = {"closed_url": "x", "open_url": "y", "opened": 1, "parts": rows, "version": 3, "overall": {"width": 117.9, "height": 176.9, "depth": 49.7}}
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "dimensions": {"width": 1.2, "height": 1.8, "depth": 0.48}, "views": meta}}}
    for lang in ("ar", "en"):
        pdf = rr.build(task, None, lang, views=(_png(), _png("#edd")))
        assert pages(pdf) == 1 + 1 + 2 + len(rr._bom_pages(rr._components(meta, lang), lang, None, rr._status_height(meta, lang))), lang


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
    assert rr._wood(out["door"], L, "ar") == "Natural beech (MDF) · السطح: Walnut (قشرة)"
    assert out["door"]["texture"] == "https://x/beech.jpg"
    assert rr._wood(out["side"], L, "ar") == "يحدد مع العميل"  # nothing saved on it, and no wood for the whole model


def test_only_identical_parts_share_a_row_each_other_size_is_its_own():
    def shelf(i, width, depth=43.6):
        return {"n": i, "id": f"s{i}", "name": "Shelf", "name_ar": "رف", "main_name_ar": "رف", "main": True, "group": "shelves", "width": width, "height": 1.8, "depth": depth, "thickness": 1.8}
    rows = rr._components({"parts": [shelf(1, 36), shelf(2, 36), shelf(3, 37.3), shelf(4, 37.5, 43.1), shelf(5, 55.9)]}, "ar")
    # A carpenter cuts one size a row: 36 × 2, then 37.3, 37.5 and 55.9 each on their own (never "36–37.5").
    assert [(r["quantity"], r["width"]) for r in rows] == [(2, 36), (1, 37.3), (1, 37.5), (1, 55.9)]
    assert all("–" not in rr._span(r, k) for r in rows for k in ("width", "height", "depth"))


def test_each_row_says_what_it_is_made_of_and_how_it_is_edged_and_joined():
    L = rr.LABELS["en"]
    door = {"name": "Left door", "group": "doors", "kind": "wood", "board": "mdf", "height": 192.0, "depth": 1.8}
    assert rr._category(door) == "board" and rr._assembly(door, L, "board") == []  # edge band and hinges are not written into the cut
    assert rr._category({"name": "Side", "kind": "wood", "board": "oak"}) == "solid"
    assert rr._category({"name": "Side"}, {"wood_type": "plywood"}) == "plywood"  # the whole model's wood
    assert rr._category({"name": "Mirror panel", "kind": "glass"}) == "mirror"
    assert rr._assembly({"name": "Mirror", "group": "doors", "height": 90}, L, "mirror") == [
        "Confirm glass or wood before cutting; the 3D size is kept"]
    assert rr._assembly({"name": "Drawer", "group": "drawers", "depth": 50}, L, "board") == []
    assert rr._assembly({"name": "Shelf", "group": "shelves"}, L, "board") == []
    assert rr._assembly({"name": "Back panel", "group": "structure"}, L, "board") == []  # the joint is a check, not a prescribed connection
    assert rr._assembly({"name": "Side", "group": "structure"}, L, "solid") == []


def test_the_checks_page_compares_requested_and_model_sizes_and_lists_every_finding_on_as_many_pages_as_needed():
    rows = [{"n": 1, "id": "s", "name": "Left side", "name_ar": "جانب أيسر", "width": 1.8, "height": 200.0, "depth": 58.0, "thickness": 1.8},
            {"n": 2, "id": "f", "name": "Shelf", "name_ar": "رف", "width": 118.0, "height": 1.8, "depth": 55.0, "thickness": 1.8}]
    checks = [{"level": "warning", "code": "mismatch", "parts": [], "params": {"axis": "height", "requested": 210, "model": 200, "diff": -10}},
              {"level": "error", "code": "exceeds", "parts": ["s"], "params": {"axis": "height", "value": 212, "overall": 200}},
              {"level": "warning", "code": "overlap", "parts": ["s", "f"], "params": {"depth": 1.8}}]
    names = {"s": "Left side", "f": "Shelf"}
    L = rr.LABELS["en"]
    assert rr._check_text(checks[0], L, names) == "Height: the customer asked for 210 cm, the 3D model is 200 cm (difference -10 cm). Confirm which one to make."
    assert rr._check_text(checks[2], L, names) == "Left side and Shelf go 1.8 cm into each other: their sizes do not fit together as modelled."
    assert rr._check_text(checks[1], rr.LABELS["ar"], {"s": "جانب أيسر"}) == "جانب أيسر: الارتفاع (212 سم) أكبر من الموديل ثلاثي الأبعاد (200 سم). الجزء لا يُصغَّر."
    meta = {"closed_url": "x", "ortho_url": "o", "open_url": "y", "opened": 1, "parts": rows, "version": 8, "checks": checks, "overall": {"width": 120, "height": 200, "depth": 60}}
    found = rr._findings(meta, "en")
    assert [f["code"] for f in found] == ["mismatch", "exceeds", "overlap", "nomaterial"]
    assert [rr._level(f) for f in found] == ["critical", "critical", "warning", "warning"]  # a stored warning on the size still blocks manufacturing
    assert found[-1]["code"] == "nomaterial" and found[-1]["params"]["list"] == "Left side, Shelf"  # no material chosen: to confirm
    assert rr._check_text({"level": "info", "code": "joint", "parts": ["s", "f"], "params": {"depth": 0.5}}, L, names).startswith("Left side and Shelf: one goes 0.5 cm into the other")
    task = {**TASK, "spec": {"design": {**TASK["spec"]["design"], "dimensions": {"width": 1.2, "height": 2.1, "depth": 0.6}, "views": meta}}}
    for lang in ("ar", "en"):
        pdf = rr.build(task, None, lang, views=(_png(), _png("#eee"), _png("#edd")))  # closed and open share a page; front / side / top has the other
        assert pages(pdf) == 1 + 1 + 2 + len(rr._bom_pages(rr._components(meta, lang), lang, None, rr._status_height(meta, lang))), lang
    # A long list goes on to more pages: never cut, never smaller.
    many = {**meta, "checks": [{**checks[2], "params": {"depth": 1 + i / 10}} for i in range(120)]}  # 120 different findings
    assert len(rr._check_pages(many, "en")) > 1 and sum(len(pg) for pg in rr._check_pages(many, "en")) == 121
    pdf = rr.build({**task, "spec": {"design": {**task["spec"]["design"], "views": many}}}, None, "en", views=(_png(), _png("#eee"), _png("#edd")))
    assert pages(pdf) == 1 + len(rr._check_pages(many, "en")) + 2 + len(rr._bom_pages(rr._components(many, "en"), "en", None, rr._status_height(many, "en")))


def test_a_request_takes_a_close_up_for_each_of_up_to_60_rows_and_its_checks():
    from app.api.marketplace import DesignViews

    views = DesignViews(closed_url="a", open_url="b", ortho_url="c", detail_urls=["d"] * 60, detail_groups=["doors"] * 60,
                        checks=[{"level": "error", "code": "overlap", "parts": ["x", "y"], "params": {"depth": 2}},
                                {"level": "warning", "code": "shelf", "parts": ["s", "f"], "params": {"gap": 0.2}},
                                {"level": "warning", "code": "drawer", "parts": ["d", "w"], "params": {"gap": 0.4}}])
    assert len(views.detail_urls) == 60 and [c.code for c in views.checks] == ["overlap", "shelf", "drawer"]


def test_a_requested_size_keeps_its_fraction_and_a_small_difference_is_not_a_second_size():
    assert rr._cm(1.205) == "120.5"  # not rounded to 120 or 121
    assert rr._cm(1.2) == "120" and rr._cm(None) == "—"
    assert rr._requested_differs(2.0, 201.5) is False  # 1.5 cm is within 1% of 200 cm
    assert rr._requested_differs(2.0, 203) is True
    assert rr._requested_differs(0.8, 81.2) is True  # more than 1 cm on a smaller piece


def test_a_board_thicker_than_its_part_in_the_model_is_flagged_never_changed():
    rows = [{"n": 1, "id": "d", "name": "Door", "main": True, "group": "doors", "width": 60.0, "height": 190.0, "depth": 1.8, "thickness": 1.8,
             "kind": "wood", "material": "Oak veneer", "board": "mdf", "board_mm": 25}]
    found = rr._findings({"parts": rows, "checks": []}, "en")
    assert [(f["code"], f["level"], f["params"]) for f in found] == [("board", "critical", {"board": 2.5, "model": 1.8})]
    assert rows[0]["thickness"] == 1.8 and rows[0]["board_mm"] == 25
    cut = rr._components({"parts": rows}, "en")[0]
    assert rr._cut(cut) == ("190", "60", "1.8")  # the model's face and thickness, not the 2.5 cm board
    assert rr._board_conflict(cut) == 2.5  # shown under the model's thickness, not instead of it
    assert rr._status_line({"parts": rows, "checks": []}, "ar") == (False, "يتطلب التأكيد — غير جاهز للتصنيع: 1 مشكلة حرجة، مفصّلة في صفحة 2.")
    agreed = rr._components({"parts": [{**rows[0], "board_mm": 18}]}, "en")[0]
    assert rr._board_conflict(agreed) is None and rr._cut(agreed) == ("190", "60", "1.8")  # 18 mm is the same 1.8 cm
    assert rr._components({"parts": [rows[0], {**rows[0], "id": "d2", "name": "Door"}]}, "en")[0]["quantity"] == 2
    body, finish = rr._material_parts(cut, rr.LABELS["en"], "en")
    assert body == "Oak veneer (MDF)" and finish == ""


def test_a_mirror_is_told_from_the_wood_behind_it_and_neither_size_is_changed():
    L = rr.LABELS["en"]
    glass = {"id": "g", "name": "Mirror", "main": True, "kind": "glass", "material": "Bronze", "width": 40.0, "height": 60.0, "depth": 0.4, "thickness": 0.4}
    frame = {"id": "f", "name": "Mirror frame", "name_ar": "إطار مرآة", "main": True, "group": "structure", "kind": "wood", "material": "Oak", "board": "mdf",
             "width": 42.0, "height": 62.0, "depth": 1.8, "thickness": 1.8}
    unknown = {"id": "u", "name": "Mirror", "main": True, "width": 40.0, "height": 80.0, "depth": 1.8, "thickness": 1.8}
    assert rr._mirror_role(glass) == "glass" and rr._mirror_role(frame) == "support" and rr._mirror_role(unknown) == "unverified"
    assert rr._category(frame) == "board"  # the frame is wood, even though its name says mirror
    assert rr._assembly(frame, L, "board") == ["Wood panel behind the mirror; the glass is cut separately"]
    assert rr._assembly(glass, L, "mirror") == ["This size is the glass or mirror, not the wood behind it"]
    found = rr._findings({"parts": [unknown], "checks": []}, "en")
    assert found[0]["code"] == "mirror" and found[0]["level"] == "critical" and found[0]["params"]["model"] == 1.8
    assert rr._cut(rr._components({"parts": [unknown]}, "en")[0]) == ("80", "40", "1.8")


def test_the_cutting_list_ends_with_one_line_that_points_at_the_check():
    meta = {"checks": [{"level": "warning", "code": "mismatch", "parts": ["s"], "params": {"axis": "height", "expected": 210, "actual": 200, "delta": -10}}], "parts": [{"id": "s", "name": "Side", "main": True, "width": 60, "height": 200, "depth": 1.8}]}
    ready, text = rr._status_line(meta, "en")
    assert ready is False and text == "Requires Confirmation: 1 critical issue(s), listed on page 2."
    assert rr._status_line(meta, "ar")[1].startswith("يتطلب التأكيد")
    assert rr._status_height(meta, "en") < 48
    assert rr._status_line({"checks": [], "parts": []}, "en") == (True, "Ready for Manufacturing. The check is on page 2.")
    assert rr._overview_sheets(3) == [[0, 2], [1]]  # closed and open together; front, side and top on the other page
    assert rr._overview_sheets(2) == [[0], [1]]
    room, width = rr._view_room(), rr._view_width()
    assert rr._take([(2400, 2400), (2400, 2400), (2400, 2400), (2400, 2400)], room, width) == 2  # full frames stay two to a page
    assert rr._take([(1400, 1400), (1400, 1400), (1400, 1400), (1400, 1400)], room, width) == 4
    assert rr._take([(1600, 1600), (1600, 1600), (1600, 1600)], room, width) == 3
    assert rr.LABEL_PX * rr._grid_scale([(1400, 1400)] * 4, room, width) >= 7
    assert rr.LABEL_PX * rr._image_scale([(2400, 2400), (2400, 2400)], room, width) >= 7


def test_a_drawer_box_is_listed_when_the_model_has_it_and_not_invented_when_it_does_not():
    front = {"id": "f", "name": "Drawer front", "name_ar": "واجهة درج", "group": "drawers", "main": True, "width": 50, "height": 18, "depth": 1.8, "thickness": 1.8}
    side = {"id": "s", "name": "Drawer side", "name_ar": "جانب درج", "group": "drawers", "main": False, "width": 1.6, "height": 12, "depth": 45, "thickness": 1.6}
    slide = {"id": "h", "name": "Drawer slide", "group": "drawers", "main": True, "width": 45, "height": 1, "depth": 1.2, "thickness": 1}
    missing = rr._findings({"parts": [front, slide], "checks": []}, "en")
    assert any(f["code"] == "drawer_box" and f["level"] == "warning" and f["params"]["list"] == "Drawer front" for f in missing)
    assert [r["label"] for r in rr._components({"parts": [front, slide]}, "en")] == ["Drawer front"]  # the slide is hardware
    rows = rr._components({"parts": [front, side]}, "en")
    assert [r["label"] for r in rows] == ["Drawer front", "Drawer side"]
    assert rr._cut(rows[1]) == ("45", "12", "1.6")  # the model's side, not a guessed box
    assert not any(f["code"] == "drawer_box" for f in rr._findings({"parts": [front, side], "checks": []}, "en"))
    L = rr.LABELS["en"]
    paint = {"kind": "paint", "material": "Sage", "paint": "Green"}
    body, finish = rr._material_parts(paint, L, "en", {"name": "MDF board", "wood_type": "mdf"})
    assert body == "MDF board (MDF)" and finish == "Paint: Sage · Green"
    assert rr._material_parts(paint, L, "en")[0] == "To confirm"  # paint alone is not a substrate
    assert rr._category(paint, {"wood_type": "mdf"}) == "board"  # the paint sits on the chosen board
    veneer = {"kind": "wood", "material": "Walnut veneer", "board": "veneer"}
    body, finish = rr._material_parts(veneer, L, "en")
    assert body == "To confirm" and finish == "Walnut veneer (Veneer)"
    found = rr._findings({"parts": [{**veneer, "id": "v", "name": "Door", "main": True, "width": 40, "height": 180, "depth": 1.8, "thickness": 1.8}], "checks": []}, "en")
    assert any(f["code"] == "substrate" and f["level"] == "warning" for f in found)


def test_a_generic_part_name_becomes_the_piece_a_carpenter_cuts():
    rows = rr._components({"parts": [
        {"id": "a", "name": "Part 3", "name_ar": "جزء 3", "main_name": "Part", "main_name_ar": "جزء", "group": "doors", "main": True,
         "width": 40, "height": 180, "depth": 1.8, "thickness": 1.8},
        {"id": "b", "name": "Left door", "name_ar": "باب أيسر", "main_name": "Door", "main_name_ar": "باب", "group": "doors", "main": True,
         "width": 40, "height": 180, "depth": 1.8, "thickness": 1.8},
    ]}, "ar")
    assert [r["label"] for r in rows] == ["باب 3", "باب أيسر"]  # same size, different pieces: two cuts, not one "جزء"


def test_the_same_finding_on_several_parts_is_one_line_naming_them_all():
    rows = [{"n": i, "id": i, "name": i, "name_ar": a, "main": True, "group": "shelves", "width": 117.4, "height": 1.8, "depth": 55.0, "thickness": 1.8,
             "kind": "wood", "material": "Oak", "board": "mdf", "board_mm": 18} for i, a in (("L", "جانب أيسر"), ("R", "جانب أيمن"), ("S1", "رف 1"), ("S2", "رف 2"))]
    joint = {"level": "info", "code": "joint", "params": {"depth": 0.5}}
    checks = [{**joint, "parts": ["L", "S1"]}, {**joint, "parts": ["R", "S1"]}, {**joint, "parts": ["L", "S2"]},
              {**joint, "parts": ["R", "S2"], "params": {"depth": 0.8}},  # a different depth: its own line
              {"level": "critical", "code": "mismatch", "parts": [], "params": {"axis": "height", "requested": 210, "model": 200, "diff": -10}}]
    found = rr._findings({"parts": rows, "checks": checks + checks[-1:]}, "en")  # the mismatch twice: said once
    assert [(f["code"], len(f.get("_also") or [])) for f in found] == [("mismatch", 0), ("joint", 2), ("joint", 0)]
    names = {r["id"]: r["name"] for r in rows}
    assert rr._check_text(found[1], rr.LABELS["en"], names).endswith("Also: R and S1; L and S2.")
    ar = {r["id"]: r["name_ar"] for r in rows}
    assert rr._check_text(found[1], rr.LABELS["ar"], ar).endswith("وأيضًا: جانب أيمن ورف 1؛ جانب أيسر ورف 2.")
    shelves = [{"level": "warning", "code": "shelf", "parts": [a, b], "params": {"gap": g}} for a, b, g in (("L", "S1", 0.2), ("R", "S1", 0.4), ("L", "S2", 0.2))]
    grouped = rr._findings({"parts": rows, "checks": shelves}, "en")
    shelf = next(f for f in grouped if f["code"] == "shelf")
    text = rr._check_text(shelf, rr.LABELS["en"], names)
    assert text.startswith("Shelf gaps, each measured:") and "0.2 cm" in text and "0.4 cm" in text and "L and S2" in text


def test_the_cutting_list_uses_the_drawings_words_and_has_no_empty_picture_column():
    keys = [k for k, _ in rr.B_COLS]
    assert keys[keys.index("qty") + 1:keys.index("qty") + 4] == ["len", "wid", "thk"]  # length, width, thickness: one cut, no second height or depth
    assert sum(w for _, w in rr.B_COLS) == sum(w for _, w in rr.B_COLS_BARE) == 491
    assert rr._cols(None) == rr.B_COLS_BARE and "img" not in dict(rr._cols(None)) and rr._cols(b"sheet") == rr.B_COLS
    for lang in ("ar", "en"):
        L = rr.LABELS[lang]
        assert not any(line.endswith("…") for line in rr.wrap(L["key"], "Sans", 7, rr.W - 2 * rr.M - 2 * rr.PAD, 2)), lang  # the key is never cut
        assert (L["c_len"], L["c_wid"], L["c_thk"]) == (("Length", "Width", "Thickness") if lang == "en" else ("الطول", "العرض", "السُمك"))
