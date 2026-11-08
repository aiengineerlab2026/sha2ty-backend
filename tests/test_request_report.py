import re

from app.api.reports import may_see
from app.services import request_report as rr

TASK = {"id": "3f2a9c1e", "title": "سرير ورد الصباح", "status": "published", "customer_id": "c", "audience": "open", "description": "ملاحظة " * 400,
        "spec": {"design": {"title": "Bed", "parts": [{"id": f"p{i}", "name": f"Part {i}", "requested": True} for i in range(40)],
                            "materials": [{"part_id": "p0", "color": "#aabbcc", "material": {"name": "قماش"}}]}}}


def test_the_report_is_always_one_page_in_both_languages():
    for lang in ("ar", "en"):
        pdf = rr.build(TASK, None, lang, customer="علاء حامد")  # 40 parts and a very long note still fit one page
        assert pdf.startswith(b"%PDF") and len(re.findall(rb"/Type\s*/Page\b(?!s)", pdf)) == 1, lang


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


def test_part_names_follow_the_report_language():
    from app.services.request_report import part_name

    assert part_name("Chair Back", "ar") == "ظهر الكرسي"
    assert part_name("ظهر الكرسي", "en") == "Chair Back"
    assert part_name("ظهر الكرسي", "ar") == "ظهر الكرسي"
    assert part_name("Drawer", "ar") == "Drawer"  # a vendor's own name stays as written
