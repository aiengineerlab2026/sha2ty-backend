"""The shopper's editor rules on the server: what a customer may add or swap on a vendor's model, drafts without a
name, and the price guide from earlier bids."""

from app.services.price_guide import price_guide
from app.services.shopper_choices import allowed_additions, shopper_choice_errors

COMPONENTS = {
    "components": [
        {"id": "chair_back", "name": "Chair Back", "variants": [
            {"id": "straight", "name": "Straight back", "glb_url": "a.glb"},
            {"id": "curved", "name": "Curved back", "glb_url": "b.glb", "offered": False},
        ]},
        {"id": "legs", "name": "Legs"},
    ],
    "addable": [{"type": "leg", "price_delta": 150}, {"type": "shelf"}],
}


def test_only_parts_and_variants_the_vendor_offers_can_be_saved():
    assert allowed_additions(COMPONENTS) == {"leg": 150.0, "shelf": None}
    ok = {"components": {"added": [{"type": "leg"}]}, "part_variants": {"chair_back": "straight"}}
    assert shopper_choice_errors(ok, COMPONENTS) == []
    bad = {"components": {"added": [{"type": "drawer"}]}, "part_variants": {"chair_back": "curved", "legs": "x"}}
    errors = shopper_choice_errors(bad, COMPONENTS)
    assert len(errors) == 3 and "drawer" in errors[0]
    # A vendor that enabled nothing: no added parts at all.
    assert shopper_choice_errors({"components": {"added": [{"type": "leg"}]}}, {"components": []})


def test_price_guide_needs_enough_bids_and_scales_to_the_size_asked():
    reqs = [
        {"id": "r1", "quantity": 6, "dimensions": {"width": 0.52, "height": 0.86, "depth": 0.56}},
        {"id": "r2", "quantity": 6, "dimensions": {"width": 0.52, "height": 0.86, "depth": 0.56}},
    ]
    bids = [
        {"task_id": "r1", "price": 16500, "price_basis": "total", "quantity": 6},
        {"task_id": "r1", "price": 13200, "status": "rejected"},  # an older bid: a total for the request's pieces
        {"task_id": "r2", "price": 2750, "price_basis": "per_item", "quantity": 6},
        {"task_id": "r2", "price": 9999, "status": "withdrawn"},
    ]
    same = price_guide(reqs, bids, {"width": 0.52, "height": 0.86, "depth": 0.56})
    assert same == {"count": 3, "requests": 2, "low": 2200.0, "high": 2750.0, "median": 2750.0}
    # Twice the volume: about 1.59 × the surface, so the range moves up with it.
    big = price_guide(reqs, bids, {"width": 1.04, "height": 0.86, "depth": 0.56})
    assert big["low"] == round(2200 * 2 ** (2 / 3), -1)
    # One request only: not enough to say anything.
    thin = price_guide(reqs[:1], bids[:2], None)
    assert thin["low"] is None and thin["count"] == 2
