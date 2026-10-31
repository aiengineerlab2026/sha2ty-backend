from app.services.plan.scale_estimator import door_references, estimate_scale, robust_scale


def test_outlier_is_rejected_and_doors_outweigh_furniture():
    refs = [{"source": "door", "mpp": v, "weight": 1.0} for v in (0.020, 0.021, 0.019)]
    refs.append({"source": "furniture", "mpp": 0.05, "weight": 0.5})
    mpp, inliers = robust_scale(refs)
    assert 0.019 <= mpp <= 0.021 and len(inliers) == 3


def test_estimate_prefers_printed_then_refs_then_heuristic():
    printed = {"meters_per_pixel": 0.02, "confidence": 0.9, "warnings": []}
    assert estimate_scale(printed, [], 10)["source"] == "printed dimensions"
    doors = door_references([{"kind": "door", "start": [0, 0], "end": [40, 0], "confidence": 0.9}] * 2)
    by_doors = estimate_scale({"meters_per_pixel": None, "confidence": 0.0, "warnings": []}, doors, 10)
    assert by_doors["source"] == "doors" and abs(by_doors["meters_per_pixel"] - 0.02) < 1e-9
    guess = estimate_scale({"meters_per_pixel": None, "confidence": 0.0, "warnings": []}, [], 10)
    assert guess["source"] == "heuristic" and guess["confidence"] <= 0.1 and guess["meters_per_pixel"] == 0.02
