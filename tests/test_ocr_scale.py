from app.services.plan.dimension_ocr import OcrBox, calibrate, parse_dimension, parse_dimension_pair, parse_length_to_meters, room_labels


def test_feet_and_inches_to_meters():
    assert abs(parse_length_to_meters("13'") - 3.9624) < 1e-4
    assert abs(parse_length_to_meters("12'") - 3.6576) < 1e-4
    assert abs(parse_length_to_meters("16'") - 4.8768) < 1e-4
    assert abs(parse_length_to_meters("14'") - 4.2672) < 1e-4
    assert abs(parse_length_to_meters("10'") - 3.048) < 1e-4
    assert abs(parse_length_to_meters("12'6\"") - 3.81) < 1e-4
    assert abs(parse_length_to_meters("12' 6\"") - 3.81) < 1e-4
    assert abs(parse_length_to_meters("12'-6\"") - 3.81) < 1e-4
    assert abs(parse_length_to_meters("12 ft 6 in") - 3.81) < 1e-4
    assert abs(parse_length_to_meters("4m") - 4.0) < 1e-9
    assert abs(parse_length_to_meters("400cm") - 4.0) < 1e-9


def test_dimension_pair_variants():
    for text in ("13'x12'", "13' x 12'", "13’ × 12’", "13'X12'", "13′ x 12′"):
        d = parse_dimension(text)
        assert d and abs(d.a - 3.9624) < 1e-4 and abs(d.b - 3.6576) < 1e-4 and not d.uncertain, text
    d = parse_dimension("12'6\" x 10'")
    assert abs(d.a - 3.81) < 1e-4 and abs(d.b - 3.048) < 1e-4
    assert parse_dimension("16' x 13'").a == parse_length_to_meters("16'")
    assert parse_dimension("Bedroom 14'x 10'").b == parse_length_to_meters("10'")
    assert parse_dimension("4m x 3m").a == 4.0
    assert parse_dimension("3.5 x 4.2 m").b == 4.2


def test_ocr_mangled_marks_are_recovered_but_flagged():
    # macOS Vision reads 12'6" x 10' on the regression drawing as: 12 6 x 10"
    d = parse_dimension('12 6 x 10"')
    assert d and abs(d.a - 3.81) < 1e-4 and abs(d.b - 3.048) < 1e-4 and d.uncertain


def test_bare_numbers_need_drawing_evidence_for_units():
    assert parse_dimension("13 × 12") is None
    d = parse_dimension("13 × 12", drawing_unit="ft")
    assert d.uncertain and abs(d.a - 3.9624) < 1e-4
    assert parse_dimension_pair("13 x 12") is not None  # legacy helper assumes feet


def test_labels_pair_names_with_dimensions_and_skip_titles():
    boxes = [
        OcrBox("3 Bedroom Apartment Floor Plan", 150, 60, 600, 36),
        OcrBox("Bedroom", 140, 230, 110, 20),
        OcrBox("13'x 12'", 140, 260, 95, 20),
        OcrBox("Kitchen", 400, 630, 88, 22),
        OcrBox("Dining", 210, 615, 75, 22),
        OcrBox('12 6 x 10"', 210, 645, 135, 20),
        OcrBox("13 x 12", 600, 290, 95, 20),
        OcrBox("Bedroom", 600, 262, 110, 20),
    ]
    labels, unit = room_labels(boxes)
    assert unit == "ft"
    names = [l.name for l in labels]
    assert names.count("Bedroom") == 2 and "Kitchen" in names and "Dining" in names
    dining = next(l for l in labels if l.name == "Dining")
    assert abs(dining.dims_m[0] - 3.81) < 1e-3
    kitchen = next(l for l in labels if l.name == "Kitchen")
    assert kitchen.dims_m is None  # the dining dimension is not stolen by the kitchen label
    right = next(l for l in labels if l.name == "Bedroom" and l.x > 500)
    assert right.dims_uncertain  # unit inferred from the other labels


def test_calibration_rejects_outlier_and_reports_non_uniform_scale():
    ft = 0.3048
    zone = lambda i, dims, px: {"id": f"z{i}", "name": f"R{i}", "dims_m": dims, "measure_px": {"width": px[0], "depth": px[1]}, "boundary": "wall"}
    zones = [
        zone(1, (13 * ft, 12 * ft), (192, 208)),
        zone(2, (14 * ft, 10 * ft), (224, 160)),
        zone(3, (16 * ft, 13 * ft), (256, 208)),
        zone(4, (12 * ft, 10 * ft), (400, 330)),  # wrong label association
    ]
    result = calibrate(zones)
    assert abs(0.3048 / result["meters_per_pixel"] - 16) < 0.3
    assert "z4" not in result["inliers"]
    assert result["uniform"]
    stretched = [zone(i, z["dims_m"], (z["measure_px"]["width"], z["measure_px"]["depth"] * 1.25)) for i, z in enumerate(zones[:3])]
    result = calibrate(stretched)
    assert not result["uniform"] and result["confidence"] <= 0.5


def test_inferred_zones_do_not_calibrate():
    z = {"id": "z", "name": "Dining", "dims_m": (3.81, 3.048), "measure_px": {"width": 300, "depth": 160}, "boundary": "inferred"}
    assert calibrate([z])["meters_per_pixel"] is None
