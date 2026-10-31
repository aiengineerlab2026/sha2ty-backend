"""Writes open_plan_apartment.analysis.json: a deterministic vision-model answer for the regression plan
(what a correct model returns, in normalised coordinates), so geometry tests never call the paid API."""

import json
from pathlib import Path

W, H = 1000, 600
n = lambda x, y: [round(x / W, 4), round(y / H, 4)]  # noqa: E731
box = lambda x0, y0, x1, y1: [n(x0, y0), n(x1, y0), n(x1, y1), n(x0, y1)]  # noqa: E731

plan = {
    "image": {"width": W, "height": H},
    "rooms": [
        {"id": "balcony_1", "name": "Balcony", "type": "balcony", "polygon": box(62, 162, 198, 438), "connects_to": ["living_1"], "estimated_width_m": 1.6, "estimated_length_m": 3.3, "dimension_source": "inferred", "confidence": 0.9},
        {"id": "kitchen_1", "name": "Kitchen", "type": "kitchen", "polygon": box(203, 53, 647, 280), "connects_to": ["living_1"], "estimated_width_m": 5.2, "estimated_length_m": 2.7, "dimension_source": "inferred", "confidence": 0.85},
        {"id": "living_1", "name": "Living Room", "type": "living", "polygon": box(203, 280, 647, 547), "connects_to": ["kitchen_1", "hall_1", "balcony_1"], "estimated_width_m": 5.2, "estimated_length_m": 3.2, "dimension_source": "inferred", "confidence": 0.9},
        {"id": "bathroom_1", "name": "Bathroom", "type": "bathroom", "polygon": box(653, 53, 947, 227), "connects_to": ["hall_1"], "estimated_width_m": 3.5, "estimated_length_m": 2.1, "dimension_source": "inferred", "confidence": 0.9},
        {"id": "hall_1", "name": "Hallway", "type": "hall", "polygon": box(653, 233, 947, 327), "connects_to": ["living_1", "bathroom_1", "bedroom_1"], "estimated_width_m": 3.5, "estimated_length_m": 1.1, "dimension_source": "inferred", "confidence": 0.85},
        {"id": "bedroom_1", "name": "Bedroom", "type": "bedroom", "polygon": box(653, 333, 947, 547), "connects_to": ["hall_1"], "estimated_width_m": 3.5, "estimated_length_m": 2.6, "dimension_source": "inferred", "confidence": 0.9},
    ],
    "walls": [
        {"id": "w_top", "start": n(200, 50), "end": n(950, 50), "exterior": True, "rooms": ["kitchen_1", "bathroom_1"]},
        {"id": "w_bottom", "start": n(200, 550), "end": n(950, 550), "exterior": True, "rooms": ["living_1", "bedroom_1"]},
        {"id": "w_left", "start": n(200, 50), "end": n(200, 550), "exterior": True, "rooms": ["kitchen_1", "living_1", "balcony_1"]},
        {"id": "w_right", "start": n(950, 50), "end": n(950, 550), "exterior": True, "rooms": ["bathroom_1", "hall_1", "bedroom_1"]},
        {"id": "w_mid_top", "start": n(650, 50), "end": n(650, 230), "rooms": ["kitchen_1", "bathroom_1"]},
        {"id": "w_mid_bottom", "start": n(650, 330), "end": n(650, 550), "rooms": ["living_1", "bedroom_1"]},
        {"id": "w_bath", "start": n(650, 230), "end": n(950, 230), "rooms": ["bathroom_1", "hall_1"]},
        {"id": "w_bed", "start": n(650, 330), "end": n(950, 330), "rooms": ["hall_1", "bedroom_1"]},
        {"id": "w_bal_top", "start": n(60, 160), "end": n(200, 160), "exterior": True, "rooms": ["balcony_1"]},
        {"id": "w_bal_left", "start": n(60, 160), "end": n(60, 440), "exterior": True, "rooms": ["balcony_1"]},
        {"id": "w_bal_bottom", "start": n(60, 440), "end": n(200, 440), "exterior": True, "rooms": ["balcony_1"]},
    ],
    "doors": [
        {"id": "d_bath", "start": n(700, 230), "end": n(780, 230), "width_m": 0.8, "swing": "left", "connects": ["hall_1", "bathroom_1"], "confidence": 0.85},
        {"id": "d_bed", "start": n(820, 330), "end": n(900, 330), "width_m": 0.8, "swing": "right", "connects": ["hall_1", "bedroom_1"], "confidence": 0.85},
        {"id": "d_entry", "start": n(950, 250), "end": n(950, 310), "width_m": 0.9, "swing": "left", "connects": ["hall_1", "outside"], "confidence": 0.8},
        {"id": "d_balcony", "start": n(200, 220), "end": n(200, 380), "width_m": 1.8, "swing": "sliding", "connects": ["living_1", "balcony_1"], "confidence": 0.8},
    ],
    "windows": [
        {"id": "win_kitchen", "start": n(320, 50), "end": n(480, 50), "room_id": "kitchen_1", "confidence": 0.8},
        {"id": "win_bath", "start": n(780, 50), "end": n(900, 50), "room_id": "bathroom_1", "confidence": 0.8},
        {"id": "win_bed", "start": n(720, 550), "end": n(880, 550), "room_id": "bedroom_1", "confidence": 0.8},
    ],
    "furniture": [
        {"id": "f_counter", "type": "countertop", "room_id": "kitchen_1", "bbox": n(230, 56) + n(620, 116), "width_m": 4.6, "depth_m": 0.6, "confidence": 0.85},
        {"id": "f_table", "type": "dining_table", "room_id": "kitchen_1", "bbox": n(250, 170) + n(380, 250), "confidence": 0.8},
        {"id": "f_sofa", "type": "sofa", "room_id": "living_1", "bbox": n(230, 460) + n(460, 530), "confidence": 0.8},
        {"id": "f_coffee", "type": "table", "room_id": "living_1", "bbox": n(290, 390) + n(400, 440), "confidence": 0.6},
        {"id": "f_tub", "type": "bathtub", "room_id": "bathroom_1", "bbox": n(790, 60) + n(940, 130), "confidence": 0.85},
        {"id": "f_toilet", "type": "toilet", "room_id": "bathroom_1", "bbox": n(672, 58) + n(708, 116), "confidence": 0.8},
        {"id": "f_bed", "type": "bed", "room_id": "bedroom_1", "bbox": n(760, 390) + n(880, 540), "width_m": 1.6, "depth_m": 2.0, "confidence": 0.9},
        {"id": "f_wardrobe", "type": "wardrobe", "room_id": "bedroom_1", "bbox": n(660, 340) + n(700, 520), "confidence": 0.7},
    ],
    "scale_references": [
        {"kind": "door", "element_id": "d_bath", "start": n(700, 230), "end": n(780, 230), "real_m": 0.8},
        {"kind": "door", "element_id": "d_bed", "start": n(820, 330), "end": n(900, 330), "real_m": 0.8},
        {"kind": "bed", "element_id": "f_bed", "start": n(760, 540), "end": n(760, 390), "real_m": 2.0},
        {"kind": "bathtub", "element_id": "f_tub", "start": n(790, 95), "end": n(940, 95), "real_m": 1.6},
        {"kind": "countertop", "element_id": "f_counter", "start": n(400, 56), "end": n(400, 116), "real_m": 0.6},
    ],
    "warnings": [],
    "model": "fixture",
    "schema_version": "fixture",
}
Path(__file__).with_name("open_plan_apartment.analysis.json").write_text(json.dumps(plan, indent=1))
