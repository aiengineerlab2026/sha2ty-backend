"""Draws tests/fixtures/open_plan_apartment.png: the regression layout (balcony left, open living +
kitchen, bathroom upper-right, hallway middle-right, bedroom bottom-right), thin walls, furniture,
no dimensions. Run: .venv/bin/python tests/fixtures/make_open_plan.py"""

from pathlib import Path

import cv2
import numpy as np

W, H, T = 1000, 600, 5  # thin walls: the style OpenCV alone could not seal into rooms
INK, GREY = (40, 40, 40), (150, 150, 150)
img = np.full((H, W, 3), 255, np.uint8)


def wall(a, b):
    cv2.line(img, a, b, INK, T)


# Exterior (gaps: balcony opening x=200 y 220–380, entrance x=950 y 250–310, windows)
wall((200, 50), (950, 50))
wall((200, 550), (950, 550))
wall((200, 50), (200, 220)); wall((200, 380), (200, 550))
wall((950, 50), (950, 250)); wall((950, 310), (950, 550))
# Interior: right column wall x=650 (open to the hallway between y 240 and 320)
wall((650, 50), (650, 230)); wall((650, 330), (650, 550))
wall((650, 230), (700, 230)); wall((780, 230), (950, 230))  # bathroom door 700–780
wall((650, 330), (820, 330)); wall((900, 330), (950, 330))  # bedroom door 820–900
# Door swings (quarter arcs) and leaves
cv2.ellipse(img, (700, 230), (80, 80), 0, 0, 90, GREY, 1); cv2.line(img, (700, 230), (700, 310), INK, 2)
cv2.ellipse(img, (900, 330), (80, 80), 0, 90, 180, GREY, 1); cv2.line(img, (900, 330), (900, 410), INK, 2)
cv2.ellipse(img, (950, 250), (60, 60), 0, 90, 180, GREY, 1); cv2.line(img, (950, 250), (890, 250), INK, 2)
# Windows: double thin lines in the exterior wall
for (x0, y0, x1, y1) in ((320, 50, 480, 50), (720, 550, 880, 550), (780, 50, 900, 50)):
    cv2.rectangle(img, (x0, y0 - 3), (x1, y1 + 3), (255, 255, 255), -1)
    cv2.line(img, (x0, y0 - 3), (x1, y1 - 3), INK, 1); cv2.line(img, (x0, y0 + 3), (x1, y1 + 3), INK, 1)
# Balcony: railing outline outside the apartment, sliding door in the opening
cv2.rectangle(img, (60, 160), (200, 440), INK, 2)
cv2.line(img, (198, 220), (198, 300), INK, 1); cv2.line(img, (202, 300), (202, 380), INK, 1)
# Kitchen along the top: counter run with sink and hob, fridge
cv2.rectangle(img, (230, 56), (620, 116), INK, 1)
cv2.rectangle(img, (300, 66), (360, 106), INK, 1)
for cx in (470, 510):
    for cy in (76, 98):
        cv2.circle(img, (cx, cy), 8, INK, 1)
cv2.rectangle(img, (560, 60), (615, 112), INK, 1); cv2.line(img, (560, 60), (615, 112), GREY, 1)
# Dining table (upper-left) with chairs, sofa and coffee table (lower-left)
cv2.rectangle(img, (250, 170), (380, 250), INK, 1)
for cx in (275, 315, 355):
    cv2.rectangle(img, (cx - 12, 152), (cx + 12, 168), INK, 1); cv2.rectangle(img, (cx - 12, 252), (cx + 12, 268), INK, 1)
cv2.rectangle(img, (230, 460), (460, 530), INK, 1); cv2.rectangle(img, (240, 470), (450, 520), GREY, 1)
cv2.rectangle(img, (290, 390), (400, 440), INK, 1)
# Bathroom: bathtub, toilet, sink
cv2.rectangle(img, (790, 60), (940, 130), INK, 1); cv2.rectangle(img, (800, 70), (930, 120), GREY, 1)
cv2.ellipse(img, (690, 90), (18, 26), 0, 0, 360, INK, 1); cv2.rectangle(img, (672, 58), (708, 70), INK, 1)
cv2.rectangle(img, (660, 160), (720, 215), INK, 1); cv2.circle(img, (690, 188), 14, INK, 1)
# Bedroom: double bed (1.6 × 2.0 m) with pillows, wardrobe
cv2.rectangle(img, (760, 390), (880, 540), INK, 1)
cv2.rectangle(img, (770, 398), (815, 425), GREY, 1); cv2.rectangle(img, (825, 398), (870, 425), GREY, 1)
cv2.rectangle(img, (660, 340), (700, 520), INK, 1)
# Labels only, no dimensions
for text, (x, y) in {"KITCHEN": (400, 150), "LIVING ROOM": (430, 360), "BALCONY": (75, 300), "BATH": (700, 150), "HALL": (770, 290), "BEDROOM": (720, 470)}.items():
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, INK, 1, cv2.LINE_AA)

cv2.imwrite(str(Path(__file__).with_name("open_plan_apartment.png")), img)
