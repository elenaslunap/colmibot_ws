"""Heuristic cardboard-box detector (demo).

Approach:
  1. Keep pixels whose colour looks like cardboard (brown/tan in HSV).
  2. Clean the mask and find connected blobs.
  3. Split touching boxes: find long straight edges inside the cardboard
     area (Canny + HoughLinesP). A line is a "seam" only if the brightness
     on its two sides differs, both near (12 px) and farther away (20 px);
     text or tape has the same box on both sides and is not a seam. Each
     seam is grown along its direction while the two sides still differ,
     then the mask is cut along it.
  4. Keep blobs that are large and box-shaped: they fill most of their
     rotated bounding rectangle and their outline simplifies to 4-6 corners
     (4 = one face seen head-on, 6 = a box seen at an angle).

Seams only exist where neighbouring boxes look different (different shade
or lighting, or a visible gap). Identical boxes under flat lighting still
merge into one blob. A box seen at an angle may be split into its two
visible faces when they are lit differently.

Limitations: anything cardboard-coloured and roughly rectangular counts
(wooden tables, some skin tones, beige walls). Coordinates are pixels, not
meters. Use the "Box Controls" sliders to tune the colour range live.
"""
import cv2
import numpy as np

# -------------------- Tunable parameters --------------------
# OpenCV HSV ranges: H 0-179, S 0-255, V 0-255.
HSV_LOW = (5, 40, 60)
HSV_HIGH = (25, 220, 245)

MIN_BOX_AREA = 1500          # pixels; ignore small blobs (a 0.3 m box at ~3.5 m)
MIN_RECT_FILL = 0.75         # blob area / rotated-rect area
MIN_SOLIDITY = 0.85          # blob area / convex-hull area
MIN_CORNERS, MAX_CORNERS = 4, 6
MORPH_KERNEL_SIZE = 7

# Splitting touching boxes along seams (step 3).
SPLIT_TOUCHING = True
CANNY_LOW, CANNY_HIGH = 10, 30   # low: seams between boxes can be faint
MIN_SEAM_LENGTH = 30             # pixels; shorter edges are ignored (texture)
MAX_SEAM_GAP = 10                # pixels; joins pieces of one seam broken by noise
SEAM_THICKNESS = 3               # pixels cut from the mask along each seam
SEAM_GROW_STEP = 2               # pixels; Hough finds only pieces of a seam, so
                                 # each seam is grown along its direction while
                                 # the two sides still differ in brightness
SEAM_GROW_MAX_MISSES = 3         # consecutive steps without contrast before stopping
SEAM_EXTEND = 6                  # pixels cut past the grown end, to close corners
SEAM_SIDE_OFFSET = 12            # pixels; where brightness is sampled on each side.
                                 # Far enough to skip a gap between boxes and to
                                 # land past a line of text or a strip of tape.
SEAM_CONFIRM_OFFSET = 20         # pixels; a real seam still separates two shades
                                 # this far away, a line of text does not
SEAM_MIN_CONTRAST = 5            # grey levels; a seam separates two different
                                 # shades. Text or thin tape has the same box on
                                 # both sides and is not cut.

CONTROLS_WINDOW = "Box Controls"


def cardboard_mask(frame, hsv_low=HSV_LOW, hsv_high=HSV_HIGH):
    """Binary mask of pixels in the cardboard colour range."""
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(hsv_low), np.array(hsv_high))

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE)
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # fill small holes
    return mask


def _sides(gray, mask, px, py, nx, ny, offset=SEAM_SIDE_OFFSET):
    """Sample both sides of a line at points (px, py) along normal (nx, ny).

    Returns (in_image, side_a_cardboard, side_b_cardboard, abs_brightness_diff),
    one value per point.
    """
    h, w = gray.shape
    ax = np.round(px + offset * nx).astype(int)
    ay = np.round(py + offset * ny).astype(int)
    bx = np.round(px - offset * nx).astype(int)
    by = np.round(py - offset * ny).astype(int)
    inside = ((ax >= 0) & (ax < w) & (ay >= 0) & (ay < h)
              & (bx >= 0) & (bx < w) & (by >= 0) & (by < h))
    ax, ay = np.clip(ax, 0, w - 1), np.clip(ay, 0, h - 1)
    bx, by = np.clip(bx, 0, w - 1), np.clip(by, 0, h - 1)
    diff = np.abs(gray[ay, ax].astype(int) - gray[by, bx].astype(int))
    return inside, mask[ay, ax] > 0, mask[by, bx] > 0, diff


def _is_seam(gray, mask, x1, y1, x2, y2):
    """True if the line separates two cardboard areas of different brightness."""
    length = np.hypot(x2 - x1, y2 - y1)
    if length == 0:
        return False
    nx, ny = -(y2 - y1) / length, (x2 - x1) / length   # unit normal
    t = np.linspace(0.0, 1.0, max(int(length // 2), 2))
    px, py = x1 + t * (x2 - x1), y1 + t * (y2 - y1)

    for offset in (SEAM_SIDE_OFFSET, SEAM_CONFIRM_OFFSET):
        inside, card_a, card_b, diff = _sides(gray, mask, px, py, nx, ny, offset)
        if not inside.any():
            return False
        card_a, card_b, diff = card_a[inside], card_b[inside], diff[inside]

        # One side is not cardboard: this is the box outline, not a seam.
        if offset == SEAM_SIDE_OFFSET and (np.mean(card_a) < 0.6 or np.mean(card_b) < 0.6):
            return False

        # Compare point by point: a seam can run past several boxes, and which
        # side is brighter may change along it (so per-side averages would cancel).
        if np.median(diff) < SEAM_MIN_CONTRAST:
            return False
    return True


def _grow_seam(gray, mask, x1, y1, x2, y2):
    """Extend a seam piece along its direction while it still separates two shades.

    Hough returns only pieces of each seam; without this, the cut stops short
    of corners and T-junctions and leaves bridges between boxes.
    """
    length = np.hypot(x2 - x1, y2 - y1)
    ux, uy = (x2 - x1) / length, (y2 - y1) / length       # unit direction
    nx, ny = -uy, ux                                        # unit normal
    max_steps = int(np.hypot(*gray.shape) / SEAM_GROW_STEP)
    k = np.arange(1, max_steps + 1) * SEAM_GROW_STEP

    ends = []
    for (ex, ey), sign in (((x1, y1), -1), ((x2, y2), 1)):
        px, py = ex + sign * k * ux, ey + sign * k * uy
        inside, card_a, card_b, diff = _sides(gray, mask, px, py, nx, ny)
        good = inside & card_a & card_b & (diff >= SEAM_MIN_CONTRAST)

        # Stop at the first run of SEAM_GROW_MAX_MISSES steps without contrast.
        misses = np.convolve(~good, np.ones(SEAM_GROW_MAX_MISSES, int), "valid")
        stop = np.flatnonzero(misses == SEAM_GROW_MAX_MISSES)
        limit = stop[0] if len(stop) else len(good)
        grown = np.flatnonzero(good[:limit])
        reach = k[grown[-1]] if len(grown) else 0

        reach += SEAM_EXTEND
        ends.append((int(round(ex + sign * reach * ux)), int(round(ey + sign * reach * uy))))
    return ends


def seam_lines(frame, mask):
    """Mask of long straight edges inside (or on the border of) the cardboard area."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)

    # Ignore edges far from cardboard (e.g. on the floor or walls).
    near_cardboard = cv2.dilate(mask, np.ones((5, 5), np.uint8))
    edges = cv2.bitwise_and(edges, near_cardboard)

    seams = np.zeros_like(mask)
    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=MIN_SEAM_LENGTH // 2,
        minLineLength=MIN_SEAM_LENGTH, maxLineGap=MAX_SEAM_GAP,
    )
    if lines is not None:
        # reshape: OpenCV 4 returns (N, 1, 4), OpenCV 5 returns (N, 4)
        for x1, y1, x2, y2 in lines.reshape(-1, 4).astype(float):
            if not _is_seam(gray, mask, x1, y1, x2, y2):
                continue
            p1, p2 = _grow_seam(gray, mask, x1, y1, x2, y2)
            cv2.line(seams, p1, p2, 255, SEAM_THICKNESS)
    return seams


def split_touching(frame, mask):
    """Cut the mask along seams so touching boxes become separate blobs."""
    cut = cv2.bitwise_and(mask, cv2.bitwise_not(seam_lines(frame, mask)))
    # Remove thin bridges and slivers left between the pieces.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(cut, cv2.MORPH_OPEN, kernel)


def find_boxes(mask):
    """Return detection dicts for box-shaped blobs in the mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < MIN_BOX_AREA:
            continue

        rotated = cv2.minAreaRect(contour)
        (rw, rh) = rotated[1]
        if rw * rh == 0:
            continue
        fill = area / (rw * rh)
        if fill < MIN_RECT_FILL:
            continue

        hull_area = cv2.contourArea(cv2.convexHull(contour))
        solidity = area / hull_area if hull_area else 0
        if solidity < MIN_SOLIDITY:
            continue

        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if not MIN_CORNERS <= len(polygon) <= MAX_CORNERS:
            continue

        x, y, w, h = (int(v) for v in cv2.boundingRect(contour))
        moments = cv2.moments(contour)
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        detections.append({
            "bbox": (x, y, x + w, y + h),
            "center": (cx, cy),
            "area": int(area),
            "angle": round(float(rotated[2]), 1),
            "corners": len(polygon),
            # Rough 0-1 shape score, not a calibrated probability.
            "score": round(float(min(fill, solidity)), 2),
            "outline": polygon,
            "rotated_rect": rotated,
        })

    detections.sort(key=lambda d: d["area"], reverse=True)
    return detections


def detect_boxes(frame, hsv_low=HSV_LOW, hsv_high=HSV_HIGH):
    """Return (annotated_frame, mask, detections).

    The returned mask already has the seam cuts (black lines), so the
    "Cardboard Mask" window shows how touching boxes were separated.
    """
    mask = cardboard_mask(frame, hsv_low, hsv_high)
    if SPLIT_TOUCHING:
        mask = split_touching(frame, mask)
    detections = find_boxes(mask)
    return draw_detections(frame, detections), mask, detections


def draw_detections(frame, detections):
    output = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx, cy = det["center"]
        box_pts = cv2.boxPoints(det["rotated_rect"]).astype(np.int32)

        cv2.drawContours(output, [det["outline"]], -1, (255, 0, 0), 2)  # blue: outline
        cv2.drawContours(output, [box_pts], -1, (0, 255, 255), 2)        # yellow: rotated rect
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 3)        # green: bbox
        cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)                 # red: center
        cv2.putText(
            output,
            f"Box  score {det['score']:.2f}  center ({cx}, {cy})",
            (x1, max(25, y1 - 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
        )

    cv2.putText(
        output, f"Possible boxes: {len(detections)}", (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
    )
    return output


# -------------------- Live demo with tuning sliders --------------------
def create_controls():
    cv2.namedWindow(CONTROLS_WINDOW)
    for name, value, maximum in [
        ("H low", HSV_LOW[0], 179), ("H high", HSV_HIGH[0], 179),
        ("S low", HSV_LOW[1], 255), ("S high", HSV_HIGH[1], 255),
        ("V low", HSV_LOW[2], 255), ("V high", HSV_HIGH[2], 255),
    ]:
        cv2.createTrackbar(name, CONTROLS_WINDOW, value, maximum, lambda _: None)


def read_controls():
    get = lambda name: cv2.getTrackbarPos(name, CONTROLS_WINDOW)
    low = (get("H low"), get("S low"), get("V low"))
    high = (get("H high"), get("S high"), get("V high"))
    return low, high


_last_count = None


def process_frame(frame):
    global _last_count
    low, high = read_controls()
    annotated, mask, detections = detect_boxes(frame, low, high)

    # Print to the terminal only when the number of boxes changes.
    if len(detections) != _last_count:
        _last_count = len(detections)
        summary = [(d["center"], d["score"]) for d in detections]
        print(f"{len(detections)} box(es): {summary}")

    return {"Box Detection": annotated, "Cardboard Mask": mask}


if __name__ == "__main__":
    from camera import run_live

    create_controls()
    print("Sliders in 'Box Controls' set the HSV colour range. Press q to quit.")
    print(f"Defaults: low={HSV_LOW} high={HSV_HIGH}")
    run_live(process_frame)
