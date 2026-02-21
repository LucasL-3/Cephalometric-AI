import os
import json
import math
import shutil
import cv2
import numpy as np
from ultralytics import YOLO


# =============================================================================
# New multi-model landmark detection (68 landmarks across 4 groups)
# =============================================================================

NEW_MODEL_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "home", "ubuntu", "cephalometric_analysis",
)

NUM_GROUPS = 4


def _load_group_landmark_info(group_id):
    info_path = os.path.join(NEW_MODEL_ROOT, f"group_info/landmarks_info_{group_id}.json")
    if not os.path.exists(info_path):
        return []
    with open(info_path, "r") as f:
        data = json.load(f)
    return sorted(data.get("landmarks", []), key=lambda x: x["id"])


def _load_all_models():
    models = {}
    for gid in range(NUM_GROUPS):
        wpath = os.path.join(NEW_MODEL_ROOT, f"runs/pose/train_model_{gid}/weights/best.pt")
        if os.path.isfile(wpath):
            models[gid] = YOLO(wpath)
    return models


MODELS = _load_all_models()

# Mapping from new-model landmark IDs to Steiner analysis internal names.
# The new model uses L1–L68; Steiner analysis needs: S, N, A, B, Go, Me,
# U1_tip, U1_apex, L1_tip, L1_apex, SoftPog, Columella/Subnasale, LowerLip,
# Or (for occlusal plane anterior proxy), Po (for occlusal plane posterior proxy).
LANDMARK_ID_TO_ANALYSIS = {
    "L1": "S",          # Sella
    "L2": "N",          # Nasion
    "L3": "Or",         # Orbitale  (OP_ant proxy)
    "L4": "Po",         # Porion    (OP_post proxy)
    "L5": "A",          # A-point / Subspinale
    "L6": "B",          # B-point / Supramentale
    "L7": "Pogonion",   # Pogonion (hard tissue)
    "L8": "Me",         # Menton
    "L9": "Gnathion",   # Gnathion
    "L10": "Go",        # Gonion
    "L11": "L1_tip",    # Lower incisor edge (LIe)
    "L12": "U1_tip",    # Upper incisor edge (UIe)
    "L13": "UpperLip",  # Upper Lip
    "L14": "LowerLip",  # Lower Lip
    "L15": "Sn",        # Subnasale
    "L16": "SoftPog",   # Soft tissue pogonion (Pog')
    "L17": "PNS",       # Posterior nasal spine
    "L18": "ANS",       # Anterior nasal spine
    "L19": "Ar",        # Articulare
    "L24": "Columella",  # Columella
    "L30": "U1_root",   # Upper incisor root (U1)
    "L31": "L1_root",   # Lower incisor root (L1)
    "L32": "U1_apex",   # Upper incisor apex (isa)
    "L33": "L1_apex",   # Lower incisor apex (iia)
    "L34": "msc",        # Molar superior cusp (occlusal plane posterior)
    "L37": "MB1",        # Mesiobuccal cusp (occlusal plane posterior alt)
}

# Build a master list of all landmark IDs -> display names
def _build_all_landmark_names():
    names = {}
    for gid in range(NUM_GROUPS):
        for lm in _load_group_landmark_info(gid):
            names[lm["id"]] = lm["name"]
    return names

ALL_LANDMARK_NAMES = _build_all_landmark_names()


def get_keypoint_names():
    """Return ordered list of display names for the edit UI (sorted by landmark ID)."""
    ids = sorted(ALL_LANDMARK_NAMES.keys(), key=lambda x: int(x[1:]))
    return [f"{lid}: {ALL_LANDMARK_NAMES[lid]}" for lid in ids]


def get_landmark_ids_ordered():
    """Return sorted list of landmark IDs the models know about."""
    return sorted(ALL_LANDMARK_NAMES.keys(), key=lambda x: int(x[1:]))


# =============================================================================
# Geometry helpers (unchanged)
# =============================================================================

MM_PER_PIXEL = 0.1


class Point:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
    def __str__(self):
        return f"{self.x},{self.y}"


class Vector:
    def __init__(self, pa, pb):
        self.x = pb.x - pa.x
        self.y = pb.y - pa.y


class Angle:
    def __init__(self, va, vb):
        self.va = va
        self.vb = vb
    def theta(self):
        dot = self.va.x * self.vb.x + self.va.y * self.vb.y
        ma = math.hypot(self.va.x, self.va.y)
        mb = math.hypot(self.vb.x, self.vb.y)
        if ma < 1e-9 or mb < 1e-9:
            return 0.0
        return math.degrees(math.acos(max(-1, min(1, dot / (ma * mb)))))


def point_to_line_distance_signed(px, py, l1, l2):
    lx = l2.x - l1.x
    ly = l2.y - l1.y
    length = math.hypot(lx, ly)
    if length < 1e-9:
        return 0.0
    qx = l1.x - px
    qy = l1.y - py
    cross = qx * ly - qy * lx
    return cross / length


def _interpret_angle(name, value, low, high):
    return [name, low, round(value, 2), high]


def _interpret_linear_mm(name, value_mm, low, high):
    return [name, low, round(value_mm, 2), high]


# =============================================================================
# Multi-model inference
# =============================================================================

def _run_group_inference(model, image_path, group_id):
    """Run one group model and return {landmark_id: Point}."""
    landmarks_info = _load_group_landmark_info(group_id)
    results = model(image_path, verbose=False)
    coords = {}
    if not results:
        return coords
    result = results[0]
    if hasattr(result, "keypoints") and result.keypoints is not None and result.keypoints.data.shape[1] > 0:
        kpts = result.keypoints.data[0].cpu().numpy()
        for i, lm in enumerate(landmarks_info):
            if i < len(kpts):
                x, y = float(kpts[i][0]), float(kpts[i][1])
                if x > 0 or y > 0:
                    coords[lm["id"]] = Point(x, y)
    return coords


def detect_all_landmarks(image_path):
    """Run all 4 group models and return {landmark_id: Point}."""
    all_coords = {}
    for gid, model in MODELS.items():
        group_coords = _run_group_inference(model, image_path, gid)
        all_coords.update(group_coords)
    return all_coords


def _lm(all_coords, analysis_name):
    """Look up a Point by its Steiner analysis name (e.g. 'S', 'N', 'A')."""
    for lid, aname in LANDMARK_ID_TO_ANALYSIS.items():
        if aname == analysis_name and lid in all_coords:
            return all_coords[lid]
    return None


# =============================================================================
# Steiner analysis (works on landmark dict instead of index-based file)
# =============================================================================

def analysis_from_landmarks(all_coords):
    """
    Steiner analysis from detected landmarks dict.
    Returns (skeletal_class, measurements).
    """
    results = []
    skeletal_class = "N/A"

    S = _lm(all_coords, "S")
    N = _lm(all_coords, "N")
    A = _lm(all_coords, "A")
    B = _lm(all_coords, "B")
    Go = _lm(all_coords, "Go")
    Me = _lm(all_coords, "Me")
    Or = _lm(all_coords, "Or")
    Po = _lm(all_coords, "Po")

    # 1. SNA
    if N and A and S:
        SNA = Angle(Vector(N, S), Vector(N, A)).theta()
        results.append(_interpret_angle("SNA", SNA, 78, 86))
    else:
        results.append(["SNA", 78, 0, 86])

    # 2. SNB
    if N and B and S:
        SNB = Angle(Vector(N, S), Vector(N, B)).theta()
        results.append(_interpret_angle("SNB", SNB, 76, 84))
    else:
        results.append(["SNB", 76, 0, 84])

    # 3. ANB + skeletal class
    if N and A and B and S:
        SNA_val = Angle(Vector(N, S), Vector(N, A)).theta()
        SNB_val = Angle(Vector(N, S), Vector(N, B)).theta()
        ANB = SNA_val - SNB_val
        if ANB < 0:
            skeletal_class = "Class III"
        elif ANB > 4:
            skeletal_class = "Class II"
        else:
            skeletal_class = "Class I"
        results.append(["ANB", 0, round(ANB, 2), 4])
    else:
        results.append(["ANB", 0, 0, 4])

    # 4. SN-MP (mandibular plane angle)
    if S and N and Go and Me:
        SN_MP = Angle(Vector(S, N), Vector(Go, Me)).theta()
        results.append(_interpret_angle("SN-MP", SN_MP, 26, 38))
    else:
        results.append(["SN-MP", 26, 0, 38])

    # 5. OP-SN (occlusal plane to SN)
    # Occlusal plane: anterior = midpoint of upper & lower incisor edges,
    #                 posterior = molar superior cusp (msc) or MB1
    U1_edge = _lm(all_coords, "U1_tip")
    L1_edge = _lm(all_coords, "L1_tip")
    molar = _lm(all_coords, "msc")
    if molar is None:
        molar = _lm(all_coords, "MB1")
    op_ant = None
    if U1_edge and L1_edge:
        op_ant = Point((U1_edge.x + L1_edge.x) / 2, (U1_edge.y + L1_edge.y) / 2)
    if S and N and op_ant and molar:
        OP_SN = Angle(Vector(S, N), Vector(molar, op_ant)).theta()
        results.append(_interpret_angle("OP-SN", OP_SN, 10, 18))
    else:
        results.append(["OP-SN", 10, 0, 18])

    # Dental landmarks
    U1_tip = _lm(all_coords, "U1_tip")
    U1_apex = _lm(all_coords, "U1_apex")
    L1_tip = _lm(all_coords, "L1_tip")
    L1_apex = _lm(all_coords, "L1_apex")

    # 6. U1-NA angle
    if N and A and U1_tip and U1_apex:
        U1_NA_ang = Angle(Vector(N, A), Vector(U1_apex, U1_tip)).theta()
        results.append(_interpret_angle("U1-NA(°)", U1_NA_ang, 18, 26))
    else:
        results.append(["U1-NA(°)", 18, 0, 26])

    # 7. U1-NA linear
    if N and A and U1_tip:
        d_mm = abs(point_to_line_distance_signed(U1_tip.x, U1_tip.y, N, A)) * MM_PER_PIXEL
        results.append(_interpret_linear_mm("U1-NA(mm)", d_mm, 2, 6))
    else:
        results.append(["U1-NA(mm)", 2, 0, 6])

    # 8. L1-NB angle
    if N and B and L1_tip and L1_apex:
        L1_NB_ang = Angle(Vector(B, N), Vector(L1_apex, L1_tip)).theta()
        results.append(_interpret_angle("L1-NB(°)", L1_NB_ang, 20, 30))
    else:
        results.append(["L1-NB(°)", 20, 0, 30])

    # 9. L1-NB linear
    if N and B and L1_tip:
        d_mm = abs(point_to_line_distance_signed(L1_tip.x, L1_tip.y, N, B)) * MM_PER_PIXEL
        results.append(_interpret_linear_mm("L1-NB(mm)", d_mm, 2, 6))
    else:
        results.append(["L1-NB(mm)", 2, 0, 6])

    # 10. Interincisal angle
    if U1_tip and U1_apex and L1_tip and L1_apex:
        inter_ang = Angle(Vector(U1_apex, U1_tip), Vector(L1_apex, L1_tip)).theta()
        results.append(_interpret_angle("Interincisal", inter_ang, 120, 140))
    else:
        results.append(["Interincisal", 120, 0, 140])

    # 11. S Line (soft tissue)
    SoftPog = _lm(all_coords, "SoftPog")
    Columella = _lm(all_coords, "Columella")
    if Columella is None:
        Columella = _lm(all_coords, "Sn")
    LowerLip = _lm(all_coords, "LowerLip")

    if SoftPog and Columella and LowerLip:
        d_mm = point_to_line_distance_signed(LowerLip.x, LowerLip.y, SoftPog, Columella) * MM_PER_PIXEL
        results.append(["S Line(mm)", -2, round(d_mm, 2), 2])
    else:
        results.append(["S Line(mm)", -2, 0, 2])

    return skeletal_class, results


# Legacy wrapper for the file-based analysis (edit landmarks still uses files)
def analysis(filename):
    coords = read_points_file(filename)
    return analysis_from_landmarks(coords)


def get_placeholder_measurements():
    return [
        ["SNA", 78, 0, 86],
        ["SNB", 76, 0, 84],
        ["ANB", 0, 0, 4],
        ["SN-MP", 26, 0, 38],
        ["OP-SN", 10, 0, 18],
        ["U1-NA(°)", 18, 0, 26],
        ["U1-NA(mm)", 2, 0, 6],
        ["L1-NB(°)", 20, 0, 30],
        ["L1-NB(mm)", 2, 0, 6],
        ["Interincisal", 120, 0, 140],
        ["S Line(mm)", -2, 0, 2],
    ]


# =============================================================================
# Visualization: draw detected landmarks on the image
# =============================================================================

GROUP_COLORS = {
    0: (0, 0, 255),    # Red (BGR)
    1: (0, 255, 0),    # Green
    2: (255, 0, 0),    # Blue
    3: (255, 255, 0),  # Cyan
}


def create_result_image(image_path, all_coords, output_path):
    """Draw detected landmarks on the image and save."""
    img = cv2.imread(image_path)
    if img is None:
        shutil.copy(image_path, output_path)
        return

    # Build reverse lookup: landmark_id -> group_id
    lid_to_group = {}
    for gid in range(NUM_GROUPS):
        for lm in _load_group_landmark_info(gid):
            lid_to_group[lm["id"]] = gid

    font = cv2.FONT_HERSHEY_SIMPLEX
    for lid, pt in all_coords.items():
        x, y = int(pt.x), int(pt.y)
        gid = lid_to_group.get(lid, 0)
        color = GROUP_COLORS.get(gid, (255, 255, 255))
        cv2.circle(img, (x, y), 4, color, -1)
        label = f"{lid}: {ALL_LANDMARK_NAMES.get(lid, lid)}"
        cv2.putText(img, label, (x + 6, y - 4), font, 0.35, (255, 255, 255), 1)

    cv2.imwrite(output_path, img)


# =============================================================================
# Points file I/O for edit-landmarks (now uses landmark_id,x,y format)
# =============================================================================

def write_points_file(all_coords, filename):
    """Write landmark_id,x,y per line."""
    ids = sorted(all_coords.keys(), key=lambda x: int(x[1:]))
    with open(filename, "w") as f:
        for lid in ids:
            pt = all_coords[lid]
            f.write(f"{lid},{pt.x},{pt.y}\n")


def read_points_file(filename):
    """Read landmark_id,x,y file; return {landmark_id: Point}."""
    coords = {}
    if not os.path.isfile(filename):
        return coords
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            lid = parts[0].strip()
            x = float(parts[1])
            y = float(parts[2])
            coords[lid] = Point(x, y)
    return coords


def read_points_xy(filename):
    """Read points file; return list of [landmark_id, x, y] for edit UI."""
    coords = read_points_file(filename)
    ids = sorted(coords.keys(), key=lambda x: int(x[1:]))
    return [[lid, coords[lid].x, coords[lid].y] for lid in ids]


# =============================================================================
# Main inference entry point (called from app.py)
# =============================================================================

def yolo_inference(image_path, output_path):
    """Run all 4 models, draw landmarks, run Steiner analysis."""
    all_coords = detect_all_landmarks(image_path)

    # Draw result image
    create_result_image(image_path, all_coords, output_path)

    if not all_coords:
        return {"skeletal_class": "N/A", "measurements": []}

    # Save points file for edit page
    points_path = os.path.join(
        os.path.dirname(output_path),
        os.path.basename(output_path).rsplit(".", 1)[0] + "_points.txt",
    )
    write_points_file(all_coords, points_path)

    # Run analysis
    skeletal_class, measurements = analysis_from_landmarks(all_coords)
    return {"skeletal_class": skeletal_class, "measurements": measurements}
