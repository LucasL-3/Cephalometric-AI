import os
import json
import math
import shutil
import logging
import cv2
import numpy as np
import requests
from ultralytics import YOLO

logger = logging.getLogger(__name__)


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

# Landmark IDs used in Steiner analysis. Names must match Edit Landmarks page
# (from group_info/landmarks_info_*.json) so Analysis and Edit show the same labels.
_ANALYSIS_LANDMARK_IDS = [
    "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10",
    "L11", "L12", "L13", "L14", "L15", "L16", "L17", "L18", "L19",
    "L24", "L30", "L31", "L32", "L33", "L34", "L37",
]


# Build a master list of all landmark IDs -> display names (from JSON; used by Edit Landmarks page)
def _build_all_landmark_names():
    names = {}
    for gid in range(NUM_GROUPS):
        for lm in _load_group_landmark_info(gid):
            names[lm["id"]] = lm["name"]
    return names

ALL_LANDMARK_NAMES = _build_all_landmark_names()

# Analysis uses same names as Edit Landmarks page (UIe, LIe, UL, LL, Sn, Cm, Pog', isa, iia, U1, L1, etc.)
LANDMARK_ID_TO_ANALYSIS = {
    lid: ALL_LANDMARK_NAMES[lid]
    for lid in _ANALYSIS_LANDMARK_IDS
    if lid in ALL_LANDMARK_NAMES
}


def get_keypoint_names():
    """Return ordered list of display names for the edit UI (sorted by landmark ID)."""
    ids = sorted(ALL_LANDMARK_NAMES.keys(), key=lambda x: int(x[1:]))
    return [f"{lid}: {ALL_LANDMARK_NAMES[lid]}" for lid in ids]


def get_landmark_ids_ordered():
    """Return sorted list of landmark IDs the models know about."""
    return sorted(ALL_LANDMARK_NAMES.keys(), key=lambda x: int(x[1:]))


def coords_for_frontend(all_coords):
    """Convert {landmark_id: Point} → {analysis_name: {x, y}} for JS."""
    result = {}
    for lid, pt in all_coords.items():
        aname = LANDMARK_ID_TO_ANALYSIS.get(lid)
        if aname:
            result[aname] = {"x": pt.x, "y": pt.y}
    return result


def extra_coords_for_frontend(all_coords):
    """Return coords for landmarks NOT used in analysis, keyed by landmark ID."""
    result = {}
    for lid, pt in all_coords.items():
        if lid not in LANDMARK_ID_TO_ANALYSIS:
            label = ALL_LANDMARK_NAMES.get(lid, lid)
            result[lid] = {"x": pt.x, "y": pt.y, "label": label}
    return result


def get_used_landmark_ids():
    """Return the set of landmark IDs that participate in analysis."""
    return set(LANDMARK_ID_TO_ANALYSIS.keys())


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


def _project_onto_line(pt, line_a, line_b):
    """Project *pt* onto the infinite line through *line_a* → *line_b*."""
    dx = line_b.x - line_a.x
    dy = line_b.y - line_a.y
    d2 = dx * dx + dy * dy
    if d2 < 1e-9:
        return line_a.x, line_a.y
    t = ((pt.x - line_a.x) * dx + (pt.y - line_a.y) * dy) / d2
    return line_a.x + t * dx, line_a.y + t * dy


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
    Comprehensive cephalometric analysis (Steiner + Tweed + Downs + soft-tissue).
    Returns (skeletal_class, measurements).
    Measurements list is ordered: skeletal → dental → soft tissue.
    """
    results = []
    skeletal_class = "N/A"

    # ------------------------------------------------------------------
    # Retrieve all landmarks
    # ------------------------------------------------------------------
    S  = _lm(all_coords, "S")
    N  = _lm(all_coords, "N")
    A  = _lm(all_coords, "A")
    B  = _lm(all_coords, "B")
    Go = _lm(all_coords, "Go")
    Me = _lm(all_coords, "Me")
    Or = _lm(all_coords, "Or")
    Po = _lm(all_coords, "Po")

    # Incisor edges (ULE/LLE): use UIe and LIe — same names as Edit Landmarks page
    UIe = _lm(all_coords, "UIe")
    LIe = _lm(all_coords, "LIe")
    isa = _lm(all_coords, "isa")
    iia = _lm(all_coords, "iia")

    molar = _lm(all_coords, "msc")
    if molar is None:
        molar = _lm(all_coords, "MB1")

    Pog_soft = _lm(all_coords, "Pog'")
    Cm       = _lm(all_coords, "Cm")
    if Cm is None:
        Cm = _lm(all_coords, "Sn")
    Sn = _lm(all_coords, "Sn")
    UL = _lm(all_coords, "UL")
    LL = _lm(all_coords, "LL")

    # Occlusal plane: anterior = midpoint of incisor edges (UIe, LIe), posterior = molar cusp
    op_ant = None
    if UIe and LIe:
        op_ant = Point((UIe.x + LIe.x) / 2,
                       (UIe.y + LIe.y) / 2)

    # ======================================================================
    # SKELETAL  (6 measurements)
    # ======================================================================

    # 1. SNA
    if N and A and S:
        SNA_val = Angle(Vector(N, S), Vector(N, A)).theta()
        results.append(_interpret_angle("SNA", SNA_val, 78, 86))
    else:
        SNA_val = None
        results.append(["SNA", 78, 0, 86])

    # 2. SNB
    if N and B and S:
        SNB_val = Angle(Vector(N, S), Vector(N, B)).theta()
        results.append(_interpret_angle("SNB", SNB_val, 76, 84))
    else:
        SNB_val = None
        results.append(["SNB", 76, 0, 84])

    # 3. ANB + skeletal class
    if SNA_val is not None and SNB_val is not None:
        ANB_val = SNA_val - SNB_val
        if ANB_val < 0:
            skeletal_class = "Class III"
        elif ANB_val > 4:
            skeletal_class = "Class II"
        else:
            skeletal_class = "Class I"
        results.append(["ANB", 0, round(ANB_val, 2), 4])
    else:
        results.append(["ANB", 0, 0, 4])

    # 4. Wits Appraisal  (Jacobson, norm ≈ 0 ± 1 mm)
    if A and B and op_ant and molar:
        ax, ay = _project_onto_line(A, molar, op_ant)
        bx, by = _project_onto_line(B, molar, op_ant)
        op_dx = op_ant.x - molar.x
        op_dy = op_ant.y - molar.y
        op_len = math.hypot(op_dx, op_dy)
        if op_len > 1e-9:
            wits_px = ((ax - bx) * op_dx + (ay - by) * op_dy) / op_len
            wits_mm = wits_px * MM_PER_PIXEL
            results.append(_interpret_linear_mm("Wits(mm)", wits_mm, -1, 1))
        else:
            results.append(["Wits(mm)", -1, 0, 1])
    else:
        results.append(["Wits(mm)", -1, 0, 1])

    # 5. SN-MP  (mandibular plane angle, Steiner)
    if S and N and Go and Me:
        SN_MP = Angle(Vector(S, N), Vector(Go, Me)).theta()
        results.append(_interpret_angle("SN-MP", SN_MP, 26, 38))
    else:
        results.append(["SN-MP", 26, 0, 38])

    # 6. FMA  (Frankfort mandibular angle, Tweed, norm 27° ± 5°)
    if Or and Po and Go and Me:
        FMA_val = Angle(Vector(Po, Or), Vector(Go, Me)).theta()
        results.append(_interpret_angle("FMA", FMA_val, 22, 32))
    else:
        results.append(["FMA", 22, 0, 32])

    # ======================================================================
    # DENTAL  (9 measurements)
    # ======================================================================

    # 7. OP-SN  (occlusal plane to SN)
    if S and N and op_ant and molar:
        OP_SN = Angle(Vector(S, N), Vector(molar, op_ant)).theta()
        results.append(_interpret_angle("OP-SN", OP_SN, 10, 18))
    else:
        results.append(["OP-SN", 10, 0, 18])

    # 8. U1-NA angle  (UIe, isa)
    if N and A and UIe and isa:
        U1_NA_ang = Angle(Vector(N, A), Vector(isa, UIe)).theta()
        results.append(_interpret_angle("U1-NA(°)", U1_NA_ang, 18, 26))
    else:
        results.append(["U1-NA(°)", 18, 0, 26])

    # 9. U1-NA linear
    if N and A and UIe:
        d_mm = abs(point_to_line_distance_signed(UIe.x, UIe.y, N, A)) * MM_PER_PIXEL
        results.append(_interpret_linear_mm("U1-NA(mm)", d_mm, 2, 6))
    else:
        results.append(["U1-NA(mm)", 2, 0, 6])

    # 10. L1-NB angle  (LIe, iia)
    if N and B and LIe and iia:
        L1_NB_ang = Angle(Vector(B, N), Vector(iia, LIe)).theta()
        results.append(_interpret_angle("L1-NB(°)", L1_NB_ang, 20, 30))
    else:
        results.append(["L1-NB(°)", 20, 0, 30])

    # 11. L1-NB linear
    if N and B and LIe:
        d_mm = abs(point_to_line_distance_signed(LIe.x, LIe.y, N, B)) * MM_PER_PIXEL
        results.append(_interpret_linear_mm("L1-NB(mm)", d_mm, 2, 6))
    else:
        results.append(["L1-NB(mm)", 2, 0, 6])

    # 12. IMPA  (lower incisor to mandibular plane; LIe, iia)
    if Go and Me and LIe and iia:
        IMPA_val = Angle(Vector(Go, Me), Vector(iia, LIe)).theta()
        results.append(_interpret_angle("IMPA", IMPA_val, 85, 95))
    else:
        results.append(["IMPA", 85, 0, 95])

    # 13. Interincisal angle  (UIe, LIe, isa, iia)
    if UIe and isa and LIe and iia:
        inter_ang = Angle(Vector(isa, UIe), Vector(iia, LIe)).theta()
        results.append(_interpret_angle("Interincisal", inter_ang, 120, 140))
    else:
        results.append(["Interincisal", 120, 0, 140])

    # 14. Overjet  (horizontal overlap at incisor edges; ULE/LLE = UIe, LIe)
    if UIe and LIe and S and N:
        ant_dx = N.x - S.x
        ant_dy = N.y - S.y
        ant_len = math.hypot(ant_dx, ant_dy)
        if ant_len > 1e-9:
            diff_x = UIe.x - LIe.x
            diff_y = UIe.y - LIe.y
            overjet_px = (diff_x * ant_dx + diff_y * ant_dy) / ant_len
            results.append(_interpret_linear_mm("Overjet(mm)", overjet_px * MM_PER_PIXEL, 1, 4))
        else:
            results.append(["Overjet(mm)", 1, 0, 4])
    else:
        results.append(["Overjet(mm)", 1, 0, 4])

    # 15. Overbite  (vertical overlap at incisor edges; ULE/LLE = UIe, LIe)
    if UIe and LIe and S and N:
        ant_dx = N.x - S.x
        ant_dy = N.y - S.y
        ant_len = math.hypot(ant_dx, ant_dy)
        if ant_len > 1e-9:
            perp_dx = -ant_dy
            perp_dy = ant_dx
            if perp_dy < 0:
                perp_dx, perp_dy = -perp_dx, -perp_dy
            diff_x = UIe.x - LIe.x
            diff_y = UIe.y - LIe.y
            overbite_px = (diff_x * perp_dx + diff_y * perp_dy) / ant_len
            results.append(_interpret_linear_mm("Overbite(mm)", overbite_px * MM_PER_PIXEL, 1, 4))
        else:
            results.append(["Overbite(mm)", 1, 0, 4])
    else:
        results.append(["Overbite(mm)", 1, 0, 4])

    # ======================================================================
    # SOFT TISSUE  (3 measurements)
    # ======================================================================

    # 16. Nasolabial angle  (Cm-Sn-UL, norm 90-110°)
    if Sn and Cm and UL:
        nasol = Angle(Vector(Sn, Cm), Vector(Sn, UL)).theta()
        results.append(_interpret_angle("Nasolabial(°)", nasol, 90, 110))
    else:
        results.append(["Nasolabial(°)", 90, 0, 110])

    # 17. Upper Lip to S-Line  (norm -2 to 2 mm)
    if Pog_soft and Cm and UL:
        ul_mm = point_to_line_distance_signed(UL.x, UL.y, Pog_soft, Cm) * MM_PER_PIXEL
        results.append(["UL-SLine(mm)", -2, round(ul_mm, 2), 2])
    else:
        results.append(["UL-SLine(mm)", -2, 0, 2])

    # 18. Lower Lip to S-Line  (norm -2 to 2 mm)
    if Pog_soft and Cm and LL:
        ll_mm = point_to_line_distance_signed(LL.x, LL.y, Pog_soft, Cm) * MM_PER_PIXEL
        results.append(["LL-SLine(mm)", -2, round(ll_mm, 2), 2])
    else:
        results.append(["LL-SLine(mm)", -2, 0, 2])

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
        ["Wits(mm)", -1, 0, 1],
        ["SN-MP", 26, 0, 38],
        ["FMA", 22, 0, 32],
        ["OP-SN", 10, 0, 18],
        ["U1-NA(°)", 18, 0, 26],
        ["U1-NA(mm)", 2, 0, 6],
        ["L1-NB(°)", 20, 0, 30],
        ["L1-NB(mm)", 2, 0, 6],
        ["IMPA", 85, 0, 95],
        ["Interincisal", 120, 0, 140],
        ["Overjet(mm)", 1, 0, 4],
        ["Overbite(mm)", 1, 0, 4],
        ["Nasolabial(°)", 90, 0, 110],
        ["UL-SLine(mm)", -2, 0, 2],
        ["LL-SLine(mm)", -2, 0, 2],
    ]


def _interpret_measurements_fallback(measurements):
    """Return two lists of interpretation strings: [clinical, layman] per measurement."""
    clinical = []
    layman = []
    for row in measurements:
        name, lo, val, hi = row[0], row[1], row[2], row[3]
        if val == 0:
            clinical.append("No data available.")
            layman.append("No data available.")
            continue

        if name == "SNA":
            if val > hi:
                clinical.append("Maxillary prognathism — the maxilla is positioned anteriorly relative to the cranial base.")
                layman.append("The upper jaw is further forward than normal.")
            elif val < lo:
                clinical.append("Maxillary retrognathism — the maxilla is positioned posteriorly relative to the cranial base.")
                layman.append("The upper jaw is further back than normal.")
            else:
                clinical.append("Normal maxillary anteroposterior position relative to the cranial base.")
                layman.append("The upper jaw is in a normal position.")

        elif name == "SNB":
            if val > hi:
                clinical.append("Mandibular prognathism — the mandible is positioned anteriorly relative to the cranial base.")
                layman.append("The lower jaw is further forward than normal.")
            elif val < lo:
                clinical.append("Mandibular retrognathism — the mandible is positioned posteriorly relative to the cranial base.")
                layman.append("The lower jaw is further back than normal, which can cause a receding chin.")
            else:
                clinical.append("Normal mandibular anteroposterior position relative to the cranial base.")
                layman.append("The lower jaw is in a normal position.")

        elif name == "ANB":
            if val > hi:
                clinical.append("Skeletal Class II relationship — the maxilla is anterior to the mandible beyond normal limits.")
                layman.append("The upper jaw is ahead of the lower jaw, which can look like an overbite.")
            elif val < lo:
                clinical.append("Skeletal Class III relationship — the mandible is anterior to the maxilla.")
                layman.append("The lower jaw is ahead of the upper jaw, which can look like an underbite.")
            else:
                clinical.append("Skeletal Class I relationship — normal anteroposterior jaw relationship.")
                layman.append("The upper and lower jaws are well-aligned relative to each other.")

        elif name == "Wits(mm)":
            if val > hi:
                clinical.append("Positive Wits — maxilla is anteriorly positioned relative to the mandible on the occlusal plane, suggesting a Class II skeletal tendency.")
                layman.append("The upper jaw sits further forward than the lower jaw when measured along the biting plane, which can contribute to an overbite.")
            elif val < lo:
                clinical.append("Negative Wits — mandible is anteriorly positioned relative to the maxilla on the occlusal plane, suggesting a Class III skeletal tendency.")
                layman.append("The lower jaw sits further forward than the upper jaw along the biting plane, which can look like an underbite.")
            else:
                clinical.append("Normal Wits appraisal — balanced sagittal jaw relationship independent of the cranial base.")
                layman.append("The upper and lower jaws line up well when checked along the biting surface.")

        elif name == "SN-MP":
            if val > hi:
                clinical.append("Hyperdivergent pattern — increased mandibular plane angle indicating a long face tendency and vertical growth pattern.")
                layman.append("The lower jaw is angled steeply, which can cause a longer-looking face and an open bite.")
            elif val < lo:
                clinical.append("Hypodivergent pattern — decreased mandibular plane angle indicating a short face tendency and horizontal growth pattern.")
                layman.append("The lower jaw is flatter than normal, which can cause a shorter-looking face and a deep bite.")
            else:
                clinical.append("Normal mandibular plane inclination — balanced vertical facial proportions.")
                layman.append("The angle of the lower jaw is normal, giving balanced face height.")

        elif name == "FMA":
            if val > hi:
                clinical.append("High FMA — steep mandibular plane relative to the Frankfort horizontal, indicating a hyperdivergent (long face) growth pattern.")
                layman.append("The lower jaw angles downward more than normal, which can make the face look longer and may cause an open bite.")
            elif val < lo:
                clinical.append("Low FMA — flat mandibular plane relative to the Frankfort horizontal, indicating a hypodivergent (short face) growth pattern.")
                layman.append("The lower jaw is very flat, which can make the face look shorter and may cause a deep bite.")
            else:
                clinical.append("Normal Frankfort mandibular angle — balanced vertical growth pattern.")
                layman.append("The angle between the ear-eye line and the lower jaw is normal, indicating balanced face height.")

        elif name == "OP-SN":
            if val > hi:
                clinical.append("Steep occlusal plane — increased inclination of the occlusal plane relative to the cranial base.")
                layman.append("The biting surface of the teeth is tilted more steeply than normal.")
            elif val < lo:
                clinical.append("Flat occlusal plane — decreased inclination of the occlusal plane relative to the cranial base.")
                layman.append("The biting surface of the teeth is flatter than normal.")
            else:
                clinical.append("Normal occlusal plane inclination relative to the cranial base.")
                layman.append("The angle of the biting surface is normal.")

        elif name == "U1-NA(°)":
            if val > hi:
                clinical.append("Proclined upper incisors — the maxillary incisors are tilted labially (forward) relative to the NA plane.")
                layman.append("The upper front teeth are tilted too far forward.")
            elif val < lo:
                clinical.append("Retroclined upper incisors — the maxillary incisors are tilted lingually (backward) relative to the NA plane.")
                layman.append("The upper front teeth are tilted too far backward.")
            else:
                clinical.append("Normal upper incisor inclination relative to the NA plane.")
                layman.append("The upper front teeth are tilted at a normal angle.")

        elif name == "U1-NA(mm)":
            if val > hi:
                clinical.append("Upper incisors positioned anteriorly — the maxillary incisors are protruding beyond normal distance from the NA line.")
                layman.append("The upper front teeth stick out further forward than normal.")
            elif val < lo:
                clinical.append("Upper incisors positioned posteriorly — the maxillary incisors are behind the normal distance from the NA line.")
                layman.append("The upper front teeth are set further back than normal.")
            else:
                clinical.append("Normal upper incisor anteroposterior position relative to the NA line.")
                layman.append("The upper front teeth are positioned normally (not sticking out).")

        elif name == "L1-NB(°)":
            if val > hi:
                clinical.append("Proclined lower incisors — the mandibular incisors are tilted labially (forward) relative to the NB plane.")
                layman.append("The lower front teeth are tilted too far forward.")
            elif val < lo:
                clinical.append("Retroclined lower incisors — the mandibular incisors are tilted lingually (backward) relative to the NB plane.")
                layman.append("The lower front teeth are tilted too far backward.")
            else:
                clinical.append("Normal lower incisor inclination relative to the NB plane.")
                layman.append("The lower front teeth are tilted at a normal angle.")

        elif name == "L1-NB(mm)":
            if val > hi:
                clinical.append("Lower incisors positioned anteriorly — the mandibular incisors are protruding beyond normal distance from the NB line.")
                layman.append("The lower front teeth stick out further forward than normal.")
            elif val < lo:
                clinical.append("Lower incisors positioned posteriorly — the mandibular incisors are behind the normal distance from the NB line.")
                layman.append("The lower front teeth are set further back than normal.")
            else:
                clinical.append("Normal lower incisor anteroposterior position relative to the NB line.")
                layman.append("The lower front teeth are positioned normally.")

        elif name == "IMPA":
            if val > hi:
                clinical.append("Proclined lower incisors relative to the mandibular plane — the lower incisors are flared labially.")
                layman.append("The lower front teeth lean too far forward relative to the jawbone, which can make them look protruding.")
            elif val < lo:
                clinical.append("Retroclined lower incisors relative to the mandibular plane — the lower incisors are tilted lingually.")
                layman.append("The lower front teeth lean too far backward relative to the jawbone.")
            else:
                clinical.append("Normal lower incisor inclination relative to the mandibular plane.")
                layman.append("The lower front teeth stand at a normal angle to the jawbone.")

        elif name == "Interincisal":
            if val > hi:
                clinical.append("Increased interincisal angle — the upper and lower incisors are upright or retroclined, reducing overbite and overjet.")
                layman.append("The upper and lower front teeth are tilted too far back toward each other, making the bite flatter.")
            elif val < lo:
                clinical.append("Decreased interincisal angle — the upper and lower incisors are proclined, increasing protrusion.")
                layman.append("The upper and lower front teeth are both tilted too far forward, making them stick out.")
            else:
                clinical.append("Normal interincisal angle — balanced upper and lower incisor relationship.")
                layman.append("The angle between the upper and lower front teeth is normal.")

        elif name == "Overjet(mm)":
            if val > hi:
                clinical.append("Increased overjet — the maxillary incisors are positioned anteriorly to the mandibular incisors beyond normal limits, common in Class II malocclusion.")
                layman.append("The upper front teeth stick out significantly further than the lower front teeth (often called 'buck teeth').")
            elif val < lo:
                if val < 0:
                    clinical.append("Negative overjet (anterior crossbite) — the mandibular incisors are positioned anterior to the maxillary incisors.")
                    layman.append("The lower front teeth are ahead of the upper front teeth, which is the reverse of normal.")
                else:
                    clinical.append("Reduced overjet — the horizontal overlap between upper and lower incisors is minimal.")
                    layman.append("The upper and lower front teeth are nearly edge-to-edge horizontally.")
            else:
                clinical.append("Normal overjet — the maxillary incisors overlap the mandibular incisors horizontally within normal limits.")
                layman.append("The upper front teeth sit a normal distance ahead of the lower front teeth.")

        elif name == "Overbite(mm)":
            if val > hi:
                clinical.append("Deep overbite — excessive vertical overlap of the maxillary incisors over the mandibular incisors, which may cause lower incisor impingement on the palate.")
                layman.append("The upper front teeth cover too much of the lower front teeth (deep bite), which can cause the lower teeth to bite into the roof of the mouth.")
            elif val < lo:
                if val < 0:
                    clinical.append("Anterior open bite — the maxillary and mandibular incisors do not overlap vertically, leaving a gap when the back teeth are together.")
                    layman.append("There is a gap between the upper and lower front teeth when biting down (open bite).")
                else:
                    clinical.append("Reduced overbite — minimal vertical overlap of the incisors.")
                    layman.append("The upper front teeth barely overlap the lower front teeth vertically.")
            else:
                clinical.append("Normal overbite — the maxillary incisors overlap the mandibular incisors vertically within normal limits.")
                layman.append("The upper front teeth overlap the lower front teeth by a normal amount.")

        elif name == "Nasolabial(°)":
            if val > hi:
                clinical.append("Obtuse nasolabial angle — may indicate maxillary retrusion, retroclined upper incisors, or a short upper lip.")
                layman.append("The angle between the nose and upper lip is larger than normal, which can make the upper lip look flat or pushed back.")
            elif val < lo:
                clinical.append("Acute nasolabial angle — may indicate maxillary protrusion, proclined upper incisors, or a prominent upper lip.")
                layman.append("The angle between the nose and upper lip is smaller than normal, which can make the lip or teeth look like they stick out.")
            else:
                clinical.append("Normal nasolabial angle — balanced relationship between the nose and upper lip.")
                layman.append("The angle between the nose and upper lip is normal, giving a balanced profile.")

        elif name == "UL-SLine(mm)":
            if val > hi:
                clinical.append("Upper lip protrusion — the upper lip is anterior to the S line, indicating lip prominence or maxillary dental protrusion.")
                layman.append("The upper lip sticks out beyond the ideal profile line.")
            elif val < lo:
                clinical.append("Upper lip retrusion — the upper lip is posterior to the S line, which may indicate a flat midface profile.")
                layman.append("The upper lip is set further back than the ideal profile line.")
            else:
                clinical.append("Normal upper lip position relative to the S line — balanced soft tissue profile.")
                layman.append("The upper lip is in a normal position, giving a balanced side profile.")

        elif name == "LL-SLine(mm)":
            if val > hi:
                clinical.append("Lower lip protrusion — the lower lip is anterior to the S line, indicating lip prominence.")
                layman.append("The lower lip sticks out beyond the ideal profile line.")
            elif val < lo:
                clinical.append("Lower lip retrusion — the lower lip is posterior to the S line, indicating a flat or retrusive lip profile.")
                layman.append("The lower lip is set further back than the ideal profile line.")
            else:
                clinical.append("Normal lower lip position relative to the S line — balanced soft tissue profile.")
                layman.append("The lower lip is in a normal position, giving a balanced side profile.")

        else:
            clinical.append("")
            layman.append("")

    return clinical, layman


# =============================================================================
# LLM-powered interpretation via Groq (Llama 3.3 70B)
# =============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"

_SYSTEM_PROMPT = """You are an expert orthodontist and cephalometric analyst. You will receive a table of cephalometric measurements with their normal ranges, the patient's values, and each measurement's status (NORMAL or ABNORMAL). You will also receive the skeletal classification determined by the system.

STRICT RULES FOR SUMMARIES:
- You MUST explicitly discuss every measurement that is marked ABNORMAL in your summaries. Do not give a generic "relatively normal" summary when any key finding is abnormal.
- When Overjet(mm) or Overbite(mm) are ABNORMAL, you MUST address them clearly in BOTH summary_clinical and summary_patient — e.g. increased overjet, deep overbite, open bite, etc. These are highly visible and clinically important; never gloss over them.
- When any skeletal, dental, or soft-tissue measurement is ABNORMAL, name it and explain its significance in the summaries. Prioritize abnormal findings over normal ones.
- Only describe the case as "normal" or "relatively normal" if the vast majority of measurements are NORMAL and no major findings (skeletal class, overjet, overbite, key angles) are abnormal.

Use the provided statuses and skeletal classification as your baseline. If the overall pattern gives you strong clinical reason to nuance an individual status, you may do so but must explain why. Analyze ALL measurements together for a complete clinical picture.

Return your response as valid JSON with exactly this structure:
{
  "clinical": ["interpretation for measurement 1", "interpretation for measurement 2", ...],
  "layman": ["plain-english explanation 1", "plain-english explanation 2", ...],
  "summary_clinical": "A 2-4 sentence holistic clinical summary. Must explicitly mention all ABNORMAL findings and their significance.",
  "summary_patient": "A 2-4 sentence summary in simple plain English. Must clearly explain any overbite/overjet or other abnormal findings in everyday terms."
}

CRITICAL — array length and order:
- The table you receive has a fixed number of measurement rows. Your "clinical" and "layman" arrays MUST each contain exactly that many entries — one string per row, in the same order as the table. No fewer, no more.
- Entry 1 in clinical and layman = first row in the table, entry 2 = second row, and so on. Do not skip, merge, or reorder.
- Before returning, verify: len(clinical) == number of table rows and len(layman) == number of table rows.

Rules:
- Clinical interpretations: proper orthodontic terminology; reference how each finding relates to the overall pattern.
- Layman interpretations: simple enough for a patient with no medical background.
- If a measurement value is 0, say "No data available." for both clinical and layman.
- Keep each individual interpretation to 1-2 sentences.
- Do NOT include markdown or code fences — only raw JSON."""


def interpret_measurements_llm(measurements, skeletal_class="N/A"):
    """Call Groq (Llama 3.3 70B) to interpret measurements holistically.
    Returns (clinical_list, layman_list, summary_str).
    Falls back to hard-coded interpretations on failure.
    """
    if not GROQ_API_KEY:
        logger.warning("GROQ_API_KEY not set — using fallback interpretations")
        c, l = _interpret_measurements_fallback(measurements)
        return c, l, "", ""

    table_lines = ["Measurement | Min | Value | Max | Status"]
    table_lines.append("---|---|---|---|---")
    for row in measurements:
        name, lo, val, hi = row[0], row[1], row[2], row[3]
        if val == 0:
            status = "NO DATA"
        elif lo <= val <= hi:
            status = "NORMAL"
        else:
            status = "ABNORMAL"
        table_lines.append(f"{name} | {lo} | {val} | {hi} | {status}")
    table_str = "\n".join(table_lines)

    n = len(measurements)
    order_list = ", ".join(row[0] for row in measurements)
    user_msg = (
        f"Skeletal classification (determined by system): {skeletal_class}\n\n"
        f"Cephalometric measurements ({n} rows):\n\n{table_str}\n\n"
        f"Required: Output exactly {n} entries in 'clinical' and exactly {n} entries in 'layman', in this order: {order_list}. "
        f"One string per measurement, same order as the table. Respect the Status column and skeletal classification above."
    )

    payload = {
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"},
    }

    try:
        resp = requests.post(
            GROQ_URL,
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()

        text = result["choices"][0]["message"]["content"].strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        parsed = json.loads(text)
        clinical = parsed["clinical"]
        layman = parsed["layman"]
        summary_clinical = parsed.get("summary_clinical", "")
        summary_patient = parsed.get("summary_patient", "")

        if len(clinical) != len(measurements) or len(layman) != len(measurements):
            logger.warning("LLM returned wrong number of interpretations, using fallback")
            c, l = _interpret_measurements_fallback(measurements)
            return c, l, summary_clinical, summary_patient

        return clinical, layman, summary_clinical, summary_patient

    except Exception as e:
        logger.warning("Groq API call failed (%s), using fallback", e)
        c, l = _interpret_measurements_fallback(measurements)
        return c, l, "", ""


def interpret_measurements(measurements, skeletal_class="N/A"):
    """Public wrapper — tries LLM first, falls back to hard-coded."""
    return interpret_measurements_llm(measurements, skeletal_class=skeletal_class)


# =============================================================================
# Visualization: draw detected landmarks on the image
# =============================================================================

GROUP_COLORS = {
    0: (0, 0, 255),    # Red (BGR)
    1: (0, 255, 0),    # Green
    2: (255, 0, 0),    # Blue
    3: (255, 255, 0),  # Cyan
}

GROUP_COLORS_HEX = {
    0: '#ff0000',
    1: '#00ff00',
    2: '#0000ff',
    3: '#00ffff',
}

_LID_TO_GROUP = {}
for _gid in range(NUM_GROUPS):
    for _lminfo in _load_group_landmark_info(_gid):
        _LID_TO_GROUP[_lminfo["id"]] = _gid


def get_landmark_colors():
    """Return {landmark_id: hex_color} for all known landmarks."""
    return {lid: GROUP_COLORS_HEX.get(gid, '#ffffff') for lid, gid in _LID_TO_GROUP.items()}


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
