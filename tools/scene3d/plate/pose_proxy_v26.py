"""3D-A v2.6: nearest-bone vertex zones, label pass decoding, coloured proxy and D strength map.

Only the FIVE zones explicitly specified by step 9 are implemented. The request for
six zones is contradictory; no sixth zone is inferred. No models or new packages.
The renderer supplies final MHR70 joints in world coordinates (label_joints.npz: kp).
"""
import json
from pathlib import Path

import numpy as np

ZONES = ("skin", "top", "bottom", "shoes", "hair")
# Transport palette, NOT outfit colours. Black means background. Widely separated
# RGB values allow nearest-palette decoding at interpolated triangle boundaries.
LABEL_RGB = np.array([[0, 0, 0], [255, 0, 0], [0, 255, 0], [0, 0, 255],
                      [255, 255, 0], [255, 0, 255]], dtype=np.uint8)


def config():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "posev2"))
    from placement_config import section
    return section("h3")


def outfit_colors(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    colors = data["outfit_colors"]
    if set(colors) != set(ZONES):
        raise ValueError(f"outfit_colors must contain exactly {ZONES}; sixth zone is unspecified")
    out = np.zeros((len(ZONES) + 1, 3), dtype=float)
    for i, zone in enumerate(ZONES, 1):
        text = colors[zone]
        if not isinstance(text, str) or len(text) != 7 or text[0] != "#":
            raise ValueError(f"invalid outfit colour {zone}: expected #RRGGBB")
        out[i] = [int(text[k:k + 2], 16) for k in (1, 3, 5)]
    return out


def bone_segments(kp):
    """MHR70 joint-to-joint segments, including finger chains (tip -> wrist).

    MHR70 has no separate spine/chest joints: pelvis midpoint -> neck is their
    common segment; neck -> shoulders are the clavicles. No anchor-point NN.
    """
    k = np.asarray(kp, float)
    if k.ndim != 2 or k.shape[0] < 70 or k.shape[1] != 3 or not np.isfinite(k).all():
        raise ValueError("label pass requires finite MHR70 joints, shape >= (70, 3)")
    pelvis = (k[9] + k[10]) / 2
    ears = (k[3] + k[4]) / 2
    segments = [("bottom", k[9], k[10]), ("top", pelvis, k[69]),
                ("skin", ears, k[0]), ("skin", k[3], k[1]),
                ("skin", k[4], k[2]), ("skin", k[69], ears)]
    parts = ["pelvis", "spine", "head", "head", "head", "neck"]
    for sh, el, wr, hip, kn, an, toes, hand in (
        (5, 7, 62, 9, 11, 13, (15, 16, 17), 42),
        (6, 8, 41, 10, 12, 14, (18, 19, 20), 21),
    ):
        segments += [("top", k[69], k[sh]), ("arm", k[sh], k[el]),
                     ("skin", k[el], k[wr]), ("bottom", k[hip], k[kn]),
                     ("bottom", k[kn], k[an])]
        parts += ["clavicle", "arm", "forearm", "thigh", "shin"]
        for toe in toes:
            segments.append(("shoes", k[an], k[toe]))
            parts.append("foot")
        for base in range(hand, hand + 20, 4):
            chain = [wr, base + 3, base + 2, base + 1, base]
            for a, b in zip(chain, chain[1:]):
                segments.append(("skin", k[a], k[b]))
                parts.append("hand")
    return segments, parts


def vertex_zones(verts, kp, cfg=None):
    """Per vertex: closest segment (Euclidean vertex -> clamped bone segment).

    Upper arm's proximal 40% is top. Hair only overrides head vertices that
    are above the eyes and behind the anatomical ear plane, independent of
    world up (lying/reclining heads work too).
    """
    cfg = config() if cfg is None else cfg
    verts, kp = np.asarray(verts, float), np.asarray(kp, float)
    seg, parts = bone_segments(kp)
    best = np.full(len(verts), np.inf)
    nearest = np.zeros(len(verts), dtype=int)
    along = np.zeros(len(verts))
    for i, (_, a, b) in enumerate(seg):
        ab = b - a
        den = float(ab @ ab)
        t = np.clip((verts - a) @ ab / den, 0, 1) if den > np.finfo(float).eps else np.zeros(len(verts))
        dd = ((verts - (a + t[:, None] * ab)) ** 2).sum(1)
        take = dd < best
        best[take], nearest[take], along[take] = dd[take], i, t[take]
    names = np.array([s[0] for s in seg])[nearest]
    arm = names == "arm"
    names[arm & (along <= cfg["upper_arm_top_fraction"])] = "top"
    names[arm & (along > cfg["upper_arm_top_fraction"])] = "skin"
    eye = (kp[1] + kp[2]) / 2
    ear = (kp[3] + kp[4]) / 2
    up = eye - kp[69]
    if np.linalg.norm(up) <= np.finfo(float).eps:
        raise ValueError("degenerate head: neck and eye line coincide")
    up /= np.linalg.norm(up)
    forward = kp[0] - ear
    forward -= (forward @ up) * up
    if np.linalg.norm(forward) <= np.finfo(float).eps:
        raise ValueError("degenerate head: undefined ear plane")
    forward /= np.linalg.norm(forward)
    head = np.array(parts)[nearest] == "head"
    hair = head & ((verts - eye) @ up > 0) & ((verts - ear) @ forward < 0)
    names[hair] = "hair"
    return np.array([ZONES.index(n) + 1 for n in names], dtype=np.uint8)


def decode_labels(rgb, silhouette):
    rgb = np.asarray(rgb, float)
    palette = LABEL_RGB[1:].astype(float)
    labels = np.argmin(((rgb[..., None, :] - palette) ** 2).sum(-1), axis=-1).astype(np.uint8) + 1
    return np.where(silhouette, labels, 0).astype(np.uint8)


def proxy_image(playblast, plate, silhouette, labels, colors):
    """Zone colour * grey-playblast luminance / mean mannequin luminance.

    Outside the silhouette retain the playblast's shadow over the plate exactly;
    no second shadow application. Plate and playblast already share the camera.
    """
    pb, plate = np.asarray(playblast, float), np.asarray(plate, float)
    sil = np.asarray(silhouette, bool)
    if not sil.any():
        raise ValueError("empty mannequin silhouette")
    if pb.shape != plate.shape or labels.shape != sil.shape or pb.shape[:2] != sil.shape:
        raise ValueError("proxy inputs must have the same raster size")
    if np.any((labels[sil] < 1) | (labels[sil] > len(ZONES))):
        raise ValueError("unlabelled mannequin vertices/pixels")
    luminance = pb @ np.array([0.299, 0.587, 0.114])
    mean = float(luminance[sil].mean())
    if mean <= np.finfo(float).eps:
        raise ValueError("zero mannequin luminance")
    out = pb.copy()  # outside figure = plate with existing cast/contact shadows
    out[sil] = colors[labels[sil]] * (luminance[sil] / mean)[:, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def strength_map(silhouette, shadow, strength, cfg=None):
    from scipy.ndimage import distance_transform_edt
    cfg = config() if cfg is None else cfg
    sil = np.asarray(silhouette, bool)
    reg = sil | (np.asarray(shadow, float) > cfg["shadow_threshold"])
    if not reg.any():
        raise ValueError("empty region for strength map")
    h = sil.shape[0]
    dist = distance_transform_edt(~reg)
    outside = np.maximum(dist - cfg["strength_dilate_height"] * h, 0)
    sigma = cfg["strength_sigma_height"] * h
    if sigma <= 0 or not 0 < strength <= 1:
        raise ValueError("invalid regional strength / sigma")
    return strength * np.exp(-outside ** 2 / (2 * sigma ** 2))
