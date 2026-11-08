"""3D-A v2.6: nearest-bone vertex zones, label pass decoding, coloured proxy and D strength map.

Anatomical labels follow step 9; the colour palette is defined by body.json keys.
Missing anatomical zones use skin, or #C8A08A when skin is not provided.
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
    if not isinstance(colors, dict):
        raise ValueError("outfit_colors must be a mapping of zone names to #RRGGBB")
    out = {}
    for zone, text in colors.items():
        if not isinstance(text, str) or len(text) != 7 or text[0] != "#":
            raise ValueError(f"invalid outfit colour {zone}: expected #RRGGBB")
        out[zone] = np.array([int(text[k:k + 2], 16) for k in (1, 3, 5)], dtype=float)
    return out


def colour_vertices(labels, colors):
    """Colour every vertex; palette names are exactly body.json keys + fallback.

    No assignment is invented for extra outfit keys without anatomical mapping.
    The returned names make that limitation measurable rather than silently
    claiming all provided zones are present on an arbitrary mesh.
    """
    labels = np.asarray(labels)
    fallback_name = "skin" if "skin" in colors else "fallback"
    fallback = colors.get("skin", np.array([200., 160., 138.]))
    rgb = np.tile(fallback, labels.shape + (1,))
    names = np.full(labels.shape, fallback_name, dtype=object)
    for i, name in enumerate(ZONES, 1):
        if name in colors:
            take = labels == i
            rgb[take], names[take] = colors[name], name
    return rgb, names


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
    luminance = pb @ np.array([0.299, 0.587, 0.114])
    mean = float(luminance[sil].mean())
    if mean <= np.finfo(float).eps:
        raise ValueError("zero mannequin luminance")
    out = pb.copy()  # outside figure = plate with existing cast/contact shadows
    rgb, _ = colour_vertices(labels[sil], colors)
    out[sil] = rgb * (luminance[sil] / mean)[:, None]
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def strength_map(silhouette, strength, cell_px):
    """Silhouette only: 1-cell dilation + compact Gaussian tail over 2 cells.

    Shifted normalised Gaussian reaches exactly zero at the tail boundary.
    No shadow, plate difference, height-fraction or other region source.
    """
    from scipy.ndimage import distance_transform_edt
    sil = np.asarray(silhouette, bool)
    if not sil.any():
        raise ValueError("empty silhouette")
    if cell_px <= 0 or not 0 < strength <= 1:
        raise ValueError("invalid cell size / strength")
    distance = distance_transform_edt(~sil)
    t = np.maximum(distance - cell_px, 0) / (2 * cell_px)
    edge = np.exp(-2.)
    falloff = np.maximum((np.exp(-2 * t * t) - edge) / (1 - edge), 0)
    return np.where(t >= 1, 0, strength * falloff)
