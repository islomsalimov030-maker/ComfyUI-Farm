"""Deterministic source-material preservation for rigid props after H3.

The repair region is restricted to the actual composite support and excludes
the body's composite support. The original depth-occluded RGBA provides edges;
the plate restores unwanted H3 halos around an isolated rigid prop.
"""
import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, distance_transform_edt


def alpha(strength):
    return gaussian_filter(binary_dilation(np.asarray(strength)>0,iterations=16).astype(np.float32),
                           sigma=8.,radius=8,mode='mirror')


def support(strength):
    return alpha(strength)>0


def outside_metrics(final, plate, composite_support, cfg):
    """Actual PNG RGB spatial SSIM, same formula as pose_h3_v26.ssim_masked."""
    a=np.clip(final.detach().cpu().numpy()[0]*255,0,255).astype('uint8').astype(float)
    b=np.clip(plate.detach().cpu().numpy()[0]*255,0,255).astype('uint8').astype(float)
    outside=~np.asarray(composite_support,dtype=bool)
    changed=int(np.any(a!=b,axis=-1)[outside].sum())
    scores=[]
    if outside.any():
        sigma=cfg['h3']['ssim_sigma']
        c1=(cfg['h3']['ssim_k1']*255)**2
        c2=(cfg['h3']['ssim_k2']*255)**2
        for channel in range(3):
            x,y=a[...,channel],b[...,channel]
            mx=gaussian_filter(x,sigma)
            my=gaussian_filter(y,sigma)
            vx=gaussian_filter(x*x,sigma)-mx*mx
            vy=gaussian_filter(y*y,sigma)-my*my
            cov=gaussian_filter(x*y,sigma)-mx*my
            value=((2*mx*my+c1)*(2*cov+c2))/((mx*mx+my*my+c1)*(vx+vy+c2))
            scores.append(float(value[outside].mean()))
    score=float(np.mean(scores)) if scores else None
    limits=cfg['native_composite']
    return dict(outside_changed_pixels=changed,outside_ssim=score,outside_pixels=int(outside.sum()),
                support_pixels=int(composite_support.sum()),outside_passed=bool(score is not None and
                changed<=limits['outside_changed_pixels_max'] and score>=limits['outside_ssim_min']))


def repair_mask(body_strength, total_strength, body_silhouette=None, prop_silhouette=None):
    if body_silhouette is not None and prop_silhouette is not None:
        body=np.asarray(body_silhouette,dtype=bool)
        prop=np.asarray(prop_silhouette,dtype=bool)
        if not prop.any():
            return np.zeros_like(prop)
        # Noise-map tails overlap even when the actual objects do not. Excluding
        # the whole body noise support would cut a horizontal seam through props
        # near the head. Assign shared background to the nearest real silhouette.
        nearest_prop=distance_transform_edt(~prop)<distance_transform_edt(~body)
        return support(total_strength)&nearest_prop
    return support(total_strength) & ~support(body_strength)


def restore(final, image, mask):
    if tuple(image.shape)!=tuple(final.shape) or tuple(mask.shape)!=tuple(final.shape[:3]):
        raise ValueError('Prop repair image/mask must match final raster')
    import torch
    if not torch.isfinite(image).all() or not torch.isfinite(mask).all() or mask.min()<0 or mask.max()>1:
        raise ValueError('Invalid prop repair image/mask')
    alpha=mask[...,None]
    return final*(1-alpha)+image*alpha


def restore_isolated(final, prop_rgb, prop_alpha, repair, repair_mask, body_strength):
    """Restore the room around the prop without overwriting generated body.

    The background guard is an actual compositing operation, recorded as part
    of its support. Keep body support intact, even where noise tails overlap.
    """
    import torch
    body_support=support(body_strength.detach().cpu().numpy()[0])
    guard=(repair_mask.detach().cpu().numpy()[0]>0)&~body_support
    guard_tensor=torch.from_numpy(guard.astype(np.float32))[None].to(final.device)
    final=restore(final,repair,guard_tensor)
    final=restore(final,prop_rgb,prop_alpha)
    final_support=body_support|guard|(prop_alpha.detach().cpu().numpy()[0]>0)
    return final,final_support,guard


def harmonize(prop_rgb, h3_rgb, prop_alpha, sigma=6., tau=.12):
    """0.3.0 «C»: after H3 only the FINE texture of the rigid prop is restored; colour and light come from H3.

    Inside the prop alpha: low frequencies (masked blur, so the room behind never leaks in) are taken from H3,
    high frequencies from the rendered prop. Where H3 moved/distorted the prop (its low-frequency colour differs
    from the lit render by more than tau) the rendered prop is kept — the 3D shape is never redrawn.
    Arrays HxWx3 / HxW in [0,1]; returns HxWx3."""
    p = np.asarray(prop_rgb, dtype=np.float32)
    h = np.asarray(h3_rgb, dtype=np.float32)
    a = np.clip(np.asarray(prop_alpha, dtype=np.float32), 0, 1)
    inside = (a > .5).astype(np.float32)
    weight = gaussian_filter(inside, sigma) + 1e-6

    def low(x):
        return np.stack([gaussian_filter(x[..., c] * inside, sigma) for c in range(3)], -1) / weight[..., None]
    lp, lh = low(p), low(h)
    detail = p - lp
    agree = (np.abs(lh - lp).max(axis=-1) < tau)[..., None]
    out = np.where(agree, lh + detail, p)
    return np.clip(np.where(inside[..., None] > 0, out, p), 0, 1)
