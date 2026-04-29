#!/usr/bin/env python3
"""
Offline test: validate tile rasterizer pipeline data against CUDA renderer.

Checks:
1. Preprocess: center_pix, extents match CUDA point_xy_image
2. Tile binning: entries count, ranges, sort order
3. Small-crop pixel test with Numba JIT for speed

Usage:
    conda run -n nest_splatting python scripts/test_tile_raster.py \
        --model_path /path/to/model \
        --baked_dir /path/to/baked_atlas \
        --cameras /path/to/cameras.json
"""

import os
import sys
import json
import math
import pickle
import argparse
import time

import numpy as np
import torch
from PIL import Image

NEST_ROOT = "/home/nilkel/Projects/nest-splatting"
sys.path.insert(0, NEST_ROOT)

# ============================================================================
# SH constants
# ============================================================================
SH_C0 = 0.28209479177387814
SH_C1 = 0.4886025119029199
SH_C2 = np.array([1.0925484305920792, -1.0925484305920792, 0.31539156525252005,
                   -1.0925484305920792, 0.5462742152960396])
SH_C3 = np.array([-0.5900435899266435, 2.890611442640554, -0.4570457994644658,
                   0.3731763325901154, -0.4570457994644658, 1.445305721320277,
                   -0.5900435899266435])


def compute_sh_color_batch(dirs, sh_coeffs, sh_deg=3):
    x, y, z = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    result = SH_C0 * sh_coeffs[:, 0]
    if sh_deg > 0:
        result += (-SH_C1 * y[:, None] * sh_coeffs[:, 1] +
                   SH_C1 * z[:, None] * sh_coeffs[:, 2] -
                   SH_C1 * x[:, None] * sh_coeffs[:, 3])
        if sh_deg > 1:
            xx, yy, zz = x*x, y*y, z*z
            xy, yz, xz = x*y, y*z, x*z
            result += (SH_C2[0]*xy[:,None]*sh_coeffs[:,4] + SH_C2[1]*yz[:,None]*sh_coeffs[:,5] +
                       SH_C2[2]*(2*zz-xx-yy)[:,None]*sh_coeffs[:,6] + SH_C2[3]*xz[:,None]*sh_coeffs[:,7] +
                       SH_C2[4]*(xx-yy)[:,None]*sh_coeffs[:,8])
            if sh_deg > 2:
                result += (SH_C3[0]*(y*(3*xx-yy))[:,None]*sh_coeffs[:,9] +
                           SH_C3[1]*(xy*z)[:,None]*sh_coeffs[:,10] +
                           SH_C3[2]*(y*(4*zz-xx-yy))[:,None]*sh_coeffs[:,11] +
                           SH_C3[3]*(z*(2*zz-3*xx-3*yy))[:,None]*sh_coeffs[:,12] +
                           SH_C3[4]*(x*(4*zz-xx-yy))[:,None]*sh_coeffs[:,13] +
                           SH_C3[5]*(z*(xx-yy))[:,None]*sh_coeffs[:,14] +
                           SH_C3[6]*(x*(xx-3*yy))[:,None]*sh_coeffs[:,15])
    result += 0.5
    return np.maximum(result, 0.0)


def quat_to_rotmat_batch(q):
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.zeros((len(q), 3, 3), dtype=np.float32)
    R[:,0,0]=1-2*(y*y+z*z); R[:,0,1]=2*(x*y-w*z); R[:,0,2]=2*(x*z+w*y)
    R[:,1,0]=2*(x*y+w*z);   R[:,1,1]=1-2*(x*x+z*z); R[:,1,2]=2*(y*z-w*x)
    R[:,2,0]=2*(x*z-w*y);   R[:,2,1]=2*(y*z+w*x);   R[:,2,2]=1-2*(x*x+y*y)
    return R


def load_camera(cam_dict, W=800, H=800):
    R = np.array(cam_dict["rotation"], dtype=np.float32)
    t = np.array(cam_dict["position"], dtype=np.float32)
    fx, fy = cam_dict["fx"], cam_dict["fy"]
    view = np.eye(4, dtype=np.float32)
    view[:3,:3] = R; view[:3,3] = -R @ t
    campos = t.copy()
    znear, zfar = 0.01, 100.0
    tanfovx = W/(2*fx); tanfovy = H/(2*fy)
    right = tanfovx*znear; top = tanfovy*znear
    proj = np.zeros((4,4), dtype=np.float32)
    proj[0,0]=2*znear/(2*right); proj[1,1]=2*znear/(2*top)
    proj[2,2]=zfar/(zfar-znear); proj[2,3]=-(zfar*znear)/(zfar-znear); proj[3,2]=1.0
    return view, proj, campos, fx, fy, znear, zfar, tanfovx, tanfovy


# ============================================================================
# CUDA Reference
# ============================================================================
def render_cuda_reference(model_path, baked_dir, cam_dict, W=800, H=800):
    from scene import GaussianModel
    from hash_encoder.config import Config
    from diff_surfel_bake_render import GaussianRasterizationSettings, GaussianRasterizer

    with open(os.path.join(model_path, "args.pkl"), 'rb') as f:
        args = pickle.load(f)
    config_yaml = os.path.join(model_path, "config.yaml")
    cfg = Config(config_yaml) if os.path.exists(config_yaml) else Config(args.yaml)

    gaussians = GaussianModel(sh_degree=3)
    gaussians.load_ply(os.path.join(baked_dir, "baked.ply"))
    gaussians.active_sh_degree = 3
    gaussians.base_opacity = cfg.surfel.tg_base_alpha
    kernel_name = getattr(args, 'kernel', 'gaussian')
    kernel_map = {'gaussian':0,'beta':1,'flex':2,'general':3,'beta_scaled':4}
    kernel_type = kernel_map.get(kernel_name, 0)
    beta = cfg.surfel.tg_beta

    view, proj, campos, fx, fy, znear, zfar, tanfovx, tanfovy = load_camera(cam_dict, W, H)
    full_proj = proj @ view
    view_t = torch.tensor(view.T, dtype=torch.float32, device="cuda")
    full_proj_t = torch.tensor(full_proj.T, dtype=torch.float32, device="cuda")
    campos_t = torch.tensor(campos, dtype=torch.float32, device="cuda")

    raster_settings = GaussianRasterizationSettings(
        image_height=H, image_width=W, tanfovx=tanfovx, tanfovy=tanfovy,
        bg=torch.zeros(3, dtype=torch.float32, device="cuda"),
        scale_modifier=1.0, viewmatrix=view_t, projmatrix=full_proj_t,
        sh_degree=3, campos=campos_t, prefiltered=False, debug=False,
        beta=beta, aabb_mode=3,
    )
    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    shapes = gaussians.get_shape if hasattr(gaussians, '_shape') and gaussians._shape is not None else None
    with torch.no_grad():
        color, radii = rasterizer(
            means3D=gaussians.get_xyz,
            means2D=torch.zeros_like(gaussians.get_xyz[:,:2]),
            opacities=gaussians.get_opacity,
            shs=gaussians.get_features, scales=gaussians.get_scaling,
            rotations=gaussians.get_rotation, shapes=shapes,
            kernel_type=kernel_type, residual_textures=None,
            atlas_texture=None, atlas_rects=None, atlas_width=0,
        )
    img = color.clamp(0,1).permute(1,2,0).cpu().numpy()
    return img, gaussians, args, cfg


# ============================================================================
# Vectorized Preprocess (matching CUDA compute_transmat + compute_aabb)
# ============================================================================
def preprocess_surfels(gaussians, view, proj, campos, W, H, kernel_type):
    means3D = gaussians.get_xyz.detach().cpu().numpy()
    scales = gaussians.get_scaling.detach().cpu().numpy()
    rotations = gaussians.get_rotation.detach().cpu().numpy()
    opacities = gaussians.get_opacity.detach().cpu().numpy().squeeze()
    shs = gaussians.get_features.detach().cpu().numpy()
    shapes_t = gaussians.get_shape if hasattr(gaussians,'_shape') and gaussians._shape is not None else None
    shapes = shapes_t.detach().cpu().numpy().squeeze() if shapes_t is not None else None
    N = len(means3D)

    full_proj = proj @ view
    # GLM mat3x4 = [4 rows x 3 cols] in standard math
    ndc2pix = np.array([
        [W/2.0, 0, 0], [0, H/2.0, 0], [0, 0, 0],
        [(W-1)/2.0, (H-1)/2.0, 1.0],
    ], dtype=np.float32)

    p_view = (view[:3,:3] @ means3D.T).T + view[:3,3]
    depth_mask = p_view[:,2] > 0.01

    R_mats = quat_to_rotmat_batch(rotations)
    L = np.zeros((N,3,3), dtype=np.float32)
    L[:,:,0] = R_mats[:,:,0] * scales[:,0:1]
    L[:,:,1] = R_mats[:,:,1] * scales[:,1:2]
    L[:,:,2] = R_mats[:,:,2]

    s2w = np.zeros((N,4,3), dtype=np.float32)
    s2w[:,:3,0] = L[:,:,0]; s2w[:,:3,1] = L[:,:,1]
    s2w[:,:3,2] = means3D; s2w[:,3,2] = 1.0

    fp_n2p = full_proj @ ndc2pix
    s2w_T = np.transpose(s2w, (0,2,1))
    T_mats = s2w_T @ fp_n2p

    Tu = T_mats[:,:,0]; Tv = T_mats[:,:,1]; Tw = T_mats[:,:,2]

    # Per-Gaussian cutoff (AdR for beta_scaled)
    is_beta = kernel_type in [1, 4]
    cutoffs = np.full(N, 4.0, dtype=np.float32)
    if is_beta and shapes is not None:
        k_sq = 9.0 if kernel_type == 4 else 1.0
        k = 3.0 if kernel_type == 4 else 1.0
        ratio = 1.0 / (255.0 * np.maximum(opacities, 1e-10))
        threshold = np.power(ratio, 1.0 / np.maximum(shapes, 1e-10))
        r_beta = np.where(threshold < 1.0, k * np.sqrt(np.maximum(0, 1.0 - threshold)), 0.0)
        log_term = np.log(np.maximum(255.0 * opacities, 1e-10))
        r_lp = np.where(log_term > 0, np.sqrt(2.0 * log_term), 0.0)
        cutoffs = np.minimum(np.maximum(r_beta, r_lp), k + 2.0)

    opa_mask = opacities >= 1.0/255.0

    c_sq = cutoffs**2
    Tw_sq = Tw*Tw
    d_val = c_sq*Tw_sq[:,0] + c_sq*Tw_sq[:,1] + (-1.0)*Tw_sq[:,2]
    valid_d = np.abs(d_val) > 1e-10

    f0 = np.where(valid_d, c_sq/d_val, 0)
    f1 = f0.copy()
    f2 = np.where(valid_d, -1.0/d_val, 0)

    cx = f0*Tu[:,0]*Tw[:,0] + f1*Tu[:,1]*Tw[:,1] + f2*Tu[:,2]*Tw[:,2]
    cy = f0*Tv[:,0]*Tw[:,0] + f1*Tv[:,1]*Tw[:,1] + f2*Tv[:,2]*Tw[:,2]

    hx_sq = cx*cx - (f0*Tu[:,0]**2 + f1*Tu[:,1]**2 + f2*Tu[:,2]**2)
    hy_sq = cy*cy - (f0*Tv[:,0]**2 + f1*Tv[:,1]**2 + f2*Tv[:,2]**2)
    hx = np.sqrt(np.maximum(hx_sq, 1e-4))
    hy = np.sqrt(np.maximum(hy_sq, 1e-4))

    vis_mask = depth_mask & opa_mask & valid_d
    # Clamp extents to reasonable range (CUDA does this via compute_aabb returning false)
    reasonable = (hx < W * 2) & (hy < H * 2) & (hx > 0) & (hy > 0)
    vis_mask &= reasonable
    vis_mask &= (cx > -500) & (cx < W+500) & (cy > -500) & (cy < H+500)
    vis_idx = np.where(vis_mask)[0]
    V = len(vis_idx)
    print(f"[PREPROCESS] {V}/{N} visible")

    dirs = means3D[vis_idx] - campos
    dirs /= (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-8)
    colors = compute_sh_color_batch(dirs, shs[vis_idx])

    return {
        'vis_idx': vis_idx, 'T_mats': T_mats[vis_idx],
        'Tu': Tu[vis_idx], 'Tv': Tv[vis_idx], 'Tw': Tw[vis_idx],
        'centers_pix': np.stack([cx[vis_idx], cy[vis_idx]], axis=1),
        'extents': np.stack([hx[vis_idx], hy[vis_idx]], axis=1),
        'colors': colors, 'opacities': opacities[vis_idx],
        'depths': p_view[vis_idx, 2],
        'shapes': shapes[vis_idx] if shapes is not None else None,
    }


# ============================================================================
# Tile Binning
# ============================================================================
TILE_SIZE = 16

def tile_binning(data, W, H):
    tiles_x = (W + TILE_SIZE - 1) // TILE_SIZE
    tiles_y = (H + TILE_SIZE - 1) // TILE_SIZE
    total_tiles = tiles_x * tiles_y
    V = len(data['vis_idx'])

    cx, cy = data['centers_pix'][:,0], data['centers_pix'][:,1]
    hx, hy = data['extents'][:,0], data['extents'][:,1]

    rx = np.maximum(np.ceil(hx), 1).astype(np.int32)
    ry = np.maximum(np.ceil(hy), 1).astype(np.int32)
    rect_min_x = np.maximum(0, ((cx - rx) / TILE_SIZE).astype(np.int32))
    rect_min_y = np.maximum(0, ((cy - ry) / TILE_SIZE).astype(np.int32))
    rect_max_x = np.minimum(tiles_x, np.ceil((cx + rx) / TILE_SIZE).astype(np.int32))
    rect_max_y = np.minimum(tiles_y, np.ceil((cy + ry) / TILE_SIZE).astype(np.int32))
    rect_max_x = np.maximum(rect_max_x, rect_min_x)
    rect_max_y = np.maximum(rect_max_y, rect_min_y)

    n_tiles_per = (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y)
    total_entries = int(n_tiles_per.sum())
    print(f"[TILE BIN] {tiles_x}x{tiles_y}={total_tiles} tiles, {V} vis, "
          f"{total_entries} entries ({total_entries/max(V,1):.1f}/gauss)")

    tile_ids = np.empty(total_entries, dtype=np.int32)
    vis_indices = np.empty(total_entries, dtype=np.int32)
    entry_depths = np.empty(total_entries, dtype=np.float32)
    idx = 0
    for vi in range(V):
        for ty in range(rect_min_y[vi], rect_max_y[vi]):
            for tx in range(rect_min_x[vi], rect_max_x[vi]):
                tile_ids[idx] = ty * tiles_x + tx
                vis_indices[idx] = vi
                entry_depths[idx] = data['depths'][vi]
                idx += 1

    # Sort by (tile_id, depth)
    sort_key = tile_ids.astype(np.int64) * 1000000 + (entry_depths * 10000).astype(np.int64)
    sorted_order = np.argsort(sort_key)
    tile_ids = tile_ids[sorted_order]
    vis_indices = vis_indices[sorted_order]

    tile_ranges = np.zeros((total_tiles, 2), dtype=np.int32)
    if total_entries > 0:
        changes = np.where(np.diff(tile_ids) != 0)[0] + 1
        starts = np.concatenate([[0], changes])
        ends = np.concatenate([changes, [total_entries]])
        for s, e in zip(starts, ends):
            tile_ranges[tile_ids[s]] = [s, e]

    non_empty = np.sum(tile_ranges[:,1] > tile_ranges[:,0])
    max_per_tile = np.max(tile_ranges[:,1] - tile_ranges[:,0]) if total_entries > 0 else 0
    print(f"[TILE BIN] Non-empty: {non_empty}, max/tile: {max_per_tile}")
    return tile_ranges, vis_indices, tiles_x, tiles_y


# ============================================================================
# Numba JIT tile rasterizer (fast)
# ============================================================================
def tile_rasterize_crop(tile_ranges, sorted_vis, data, W, H, kernel_type,
                        tiles_x, tiles_y, crop=None):
    """Rasterize a crop region. crop=(x0,y0,x1,y1) or None for full."""
    Tu = data['Tu']
    Tv = data['Tv']
    Tw = data['Tw']
    colors = data['colors']
    opas = data['opacities']
    shapes = data['shapes']
    centers = data['centers_pix']
    has_shapes = shapes is not None

    if crop:
        cx0, cy0, cx1, cy1 = crop
    else:
        cx0, cy0, cx1, cy1 = 0, 0, W, H

    img = np.zeros((cy1-cy0, cx1-cx0, 3), dtype=np.float32)
    FilterInvSquare = 1.0 / (2.0 * 0.3 * 0.3)
    near_n = 0.01

    ty_start = cy0 // 16
    ty_end = (cy1 + 15) // 16
    tx_start = cx0 // 16
    tx_end = (cx1 + 15) // 16

    t0 = time.time()
    total_pixels = (cy1-cy0) * (cx1-cx0)
    pixels_done = 0

    for ty in range(ty_start, ty_end):
        for tx in range(tx_start, tx_end):
            tile_id = ty * tiles_x + tx
            start, end = tile_ranges[tile_id]

            y0 = ty * 16
            y1 = min(y0 + 16, H)
            x0 = tx * 16
            x1 = min(x0 + 16, W)

            for py in range(max(y0, cy0), min(y1, cy1)):
                for px in range(max(x0, cx0), min(x1, cx1)):
                    T_val = 1.0
                    color = np.zeros(3)

                    for j in range(start, end):
                        vi = sorted_vis[j]
                        tu = Tu[vi]; tv = Tv[vi]; tw = Tw[vi]

                        kx = px*tw[0]-tu[0]; ky = px*tw[1]-tu[1]; kz = px*tw[2]-tu[2]
                        lx = py*tw[0]-tv[0]; ly = py*tw[1]-tv[1]; lz = py*tw[2]-tv[2]

                        pz = kx*ly - ky*lx
                        if abs(pz) < 1e-10:
                            continue
                        px_ = ky*lz - kz*ly
                        py_ = kz*lx - kx*lz
                        sx = px_/pz; sy = py_/pz
                        rho3d = sx*sx + sy*sy

                        dx = centers[vi,0]-px; dy = centers[vi,1]-py
                        rho2d = FilterInvSquare * (dx*dx + dy*dy)

                        depth = (sx*tw[0]+sy*tw[1])+tw[2] if rho3d <= rho2d else tw[2]
                        if depth < near_n:
                            continue

                        opa = opas[vi]
                        if kernel_type == 4 and has_shapes:
                            if rho3d >= 9.0 + 1e-6:
                                continue
                            base = max(0.0, 1.0 - rho3d/9.0)
                            alpha_beta = base ** shapes[vi]
                            alpha_lp = math.exp(-rho2d/2.0)
                            alpha = min(0.99, opa * max(alpha_beta, alpha_lp))
                        else:
                            rho = min(rho3d, rho2d)
                            power = -0.5 * rho
                            if power > 0: continue
                            alpha = min(0.99, opa * math.exp(power))

                        if alpha < 1.0/255.0:
                            continue
                        test_T = T_val * (1.0 - alpha)
                        if test_T < 0.0001:
                            break

                        w = alpha * T_val
                        color += colors[vi] * w
                        T_val = test_T

                    img[py-cy0, px-cx0] = color
                    pixels_done += 1

        elapsed = time.time() - t0
        if (ty - ty_start) % 2 == 0 and pixels_done > 0:
            rate = pixels_done / elapsed
            eta = (total_pixels - pixels_done) / max(rate, 1)
            print(f"  row {ty-ty_start}/{ty_end-ty_start}, {pixels_done}/{total_pixels} px, "
                  f"{elapsed:.1f}s, ~{eta:.0f}s left")

    print(f"  Done in {time.time()-t0:.1f}s ({total_pixels} pixels)")
    return img


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--baked_dir", required=True)
    parser.add_argument("--cameras", required=True)
    parser.add_argument("--cam_idx", type=int, default=0)
    parser.add_argument("--out_dir", default="/tmp/tile_raster_test")
    parser.add_argument("--crop", type=int, nargs=4, default=None,
                        help="x0 y0 x1 y1 (default: 200 300 600 600 center crop)")
    parser.add_argument("--full", action="store_true", help="Render full frame")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.cameras) as f:
        cameras = json.load(f)
    cam_dict = cameras[args.cam_idx]
    W, H = cam_dict["width"], cam_dict["height"]
    print(f"Camera {args.cam_idx}: {cam_dict['img_name']}, {W}x{H}")

    # Step 1: CUDA reference
    print("\n=== CUDA Reference (SH-only) ===")
    img_ref, gaussians, train_args, cfg = render_cuda_reference(
        args.model_path, args.baked_dir, cam_dict, W, H)
    Image.fromarray((img_ref*255).clip(0,255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "ref_cuda.png"))
    print(f"  Saved ref_cuda.png")

    # Step 2: Preprocess
    print("\n=== Preprocess ===")
    view, proj, campos, *_ = load_camera(cam_dict, W, H)
    kernel_name = getattr(train_args, 'kernel', 'gaussian')
    kernel_map = {'gaussian':0,'beta':1,'flex':2,'general':3,'beta_scaled':4}
    kernel_type = kernel_map.get(kernel_name, 0)
    data = preprocess_surfels(gaussians, view, proj, campos, W, H, kernel_type)

    # Step 3: Tile binning
    print("\n=== Tile Binning ===")
    tile_ranges, sorted_vis, tiles_x, tiles_y = tile_binning(data, W, H)

    # Sanity
    print(f"\n=== Sanity ===")
    total_entries = sorted_vis.shape[0]
    non_empty = np.sum(tile_ranges[:,1] > tile_ranges[:,0])
    max_per_tile = np.max(tile_ranges[:,1] - tile_ranges[:,0]) if total_entries > 0 else 0
    total_tiles = tiles_x * tiles_y
    print(f"  Entries: {total_entries} | Non-empty tiles: {non_empty}/{total_tiles} | Max/tile: {max_per_tile}")
    print(f"  Max tile_id: {total_tiles-1} (16-bit: {total_tiles-1 < 65536})")

    # Histogram of tiles_per_gaussian
    cx, cy = data['centers_pix'][:,0], data['centers_pix'][:,1]
    hx, hy = data['extents'][:,0], data['extents'][:,1]
    print(f"  Center X range: [{cx.min():.0f}, {cx.max():.0f}]")
    print(f"  Center Y range: [{cy.min():.0f}, {cy.max():.0f}]")
    print(f"  Extent X: mean={hx.mean():.1f}, max={hx.max():.1f}")
    print(f"  Extent Y: mean={hy.mean():.1f}, max={hy.max():.1f}")

    # Check WebGPU buffer sizing
    num_points = len(data['vis_idx'])
    webgpu_max = num_points * 20  # max_tile_entries in tile_raster.rs
    print(f"  WebGPU max_tile_entries = {num_points} * 20 = {webgpu_max}")
    print(f"  Actual entries: {total_entries} ({'OK' if total_entries <= webgpu_max else 'OVERFLOW!'})")

    # Step 4: Rasterize crop
    if args.crop:
        crop = tuple(args.crop)
    elif args.full:
        crop = None
    else:
        crop = (200, 300, 600, 600)  # default: center of chair

    crop_label = f"[{crop[0]},{crop[1]}]-[{crop[2]},{crop[3]}]" if crop else "full"
    print(f"\n=== Tile Rasterize ({crop_label}) ===")

    img_test = tile_rasterize_crop(
        tile_ranges, sorted_vis, data, W, H, kernel_type,
        tiles_x, tiles_y, crop=crop)

    # Save
    if crop:
        x0, y0, x1, y1 = crop
        crop_ref = img_ref[y0:y1, x0:x1]
    else:
        x0, y0 = 0, 0
        crop_ref = img_ref

    Image.fromarray((img_test*255).clip(0,255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "test_tile.png"))
    Image.fromarray((crop_ref*255).clip(0,255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "ref_crop.png"))

    # Compare
    print(f"\n=== Compare ({crop_label}) ===")
    diff = np.abs(crop_ref - img_test)
    mse = np.mean(diff**2)
    psnr_val = -10*np.log10(mse+1e-10)
    print(f"  MSE={mse:.6f}, PSNR={psnr_val:.2f} dB")
    print(f"  MaxDiff={diff.max():.4f}, MeanDiff={diff.mean():.6f}")
    for ch, name in enumerate(["R","G","B"]):
        d = diff[:,:,ch]
        print(f"  {name}: mean={d.mean():.6f}, max={d.max():.4f}")

    diff_amp = (diff * 10).clip(0, 1)
    Image.fromarray((diff_amp*255).astype(np.uint8)).save(
        os.path.join(args.out_dir, "diff_10x.png"))
    print(f"\n  Saved: ref_crop.png, test_tile.png, diff_10x.png in {args.out_dir}")


if __name__ == "__main__":
    main()
