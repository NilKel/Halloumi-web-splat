// 2DGS Surfel tile-based compute rasterizer.
// Ray-disk intersection with transmat (Tu, Tv, Tw).
//
// Compile-time constants (injected by Rust):
//   KERNEL_TYPE: 0 = Gaussian, 4 = BetaScaled
//   USE_SHARED_MEM: 1 = shared memory batching, 0 = direct global reads

// TILE_SIZE injected at compile time
const BLOCK_SIZE: u32 = TILE_SIZE * TILE_SIZE;
const T_THRESHOLD: f32 = 0.0001;
const FILTER_INV_SQUARE: f32 = 2.0; // 1 / (2 * FilterSize^2) where FilterSize = sqrt(2)/2

// Splat2DGS layout in storage buffer (64 bytes)
struct Splat2DGS {
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opacity: f32,
    pos: u32,
    extent: u32,
    color_rg: u32,
    color_b_shape: u32,
    gauss_id: u32,
    _pad: u32,
};

// Compact shared memory struct (52 bytes)
struct TileSplat {
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opacity: f32,
    pos: u32,
    color_rg: u32,
    color_b_shape: u32,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct TileRasterInfo {
    tiles_x: u32,
    tiles_y: u32,
    viewport_w: u32,
    viewport_h: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    _pad: u32,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(0) @binding(1)
var<storage, read> splats: array<Splat2DGS>;

@group(0) @binding(2)
var<storage, read> tile_payloads: array<u32>;

@group(0) @binding(3)
var<storage, read> tile_starts: array<u32>;

@group(0) @binding(6)
var<storage, read> tile_ends: array<u32>;

@group(0) @binding(4)
var<storage, read_write> output_buf: array<u32>;

@group(0) @binding(5)
var<uniform> tile_info: TileRasterInfo;

// Shared memory — only allocated when USE_SHARED_MEM == 1
// When USE_SHARED_MEM == 0, this is still declared but never written/read,
// so the compiler should optimize it away.
var<workgroup> sh_splats: array<TileSplat, BLOCK_SIZE>;

// ---- Helper: evaluate one splat against a pixel ----
fn eval_splat_fields(
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opa: f32, pos_packed: u32, color_rg_packed: u32, color_b_shape_packed: u32,
    pixf: vec2<f32>,
    T_acc_in: f32,
) -> vec4<f32> {
    // Returns vec4(color_contribution.rgb, new_T) or vec4(0,0,0, -1) to skip

    let Tu = vec3<f32>(tu_x, tu_y, tu_z);
    let Tv = vec3<f32>(tv_x, tv_y, tv_z);
    let Tw = vec3<f32>(tw_x, tw_y, tw_z);

    let k_vec = pixf.x * Tw - Tu;
    let l_vec = pixf.y * Tw - Tv;
    let p_vec = cross(k_vec, l_vec);

    if p_vec.z == 0.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    let s = p_vec.xy / p_vec.z;
    let rho3d = dot(s, s);

    let center_pix = unpack2x16float(pos_packed);
    let d_pix = center_pix - pixf;
    let rho2d = FILTER_INV_SQUARE * dot(d_pix, d_pix);

    let ba = unpack2x16float(color_b_shape_packed);

    var alpha: f32;
    if KERNEL_TYPE == 4u {
        let shape = ba.y;
        let k_sq = 9.0;
        if rho3d >= k_sq + 1e-6 {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let base = max(0.0, 1.0 - rho3d / k_sq);
        let alpha_beta = pow(base, shape);
        let alpha_lp = exp(-rho2d / 2.0);
        alpha = min(0.99, opa * max(alpha_beta, alpha_lp));
    } else {
        let alpha_3d = exp(-rho3d);
        let alpha_lp = exp(-rho2d / 2.0);
        alpha = min(0.99, opa * max(alpha_3d, alpha_lp));
    }

    if alpha < 1.0 / 255.0 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    let test_T = T_acc_in * (1.0 - alpha);
    if test_T < T_THRESHOLD {
        return vec4<f32>(0.0, 0.0, 0.0, -2.0); // signal: done
    }

    let w = alpha * T_acc_in;
    let rg = unpack2x16float(color_rg_packed);
    return vec4<f32>(rg.x * w, rg.y * w, ba.x * w, test_T);
}

@compute @workgroup_size(TILE_SIZE, TILE_SIZE, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let pix = vec2<u32>(
        wg_id.x * TILE_SIZE + local_id.x,
        wg_id.y * TILE_SIZE + local_id.y,
    );

    let vp_w = tile_info.viewport_w;
    let vp_h = tile_info.viewport_h;
    let inside = pix.x < vp_w && pix.y < vp_h;

    let pixf = vec2<f32>(f32(pix.x), f32(pix.y));

    let tile_id = wg_id.y * tile_info.tiles_x + wg_id.x;
    let range_start = tile_starts[tile_id];
    let range_end = tile_ends[tile_id];
    let num_gaussians = range_end - range_start;

    let thread_idx = local_id.y * TILE_SIZE + local_id.x;

    var color = vec3<f32>(0.0);
    var T_acc: f32 = 1.0;
    var done = false;

    if USE_SHARED_MEM == 1u {
        // ===== SHARED MEMORY PATH =====
        let rounds = min((num_gaussians + BLOCK_SIZE - 1u) / BLOCK_SIZE, 256u);

        for (var round = 0u; round < rounds; round++) {
            workgroupBarrier();
            let load_idx = range_start + round * BLOCK_SIZE + thread_idx;
            if load_idx < range_end {
                let src = splats[tile_payloads[load_idx]];
                sh_splats[thread_idx] = TileSplat(
                    src.tu_x, src.tu_y, src.tu_z,
                    src.tv_x, src.tv_y, src.tv_z,
                    src.tw_x, src.tw_y, src.tw_z,
                    src.opacity,
                    src.pos,
                    src.color_rg,
                    src.color_b_shape,
                );
            }
            workgroupBarrier();

            if done || !inside {
                continue;
            }

            let batch_count = min(BLOCK_SIZE, num_gaussians - round * BLOCK_SIZE);
            for (var j = 0u; j < batch_count; j++) {
                if T_acc < T_THRESHOLD {
                    done = true;
                    break;
                }

                let sp = sh_splats[j];
                let result = eval_splat_fields(
                    sp.tu_x, sp.tu_y, sp.tu_z,
                    sp.tv_x, sp.tv_y, sp.tv_z,
                    sp.tw_x, sp.tw_y, sp.tw_z,
                    sp.opacity, sp.pos, sp.color_rg, sp.color_b_shape,
                    pixf, T_acc,
                );

                if result.w == -1.0 {
                    continue;
                }
                if result.w == -2.0 {
                    done = true;
                    continue;
                }
                color += result.xyz;
                T_acc = result.w;
            }
        }
    } else {
        // ===== DIRECT GLOBAL READ PATH (no shared memory, no barriers) =====
        if inside {
            for (var j = 0u; j < num_gaussians; j++) {
                if T_acc < T_THRESHOLD {
                    break;
                }

                let src = splats[tile_payloads[range_start + j]];
                let result = eval_splat_fields(
                    src.tu_x, src.tu_y, src.tu_z,
                    src.tv_x, src.tv_y, src.tv_z,
                    src.tw_x, src.tw_y, src.tw_z,
                    src.opacity, src.pos, src.color_rg, src.color_b_shape,
                    pixf, T_acc,
                );

                if result.w == -1.0 {
                    continue;
                }
                if result.w == -2.0 {
                    break;
                }
                color += result.xyz;
                T_acc = result.w;
            }
        }
    }

    if !inside {
        return;
    }

    color += T_acc * vec3<f32>(tile_info.bg_r, tile_info.bg_g, tile_info.bg_b);

    let r = u32(clamp(color.x, 0.0, 1.0) * 255.0);
    let g = u32(clamp(color.y, 0.0, 1.0) * 255.0);
    let b = u32(clamp(color.z, 0.0, 1.0) * 255.0);
    let pixel_idx = pix.y * vp_w + pix.x;
    output_buf[pixel_idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
