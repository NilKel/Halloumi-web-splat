// 2DGS Surfel tile-based compute rasterizer.
// Ray-disk intersection with transmat (Tu, Tv, Tw).
// Kernel type injected at compile time: KERNEL_TYPE constant.
//   0 = Gaussian (exp(-rho3d))
//   4 = BetaScaled (pow(1-rho3d/k², shape))

const BLOCK_SIZE: u32 = 256u;
const T_THRESHOLD: f32 = 0.0001;
const FILTER_INV_SQUARE: f32 = 2.0; // 1 / (2 * FilterSize^2) where FilterSize = sqrt(2)/2

// Splat2DGS layout in storage buffer (64 bytes)
struct Splat2DGS {
    tu_x: f32, tu_y: f32, tu_z: f32,  // Tu (transmat column 0)
    tv_x: f32, tv_y: f32, tv_z: f32,  // Tv (transmat column 1)
    tw_x: f32, tw_y: f32, tw_z: f32,  // Tw (transmat column 2)
    opacity: f32,
    pos: u32,           // pixel-space center x, y (f16 pair)
    extent: u32,        // unused
    color_rg: u32,      // R, G (f16 pair)
    color_b_shape: u32, // B, shape (f16 pair)
    gauss_id: u32,      // original Gaussian index
    _pad: u32,
};

// Compact shared memory struct (52 bytes, well within 16KB for 256 elements = 13312 bytes)
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
var<storage, read> tile_payloads: array<u32>;    // sorted splat indices per tile

@group(0) @binding(3)
var<storage, read> tile_starts: array<u32>;       // start index per tile

@group(0) @binding(6)
var<storage, read> tile_ends: array<u32>;         // end index per tile

@group(0) @binding(4)
var<storage, read_write> output_buf: array<u32>;  // W*H packed RGBA8

@group(0) @binding(5)
var<uniform> tile_info: TileRasterInfo;

// Shared memory: compact struct (13 * 4 = 52 bytes each, 256 * 52 = 13312 bytes < 16384 limit)
var<workgroup> sh_splats: array<TileSplat, 256>;

@compute @workgroup_size(16, 16, 1)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    // Pixel coordinates
    let pix = vec2<u32>(
        wg_id.x * 16u + local_id.x,
        wg_id.y * 16u + local_id.y,
    );

    let vp_w = tile_info.viewport_w;
    let vp_h = tile_info.viewport_h;
    let inside = pix.x < vp_w && pix.y < vp_h;

    // Pixel position for ray-disk intersection
    let pixf = vec2<f32>(f32(pix.x), f32(pix.y));

    // Get tile range
    let tile_id = wg_id.y * tile_info.tiles_x + wg_id.x;
    let range_start = tile_starts[tile_id];
    let range_end = tile_ends[tile_id];
    let num_gaussians = range_end - range_start;

    let thread_idx = local_id.y * 16u + local_id.x;

    // Accumulation state
    var color = vec3<f32>(0.0);
    var T_acc: f32 = 1.0;
    var done = false;

    // Cap rounds to prevent GPU timeout during debugging (256 rounds = 64K gaussians max per tile)
    let rounds = min((num_gaussians + BLOCK_SIZE - 1u) / BLOCK_SIZE, 256u);

    for (var round = 0u; round < rounds; round++) {
        // Batch load: each thread loads one splat into shared memory (compact copy)
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

            let splat = sh_splats[j];

            // Read transmat
            let Tu = vec3<f32>(splat.tu_x, splat.tu_y, splat.tu_z);
            let Tv = vec3<f32>(splat.tv_x, splat.tv_y, splat.tv_z);
            let Tw = vec3<f32>(splat.tw_x, splat.tw_y, splat.tw_z);

            // Ray-disk intersection: k = px*Tw - Tu, l = py*Tw - Tv, p = cross(k,l)
            let k_vec = pixf.x * Tw - Tu;
            let l_vec = pixf.y * Tw - Tv;
            let p_vec = cross(k_vec, l_vec);

            if p_vec.z == 0.0 {
                continue;
            }

            let s = p_vec.xy / p_vec.z;  // surfel UV intersection
            let rho3d = dot(s, s);

            // 2D low-pass filter distance
            let center_pix = unpack2x16float(splat.pos);
            let d_pix = center_pix - pixf;
            let rho2d = FILTER_INV_SQUARE * dot(d_pix, d_pix);

            let opa = splat.opacity;
            let ba = unpack2x16float(splat.color_b_shape);

            // Kernel evaluation — selected at compile time via KERNEL_TYPE constant
            var alpha: f32;

            if KERNEL_TYPE == 4u {
                // BetaScaled kernel: pow(max(0, 1-rho3d/k²), shape), k²=9
                let shape = ba.y;
                let k_sq = 9.0;

                if rho3d >= k_sq + 1e-6 {
                    continue;
                }

                let base = max(0.0, 1.0 - rho3d / k_sq);
                let alpha_beta = pow(base, shape);
                let alpha_lp = exp(-rho2d / 2.0);
                let kernel_val = max(alpha_beta, alpha_lp);
                alpha = min(0.99, opa * kernel_val);
            } else {
                // Gaussian kernel: exp(-rho3d)
                let alpha_3d = exp(-rho3d);
                let alpha_lp = exp(-rho2d / 2.0);
                let kernel_val = max(alpha_3d, alpha_lp);
                alpha = min(0.99, opa * kernel_val);
            }

            if alpha < 1.0 / 255.0 {
                continue;
            }

            let test_T = T_acc * (1.0 - alpha);
            if test_T < T_THRESHOLD {
                done = true;
                continue;
            }

            // Front-to-back compositing
            let w = alpha * T_acc;
            let rg = unpack2x16float(splat.color_rg);
            color += vec3<f32>(rg.x, rg.y, ba.x) * w;
            T_acc = test_T;
        }
    }

    if !inside {
        return;
    }

    // Add background color
    color += T_acc * vec3<f32>(tile_info.bg_r, tile_info.bg_g, tile_info.bg_b);

    // Pack RGBA8 and write to output buffer
    let r = u32(clamp(color.x, 0.0, 1.0) * 255.0);
    let g = u32(clamp(color.y, 0.0, 1.0) * 255.0);
    let b = u32(clamp(color.z, 0.0, 1.0) * 255.0);
    let pixel_idx = pix.y * vp_w + pix.x;
    output_buf[pixel_idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
