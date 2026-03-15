// Tile-based compute rasterizer with shared memory batching and early termination.
// Each workgroup (16x16 = 256 threads) processes one tile of pixels.
// Matches the CUDA tile-based rasterizer approach.

const CUTOFF: f32 = 2.3539888583335364; // sqrt(log(255)), matches gaussian.wgsl
const CUTOFF_SQ: f32 = 2.0 * 2.3539888583335364; // = 2*CUTOFF, discard threshold for dot(pos,pos)
const BLOCK_SIZE: u32 = 256u;
const T_THRESHOLD: f32 = 0.001;

struct Splat {
    v_0: u32, v_1: u32,
    pos: u32,
    color_0: u32, color_1: u32,
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
var<storage, read> splats: array<Splat>;

@group(0) @binding(2)
var<storage, read> tile_payloads: array<u32>;    // sorted splat indices per tile

@group(0) @binding(3)
var<storage, read> tile_ranges: array<vec2<u32>>; // (start, end) per tile

@group(0) @binding(4)
var<storage, read_write> output_buf: array<u32>;  // W*H packed RGBA8

@group(0) @binding(5)
var<uniform> tile_info: TileRasterInfo;

// Shared memory for batch loading
var<workgroup> sh_splats: array<Splat, 256>;

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

    // Bounds check (for partial tiles at edges)
    let inside = pix.x < vp_w && pix.y < vp_h;

    // Fragment NDC position (texel center)
    let frag_ndc = (vec2<f32>(pix) + 0.5) * 2.0 / vec2<f32>(f32(vp_w), f32(vp_h)) - 1.0;

    // Get tile range
    let tile_id = wg_id.y * tile_info.tiles_x + wg_id.x;
    let range = tile_ranges[tile_id];
    let num_gaussians = range.y - range.x;

    let thread_idx = local_id.y * 16u + local_id.x;

    // Accumulation state
    var color = vec3<f32>(0.0);
    var T: f32 = 1.0;  // transmittance
    var done = false;

    // Process in batches of BLOCK_SIZE
    let rounds = (num_gaussians + BLOCK_SIZE - 1u) / BLOCK_SIZE;

    for (var round = 0u; round < rounds; round++) {
        // Batch load: each thread loads one Gaussian into shared memory
        workgroupBarrier();
        let load_idx = range.x + round * BLOCK_SIZE + thread_idx;
        if load_idx < range.y {
            sh_splats[thread_idx] = splats[tile_payloads[load_idx]];
        }
        workgroupBarrier();

        // Skip if this pixel is done
        if done || !inside {
            continue;
        }

        // Process batch
        let batch_count = min(BLOCK_SIZE, num_gaussians - round * BLOCK_SIZE);
        for (var j = 0u; j < batch_count; j++) {
            if T < T_THRESHOLD {
                done = true;
                break;
            }

            let splat = sh_splats[j];

            // Unpack splat data
            let v1 = unpack2x16float(splat.v_0);  // eigenvector 1 / viewport
            let v2 = unpack2x16float(splat.v_1);  // eigenvector 2 / viewport
            let center = unpack2x16float(splat.pos);
            let rg = unpack2x16float(splat.color_0);
            let ba = unpack2x16float(splat.color_1);

            // Offset from splat center in NDC
            let d = frag_ndc - center;

            // Invert the 2x2 eigenvector matrix to get eigenvector-basis position
            // Vertex shader: offset_ndc = 2 * mat2x2(v1, v2) * position
            // Inverse: position = inv(2 * M) * d
            let det2 = 2.0 * (v1.x * v2.y - v2.x * v1.y);
            if abs(det2) < 1e-10 {
                continue;
            }
            let inv_det = 1.0 / det2;
            let screen_pos = vec2<f32>(
                (v2.y * d.x - v2.x * d.y) * inv_det,
                (-v1.y * d.x + v1.x * d.y) * inv_det,
            );

            // Squared distance in eigenvector space
            let a = dot(screen_pos, screen_pos);
            if a > CUTOFF_SQ {
                continue;
            }

            // Alpha computation (matches gaussian.wgsl fragment shader)
            let alpha = min(0.99, exp(-a) * ba.y);
            if alpha < 1.0 / 255.0 {
                continue;
            }

            // Front-to-back compositing
            let w = alpha * T;
            color += vec3<f32>(rg.x, rg.y, ba.x) * w;
            T *= (1.0 - alpha);
        }
    }

    if !inside {
        return;
    }

    // Add background color
    color += T * vec3<f32>(tile_info.bg_r, tile_info.bg_g, tile_info.bg_b);

    // Pack RGBA8 and write to output buffer
    let r = u32(clamp(color.x, 0.0, 1.0) * 255.0);
    let g = u32(clamp(color.y, 0.0, 1.0) * 255.0);
    let b = u32(clamp(color.z, 0.0, 1.0) * 255.0);
    let pixel_idx = pix.y * vp_w + pix.x;
    output_buf[pixel_idx] = r | (g << 8u) | (b << 16u) | (255u << 24u);
}
