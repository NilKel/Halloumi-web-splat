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
    depth_u: f32,
    depth_v: f32,
    depth_center: f32,
    _pad: u32,
};

// Compact shared memory struct (56 bytes)
struct TileSplat {
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opacity: f32,
    pos: u32,
    color_rg: u32,
    color_b_shape: u32,
    gauss_id: u32,
    depth_u: f32,
    depth_v: f32,
    depth_center: f32,
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

@group(0) @binding(7)
var<storage, read> atlas_texture: array<u32>;

@group(0) @binding(8)
var<storage, read> atlas_rects: array<f32>;

// Mirrors renderer.rs::TexParamsUniform (32 bytes). Must match byte-for-byte.
struct TexParams {
    atlas_width: u32,
    atlas_layer_h: u32,
    kernel_type: u32,
    uv_extent_bits: u32,
    atlas_format: u32,     // 0 = FP16 RGB, 1 = UINT8 RGBA (dequant via scale/offset)
    atlas_scale: f32,      // UINT8 dequant: residual = u8_norm * scale + offset
    atlas_offset: f32,
    res_bias: f32,         // additive bias before final ReLU (CUDA d_res_bias)
};

const ATLAS_FORMAT_FP16_RGB: u32 = 0u;
const ATLAS_FORMAT_UINT8_RGBA: u32 = 1u;
const ATLAS_FORMAT_BC7: u32 = 2u;

@group(0) @binding(9)
var<uniform> tex_params: TexParams;

// BC7 atlas as a sampled texture_2d_array. atlas_format == 2 routes through
// these via textureSampleLevel (HW BC7 decoder). Other formats use the
// storage-buffer atlas_texture above. Same binding shape as HW path's
// gaussian_2dgs.wgsl @group(3) @binding(0) / @binding(2).
@group(0) @binding(10)
var atlas_bc7: texture_2d_array<f32>;
@group(0) @binding(11)
var atlas_samp: sampler;

// Shared memory — only allocated when USE_SHARED_MEM == 1
// When USE_SHARED_MEM == 0, this is still declared but never written/read,
// so the compiler should optimize it away.
var<workgroup> sh_splats: array<TileSplat, BLOCK_SIZE>;

// Read a single FP16 value from the legacy FP16-RGB atlas (6 bytes/texel, 3 chans).
fn read_atlas_f16(row: i32, col: i32, ch: u32, atlas_w: i32) -> f32 {
    let f16_index = u32(row * atlas_w + col) * 3u + ch;
    let u32_index = f16_index / 2u;
    let component = f16_index % 2u;
    return unpack2x16float(atlas_texture[u32_index])[component];
}

// Read a UINT8 RGBA texel and dequantize to residual RGB (A channel ignored).
// Matches CUDA's cudaReadModeNormalizedFloat path:
//   residual = u8_norm * atlas_scale + atlas_offset.
fn read_atlas_uint8(row: i32, col: i32, atlas_w: i32) -> vec3<f32> {
    let word = atlas_texture[u32(row * atlas_w + col)];
    let norm = unpack4x8unorm(word);  // vec4 in [0, 1]
    let s = tex_params.atlas_scale;
    let o = tex_params.atlas_offset;
    return vec3<f32>(norm.x * s + o, norm.y * s + o, norm.z * s + o);
}

// Sample atlas with bilinear interpolation at surfel UV.
//
// BC7 path (atlas_format == 2): use HW textureSampleLevel on the
// texture_2d_array. atlas_rects layout is 5-stride (u0, v0_local, w_span,
// h_span, layer) — matches HW gaussian_2dgs.wgsl exactly.
//
// Legacy UINT8/FP16 paths: hand-rolled bilinear from the storage buffer
// atlas_texture, 4-stride atlas_rects (u0_px, v0_px, u_span, v_span).
fn sample_atlas(surfel_uv: vec2<f32>, gauss_id: u32) -> vec3<f32> {
    let E = bitcast<f32>(tex_params.uv_extent_bits);

    if tex_params.atlas_format == ATLAS_FORMAT_BC7 {
        // BC7 layout matches HW path's gaussian_2dgs.wgsl. atlas_rects
        // stride is 5 floats: u0, v0_local, w_span, h_span, layer.
        let r_base = gauss_id * 5u;
        let u0       = atlas_rects[r_base + 0u];
        let v0_local = atlas_rects[r_base + 1u];
        let w_span   = atlas_rects[r_base + 2u];
        let h_span   = atlas_rects[r_base + 3u];
        let layer    = i32(atlas_rects[r_base + 4u]);

        let au = u0       + (surfel_uv.x + E) / (2.0 * E) * w_span;
        let av = v0_local + (surfel_uv.y + E) / (2.0 * E) * h_span;
        let uv = vec2<f32>(
            (au + 0.5) / f32(tex_params.atlas_width),
            (av + 0.5) / f32(tex_params.atlas_layer_h),
        );
        let rgba = textureSampleLevel(atlas_bc7, atlas_samp, uv, layer, 0.0);
        return rgba.rgb * tex_params.atlas_scale + vec3<f32>(tex_params.atlas_offset);
    }

    let r_base = gauss_id * 4u;
    let u0_px  = atlas_rects[r_base + 0u];
    let v0_px  = atlas_rects[r_base + 1u];
    let u_span = atlas_rects[r_base + 2u];
    let v_span = atlas_rects[r_base + 3u];

    // Bake kernel places sample i at s = (i+0.5)*step - E, so invert:
    //   tex_coord = (s+E)/(2E) * G - 0.5
    let au = u0_px + (surfel_uv.x + E) / (2.0 * E) * u_span - 0.5;
    let av = v0_px + (surfel_uv.y + E) / (2.0 * E) * v_span - 0.5;
    let au_c = clamp(au, u0_px, u0_px + u_span - 1.001);
    let av_c = clamp(av, v0_px, v0_px + v_span - 1.001);

    let au0 = i32(au_c);
    let av0 = i32(av_c);
    let fu = au_c - f32(au0);
    let fv = av_c - f32(av0);
    let au1 = min(au0 + 1, i32(u0_px + u_span - 1.0));
    let av1 = min(av0 + 1, i32(v0_px + v_span - 1.0));

    let aw = i32(tex_params.atlas_width);

    if tex_params.atlas_format == ATLAS_FORMAT_UINT8_RGBA {
        let c00 = read_atlas_uint8(av0, au0, aw);
        let c10 = read_atlas_uint8(av0, au1, aw);
        let c01 = read_atlas_uint8(av1, au0, aw);
        let c11 = read_atlas_uint8(av1, au1, aw);
        return (1.0 - fu) * (1.0 - fv) * c00
             + fu         * (1.0 - fv) * c10
             + (1.0 - fu) * fv         * c01
             + fu         * fv         * c11;
    }

    // Legacy FP16 RGB fallback.
    var residual = vec3<f32>(0.0);
    for (var ch = 0u; ch < 3u; ch++) {
        let c00 = read_atlas_f16(av0, au0, ch, aw);
        let c10 = read_atlas_f16(av0, au1, ch, aw);
        let c01 = read_atlas_f16(av1, au0, ch, aw);
        let c11 = read_atlas_f16(av1, au1, ch, aw);
        residual[ch] = (1.0 - fu) * (1.0 - fv) * c00
                     + fu * (1.0 - fv) * c10
                     + (1.0 - fu) * fv * c01
                     + fu * fv * c11;
    }
    return residual;
}

// ---- Helper: evaluate one splat against a pixel ----
fn eval_splat_fields(
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opa: f32, pos_packed: u32, color_rg_packed: u32, color_b_shape_packed: u32,
    gauss_id: u32, depth_u: f32, depth_v: f32, depth_center: f32,
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

    let depth = dot(vec3<f32>(depth_u, depth_v, depth_center), vec3<f32>(s, 1.0));
    if depth < 0.2 {
        return vec4<f32>(0.0, 0.0, 0.0, -1.0);
    }

    let ba = unpack2x16float(color_b_shape_packed);

    var alpha: f32;
    if KERNEL_TYPE == 4u {
        // BetaScaled kernel (k² = 9). Matches CUDA renderBakedCUDA case 4.
        let shape = ba.y;
        let k_sq = 9.0;
        if rho3d >= k_sq + 1e-6 {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let base = max(0.0, 1.0 - rho3d / k_sq);
        let alpha_beta = pow(base, shape);
        let alpha_lp = exp(-rho2d / 2.0);
        alpha = min(0.99, opa * max(alpha_beta, alpha_lp));
    } else if KERNEL_TYPE == 1u {
        // Beta kernel (k² = 1). Matches CUDA case 1.
        let shape = ba.y;
        let k_sq = 1.0;
        if rho3d >= k_sq + 1e-6 {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        let base = max(0.0, 1.0 - rho3d / k_sq);
        let alpha_beta = pow(base, shape);
        let alpha_lp = exp(-rho2d / 2.0);
        alpha = min(0.99, opa * max(alpha_beta, alpha_lp));
    } else {
        // Standard Gaussian. CUDA uses rho = min(rho3d, rho2d), power = -0.5 * rho,
        // which is equivalent to max(exp(-rho3d/2), exp(-rho2d/2)). The old Halloumi
        // path had exp(-rho3d) (no /2) which produced too-sharp falloff.
        let power = -0.5 * min(rho3d, rho2d);
        if power > 0.0 {
            return vec4<f32>(0.0, 0.0, 0.0, -1.0);
        }
        alpha = min(0.99, opa * exp(power));
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
    // color carries clamp(SH + sh_bias, 0) + SB (view-dep lobe contribution),
    // folded together in preprocess. Tile raster adds residual + res_bias and
    // then applies the final CUDA per-Gaussian ReLU before compositing.
    var color = vec3<f32>(rg.x, rg.y, ba.x);

    if tex_params.atlas_width > 0u {
        color += sample_atlas(s, gauss_id);
    }

    // Final activation: max(0, color + res_bias) per channel. Mirrors CUDA:
    //   feat[ch] = fmaxf(0.0, feat[ch] + d_res_bias) done once before accumulation.
    color = max(vec3<f32>(0.0), color + vec3<f32>(tex_params.res_bias));

    return vec4<f32>(color.x * w, color.y * w, color.z * w, test_T);
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
                    src.gauss_id,
                    src.depth_u,
                    src.depth_v,
                    src.depth_center,
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
                    sp.gauss_id, sp.depth_u, sp.depth_v, sp.depth_center,
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
                    src.gauss_id, src.depth_u, src.depth_v, src.depth_center,
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
