// 2DGS Surfel Render Shader (HW raster, transmat-based).
//
// Vertex: pixel-space quad from precomputed AABB (center_pix, extent_pix) +
// `filter_r` margin (matches CUDA's compute_aabb + max(extent, filter_r)).
// Fragment: ray-disk intersection via Tu/Tv/Tw (CUDA cross-product formula),
// rho3d = dot(s, s), max(alpha_beta, alpha_lp) for BetaScaled / Beta.
//
// Y conventions: center_pix uses CUDA's pixel loop convention, same as
// renderBakedCUDA: pix.y = 0 at the top and increases downward. WebGPU
// framebuffer positions use the same top-left pixel convention, while clip
// NDC has +Y at the top after viewport transform. The vertex shader therefore
// flips pixel Y when writing clip space; the fragment uses builtin position
// directly for ray-disk math.

const FILTER_INV_SQUARE: f32 = 2.0;     // matches CUDA auxiliary.h
const FILTER_SIZE: f32       = 0.7071067811865476;  // sqrt(2)/2
const CUTOFF: f32            = 4.0;     // surfel UV cutoff used in compute_aabb

// Splat2DGS: same layout as preprocess_tile_2dgs.wgsl.
struct Splat2DGS {
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opacity: f32,
    pos: u32,           // CUDA-pixel center (f16 pair)
    extent: u32,        // CUDA-pixel half-extent (f16 pair)
    color_rg: u32,
    color_b_shape: u32, // .x = blue, .y = shape (kernel exponent)
    gauss_id: u32,
    depth_u: f32,
    depth_v: f32,
    depth_center: f32,
    _pad: u32,
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) Tu: vec3<f32>,
    @location(1) @interpolate(flat) Tv: vec3<f32>,
    @location(2) @interpolate(flat) Tw: vec3<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) gauss_id: u32,
    @location(5) @interpolate(flat) shape: f32,
    @location(6) @interpolate(flat) center_pix: vec2<f32>,  // CUDA Y-up
    @location(7) @interpolate(flat) depth_plane: vec3<f32>,
};

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
};

// Mirrors renderer.rs::TexParamsUniform (48 bytes).
struct TexParams {
    atlas_width: u32,
    atlas_layer_h: u32,
    kernel_type: u32,
    uv_extent_bits: u32,
    atlas_format: u32,
    atlas_scale: f32,
    atlas_offset: f32,
    res_bias: f32,
    viewport_w: u32,
    viewport_h: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(2) var<storage, read> splats_2d: array<Splat2DGS>;

@group(1) @binding(0) var<storage, read> sort_infos: SortInfos;
@group(1) @binding(4) var<storage, read> indices: array<u32>;

@group(3) @binding(0) var atlas: texture_2d_array<f32>;
@group(3) @binding(1) var<storage, read> atlas_rects: array<f32>;
@group(3) @binding(2) var atlas_samp: sampler;
@group(3) @binding(3) var<uniform> tex_params: TexParams;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let visible_count = sort_infos.keys_size;
    if in_instance_index >= visible_count {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let splat = splats_2d[indices[in_instance_index]];
    let center_pix = unpack2x16float(splat.pos);
    let extent_pix = unpack2x16float(splat.extent);

    // CUDA's rect: max(extent, filter_r) on each axis. filter_r = cutoff·FilterSize.
    let filter_r = CUTOFF * FILTER_SIZE;
    let half = vec2<f32>(max(extent_pix.x, filter_r), max(extent_pix.y, filter_r));

    let sx = f32(in_vertex_index % 2u == 0u) * 2.0 - 1.0;
    let sy = f32(in_vertex_index < 2u) * 2.0 - 1.0;
    let corner_pix = center_pix + vec2<f32>(sx, sy) * half;  // CUDA Y-up pixel

    // CUDA/WebGPU framebuffer pixel -> wgpu NDC. Framebuffer y grows downward,
    // while clip-space y grows upward, so y needs the sign flip here.
    let vp = vec2<f32>(f32(tex_params.viewport_w), f32(tex_params.viewport_h));
    let ndc = vec2<f32>(
        (corner_pix.x * 2.0 - (vp.x - 1.0)) / vp.x,
        -((corner_pix.y * 2.0 - (vp.y - 1.0)) / vp.y),
    );
    out.position = vec4<f32>(ndc, 0.0, 1.0);

    out.Tu = vec3<f32>(splat.tu_x, splat.tu_y, splat.tu_z);
    out.Tv = vec3<f32>(splat.tv_x, splat.tv_y, splat.tv_z);
    out.Tw = vec3<f32>(splat.tw_x, splat.tw_y, splat.tw_z);

    let rg = unpack2x16float(splat.color_rg);
    let ba = unpack2x16float(splat.color_b_shape);
    out.color = vec4<f32>(rg.x, rg.y, ba.x, splat.opacity);
    out.shape = ba.y;
    out.gauss_id = splat.gauss_id;
    out.center_pix = center_pix;
    out.depth_plane = vec3<f32>(splat.depth_u, splat.depth_v, splat.depth_center);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Matches CUDA renderBakedCUDA's pixf = { pix.x, pix.y }.
    let pixf = floor(in.position.xy);

    // Ray-disk intersection (CUDA renderBakedCUDA inner loop).
    let k_vec = pixf.x * in.Tw - in.Tu;
    let l_vec = pixf.y * in.Tw - in.Tv;
    let p_vec = cross(k_vec, l_vec);
    if abs(p_vec.z) < 1e-12 { discard; }
    let s = vec2<f32>(p_vec.x / p_vec.z, p_vec.y / p_vec.z);
    let rho3d = dot(s, s);

    // Screen-pixel Gaussian distance² (alpha_lp).
    let d_pix = in.center_pix - pixf;
    let rho2d = FILTER_INV_SQUARE * dot(d_pix, d_pix);

    let depth = dot(in.depth_plane, vec3<f32>(s, 1.0));
    if depth < 0.2 { discard; }

    let kt  = tex_params.kernel_type;
    let opa = in.color.a;

    var b: f32;
    if kt == 4u {
        // BetaScaled: k²=9.
        if rho3d >= 9.0 + 1e-6 { discard; }
        let base = max(0.0, 1.0 - rho3d / 9.0);
        let alpha_beta = pow(base, in.shape);
        let alpha_lp   = exp(-rho2d * 0.5);
        b = min(0.99, opa * max(alpha_beta, alpha_lp));
    } else if kt == 1u {
        if rho3d >= 1.0 + 1e-6 { discard; }
        let base = max(0.0, 1.0 - rho3d / 1.0);
        let alpha_beta = pow(base, in.shape);
        let alpha_lp   = exp(-rho2d * 0.5);
        b = min(0.99, opa * max(alpha_beta, alpha_lp));
    } else {
        // Gaussian fallback — CUDA uses rho = min(rho3d, rho2d).
        let rho = min(rho3d, rho2d);
        let power = -0.5 * rho;
        if power > 0.0 { discard; }
        b = min(0.99, opa * exp(power));
    }
    if b < 1.0 / 255.0 { discard; }

    var color = in.color.rgb;

    // BC7 atlas residual lookup (using s = surfel UV from ray-disk).
    if tex_params.atlas_width > 0u {
        let E = bitcast<f32>(tex_params.uv_extent_bits);
        let r_base = in.gauss_id * 5u;
        let u0       = atlas_rects[r_base + 0u];
        let v0_local = atlas_rects[r_base + 1u];
        let w_span   = atlas_rects[r_base + 2u];
        let h_span   = atlas_rects[r_base + 3u];
        let layer    = i32(atlas_rects[r_base + 4u]);

        let au = u0       + (s.x + E) / (2.0 * E) * w_span;
        let av = v0_local + (s.y + E) / (2.0 * E) * h_span;
        let uv = vec2<f32>(
            (au + 0.5) / f32(tex_params.atlas_width),
            (av + 0.5) / f32(tex_params.atlas_layer_h),
        );
        let rgba = textureSampleLevel(atlas, atlas_samp, uv, layer, 0.0);
        color = color + rgba.rgb * tex_params.atlas_scale + vec3<f32>(tex_params.atlas_offset);
    }

    color = max(vec3<f32>(0.0), color + vec3<f32>(tex_params.res_bias));
    return vec4<f32>(color, 1.0) * b;
}
