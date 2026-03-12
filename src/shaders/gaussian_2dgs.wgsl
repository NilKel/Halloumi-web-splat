// 2DGS Surfel Rendering Shader
// Eigenvector-based Gaussian splatting with optional atlas texture residual.

const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

// Splat2DGS: 64 bytes. Layout:
// tu_x, tu_y = v1 / viewport (eigenvector 1)
// tu_z, tv_x = v2 / viewport (eigenvector 2)
// tv_y, tv_z = B matrix col 0 (screen_pos → surfel UV)
// tw_x, tw_y = B matrix col 1
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

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) screen_pos: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) @interpolate(flat) B_col0: vec2<f32>,
    @location(3) @interpolate(flat) B_col1: vec2<f32>,
    @location(4) @interpolate(flat) gauss_id: u32,
};

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
};

struct TexParams {
    atlas_width: u32,
    atlas_height: u32,
    kernel_type: u32,
    uv_extent_bits: u32,
};

// Group 0: splat_2d buffer (binding 2)
@group(0) @binding(2)
var<storage, read> splats_2d: array<Splat2DGS>;

// Group 1: sort info + indices
@group(1) @binding(0)
var<storage, read> sort_infos: SortInfos;
@group(1) @binding(4)
var<storage, read> indices: array<u32>;

// Group 3: atlas + params (bindings 0,1,3 used; binding 2 = camera unused)
@group(3) @binding(0)
var<storage, read> atlas_texture: array<u32>;
@group(3) @binding(1)
var<storage, read> atlas_rects: array<f32>;
@group(3) @binding(3)
var<uniform> tex_params: TexParams;

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // Instance culling (matching 3DGS pattern)
    let visible_count = sort_infos.keys_size;
    if in_instance_index >= visible_count {
        out.position = vec4<f32>(0.0, 0.0, 2.0, 1.0);
        return out;
    }

    let splat = splats_2d[indices[in_instance_index]];

    // Read eigenvectors (stored as v/viewport in f32 fields)
    let v1 = vec2<f32>(splat.tu_x, splat.tu_y);
    let v2 = vec2<f32>(splat.tu_z, splat.tv_x);
    let v_center = unpack2x16float(splat.pos);

    // Quad vertex (same as 3DGS)
    let x = f32(in_vertex_index % 2u == 0u) * 2.0 - 1.0;
    let y = f32(in_vertex_index < 2u) * 2.0 - 1.0;
    let position = vec2<f32>(x, y) * CUTOFF;

    let offset = 2.0 * mat2x2<f32>(v1, v2) * position;
    out.position = vec4<f32>(v_center + offset, 0.0, 1.0);
    out.screen_pos = position;

    let rg = unpack2x16float(splat.color_rg);
    let ba = unpack2x16float(splat.color_b_shape);
    out.color = vec4<f32>(rg.x, rg.y, ba.x, splat.opacity);

    // Pass B matrix and gauss_id for texture lookup
    out.B_col0 = vec2<f32>(splat.tv_y, splat.tv_z);
    out.B_col1 = vec2<f32>(splat.tw_x, splat.tw_y);
    out.gauss_id = splat.gauss_id;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2.0 * CUTOFF {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color.a);

    // Base SH color
    var color = in.color.rgb;

    // Atlas texture residual lookup
    let atlas_w = tex_params.atlas_width;
    if atlas_w > 0u {
        let E = bitcast<f32>(tex_params.uv_extent_bits);

        // Convert screen_pos to surfel UV via precomputed B matrix
        let B = mat2x2<f32>(in.B_col0, in.B_col1);
        let s = B * in.screen_pos;

        // Map surfel UV [-E, +E] to atlas pixel coordinates
        let gauss_id = in.gauss_id;
        let r_base = gauss_id * 4u;
        let u0_px  = atlas_rects[r_base + 0u];
        let v0_px  = atlas_rects[r_base + 1u];
        let u_span = atlas_rects[r_base + 2u];
        let v_span = atlas_rects[r_base + 3u];

        let au = u0_px + (s.x + E) / (2.0 * E) * u_span - 0.5;
        let av = v0_px + (s.y + E) / (2.0 * E) * v_span - 0.5;
        let au_c = clamp(au, u0_px, u0_px + u_span - 1.001);
        let av_c = clamp(av, v0_px, v0_px + v_span - 1.001);

        let au0 = i32(au_c);
        let av0 = i32(av_c);
        let fu = au_c - f32(au0);
        let fv = av_c - f32(av0);
        let au1 = min(au0 + 1, i32(u0_px + u_span - 1.0));
        let av1 = min(av0 + 1, i32(v0_px + v_span - 1.0));

        // Bilinear interpolation from atlas (3 channels, FP16 packed as u32)
        let aw = i32(atlas_w);
        for (var ch = 0u; ch < 3u; ch++) {
            let c00 = read_atlas_f16(av0, au0, ch, aw);
            let c10 = read_atlas_f16(av0, au1, ch, aw);
            let c01 = read_atlas_f16(av1, au0, ch, aw);
            let c11 = read_atlas_f16(av1, au1, ch, aw);
            color[ch] += (1.0 - fu) * (1.0 - fv) * c00
                       + fu * (1.0 - fv) * c10
                       + (1.0 - fu) * fv * c01
                       + fu * fv * c11;
        }
    }

    // Premultiplied alpha output
    return vec4<f32>(color, 1.0) * b;
}

// Read a single FP16 value from the atlas texture buffer
fn read_atlas_f16(row: i32, col: i32, ch: u32, atlas_w: i32) -> f32 {
    let f16_index = u32(row * atlas_w + col) * 3u + ch;
    let u32_index = f16_index / 2u;
    let component = f16_index % 2u;
    return unpack2x16float(atlas_texture[u32_index])[component];
}
