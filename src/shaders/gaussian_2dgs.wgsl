// 2DGS Surfel Rendering Shader
// Uses eigenvector-based rendering (same as 3DGS) for projected 2DGS surfels.
// The preprocess computes 2D covariance from surfel axes and eigendecomposes.

const CUTOFF:f32 = 2.3539888583335364; // = sqrt(log(255))

// Splat2DGS: 64 bytes. Eigenvectors stored in f32 fields:
// tu_x, tu_y = v1 / viewport (eigenvector 1)
// tu_z, tv_x = v2 / viewport (eigenvector 2)
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
};

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
};

// Group 0: splat_2d buffer (binding 2)
@group(0) @binding(2)
var<storage, read> splats_2d: array<Splat2DGS>;

// Group 1: sort info + indices
@group(1) @binding(0)
var<storage, read> sort_infos: SortInfos;
@group(1) @binding(4)
var<storage, read> indices: array<u32>;

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

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let a = dot(in.screen_pos, in.screen_pos);
    if a > 2.0 * CUTOFF {
        discard;
    }
    let b = min(0.99, exp(-a) * in.color.a);
    return vec4<f32>(in.color.rgb, 1.0) * b;
}
