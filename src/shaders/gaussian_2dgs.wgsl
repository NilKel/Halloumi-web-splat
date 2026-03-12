// 2DGS Surfel Rendering Shader
// Ray-disk intersection + kernel evaluation + atlas texture lookup

const FilterInvSquare: f32 = 2.0;  // 1 / (FilterSize^2) where FilterSize = sqrt(0.5)

// Splat2DGS: 64 bytes (16 u32s). Transmat at f32 precision.
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

// Surfel struct (for means3D access)
struct Surfel {
    x: f32, y: f32, z: f32,
    opacity_shape: u32,
    scale_rot: array<u32, 3>,
};

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>,
};

struct TexParams {
    atlas_width: u32,
    atlas_height: u32,
    kernel_type: u32,       // 0=Gaussian, 1=Beta, 2=Flex, 3=General, 4=BetaScaled
    uv_extent_bits: u32,    // f32 reinterpreted as u32
};

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) @interpolate(flat) Tu: vec3<f32>,
    @location(1) @interpolate(flat) Tv: vec3<f32>,
    @location(2) @interpolate(flat) Tw: vec3<f32>,
    @location(3) @interpolate(flat) base_color: vec3<f32>,
    @location(4) @interpolate(flat) opacity: f32,
    @location(5) @interpolate(flat) center_pix: vec2<f32>,
    @location(6) @interpolate(flat) gauss_id: u32,
    @location(7) @interpolate(flat) gauss_xyz: vec3<f32>,
    @location(8) @interpolate(flat) shape: f32,
};

// Group 0: Point cloud render data (splats)
@group(0) @binding(2)
var<storage, read> splats_2d: array<Splat2DGS>;

// Group 1: Sort info + indices
struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
};
@group(1) @binding(0)
var<storage, read> sort_infos: SortInfos;
@group(1) @binding(4)
var<storage, read> indices: array<u32>;

// Group 2: Surfels (for means3D)
@group(2) @binding(0)
var<storage, read> surfels: array<Surfel>;

// Group 3: Atlas + camera + params
@group(3) @binding(0)
var<storage, read> atlas_texture: array<u32>;  // [H, W, 3] FP16 packed as u32
@group(3) @binding(1)
var<storage, read> atlas_rects: array<f32>;    // [N, 4] flat: (u0_px, v0_px, w_px, h_px)
@group(3) @binding(2)
var<uniform> camera: CameraUniforms;
@group(3) @binding(3)
var<uniform> tex_params: TexParams;


@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    // DEBUG: Only draw instance 0 as a fullscreen quad
    // Pass splat data through opacity varying to test if buffer is readable
    if in_instance_index > 0u {
        out.position = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return out;
    }

    let splat = splats_2d[0];

    let x = f32(in_vertex_index % 2u == 0u) * 2.0 - 1.0;
    let y = f32(in_vertex_index < 2u) * 2.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);

    // Encode: opacity > 0 means buffer was written by preprocess
    out.opacity = splat.opacity;
    // Also pass Tu.x via base_color.r to check transmat
    out.base_color = vec3<f32>(splat.tu_x, splat.tu_y, splat.tu_z);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // GREEN = buffer has data (opacity > 0), RED = buffer is empty
    if in.opacity > 0.0 || in.base_color.x != 0.0 {
        return vec4<f32>(0.0, 1.0, 0.0, 1.0);
    }
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);

    let pix = in.position.xy;

    let Tu = in.Tu;
    let Tv = in.Tv;
    let Tw = in.Tw;

    // Ray-disk intersection
    let k = pix.x * Tw - Tu;
    let l = pix.y * Tw - Tv;
    let p = cross(k, l);

    if abs(p.z) < 1e-6 {
        discard;
    }

    let s = vec2<f32>(p.x / p.z, p.y / p.z);

    // Compute rho3d and rho2d
    let rho3d = dot(s, s);
    let d = in.center_pix - pix;
    let rho2d = FilterInvSquare * dot(d, d);
    let rho = min(rho3d, rho2d);

    // Kernel evaluation
    let opa = in.opacity;
    let kernel_type = tex_params.kernel_type;
    let shape_val = in.shape;
    var alpha: f32;

    if (kernel_type == 1u || kernel_type == 4u) {
        let k_sq = select(1.0, 9.0, kernel_type == 4u);
        if rho3d >= k_sq + 1e-6 {
            discard;
        }
        let base = max(0.0, 1.0 - rho3d / k_sq);
        let alpha_beta = pow(base, shape_val);
        let alpha_lp = exp(-rho2d / 2.0);
        alpha = min(0.99, opa * max(alpha_beta, alpha_lp));
    } else if kernel_type == 2u {
        let power = -0.5 * rho;
        if power > 0.0 { discard; }
        var G = exp(power);
        if shape_val > 0.0 {
            G = (1.0 + shape_val) * G / (1.0 + shape_val * G);
        }
        alpha = min(0.99, opa * G);
    } else if kernel_type == 3u {
        let power = -0.5 * pow(rho, shape_val * 0.5);
        if power > 0.0 { discard; }
        alpha = min(0.99, opa * exp(power));
    } else {
        let power = -0.5 * rho;
        if power > 0.0 { discard; }
        alpha = min(0.99, opa * exp(power));
    }

    if alpha < 1.0 / 255.0 {
        discard;
    }

    // Start with SH base color
    var color = in.base_color;

    // Atlas texture lookup
    let atlas_w = tex_params.atlas_width;
    if atlas_w > 0u {
        let E = bitcast<f32>(tex_params.uv_extent_bits);
        let gauss_id = in.gauss_id;

        // Read per-Gaussian rect from atlas_rects [N, 4]
        let r_base = gauss_id * 4u;
        let u0_px  = atlas_rects[r_base + 0u];
        let v0_px  = atlas_rects[r_base + 1u];
        let u_span = atlas_rects[r_base + 2u];
        let v_span = atlas_rects[r_base + 3u];

        // Surfel s -> atlas pixel coords (texel-center convention)
        // Inverse: tex_coord = (s + E) / (2*E) * span - 0.5
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
        // Atlas layout: [H, W, 3] as FP16 = [H, W, 3] f16 values
        // Packed as u32: atlas_texture[idx] = pack2x16float(ch0, ch1)
        // For 3 channels per pixel: 3 f16 = 1.5 u32 → 2 u32s per pixel (with 1 f16 waste)
        // Actually stored as flat f16 array packed into u32: f16_index = (row * W + col) * 3 + ch
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
    return vec4<f32>(color, 1.0) * alpha;
}

// Read a single FP16 value from the atlas texture buffer
// Atlas is stored as flat [H * W * C] f16 values packed into u32 pairs
fn read_atlas_f16(row: i32, col: i32, ch: u32, atlas_w: i32) -> f32 {
    let f16_index = u32(row * atlas_w + col) * 3u + ch;
    let u32_index = f16_index / 2u;
    let component = f16_index % 2u;
    return unpack2x16float(atlas_texture[u32_index])[component];
}
