// 2DGS Surfel Rendering Shader
// Ray-disk intersection + hybrid kernel + texture residual lookup

const FilterInvSquare: f32 = 2.0;  // 1 / (FilterSize^2) where FilterSize = sqrt(0.5)

// SH constants for 48D residual evaluation
const SH_C0: f32 = 0.28209479177387814;
const SH_C1: f32 = 0.4886025119029199;
const SH_C2 = array<f32,5>(
    1.0925484305920792,
    -1.0925484305920792,
    0.31539156525252005,
    -1.0925484305920792,
    0.5462742152960396
);
const SH_C3 = array<f32,7>(
    -0.5900435899266435,
    2.890611442640554,
    -0.4570457994644658,
    0.3731763325901154,
    -0.4570457994644658,
    1.445305721320277,
    -0.5900435899266435
);

// Splat2DGS: 40 bytes (10 u32s)
struct Splat2DGS {
    tu_01: u32,
    tu_2_tv_0: u32,
    tv_12: u32,
    tw_01: u32,
    tw_2_opa: u32,
    pos: u32,
    extent: u32,
    color_rg: u32,
    color_b_shape: u32,
    gauss_id: u32,
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
    residual_dim: u32,      // 3 or 48
    kernel_type: u32,       // 0=Gaussian, 1=Beta, 2=Flex, 3=General, 4=BetaScaled
    _pad1: u32,
    _pad2: u32,
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

// Group 0: Point cloud render data (splats + surfels)
@group(0) @binding(2)
var<storage, read> splats_2d: array<Splat2DGS>;

// Group 1: Sort indices
@group(1) @binding(4)
var<storage, read> indices: array<u32>;

// Group 2: Surfels (for means3D)
@group(2) @binding(0)
var<storage, read> surfels: array<Surfel>;

// Group 3: Textures + camera + params
@group(3) @binding(0)
var<storage, read> textures: array<u32>;  // FP16 data packed as u32 pairs
@group(3) @binding(1)
var<uniform> camera: CameraUniforms;
@group(3) @binding(2)
var<uniform> tex_params: TexParams;


fn unpack_f16(packed: u32, idx: u32) -> f32 {
    return unpack2x16float(packed)[idx];
}

@vertex
fn vs_main(
    @builtin(vertex_index) in_vertex_index: u32,
    @builtin(instance_index) in_instance_index: u32,
) -> VertexOutput {
    var out: VertexOutput;

    let splat = splats_2d[indices[in_instance_index]];

    // Unpack transmat
    let tu_xy = unpack2x16float(splat.tu_01);
    let tu_z_tv_x = unpack2x16float(splat.tu_2_tv_0);
    let tv_yz = unpack2x16float(splat.tv_12);
    let tw_xy = unpack2x16float(splat.tw_01);
    let tw_z_opa = unpack2x16float(splat.tw_2_opa);

    out.Tu = vec3<f32>(tu_xy.x, tu_xy.y, tu_z_tv_x.x);
    out.Tv = vec3<f32>(tu_z_tv_x.y, tv_yz.x, tv_yz.y);
    out.Tw = vec3<f32>(tw_xy.x, tw_xy.y, tw_z_opa.x);
    out.opacity = tw_z_opa.y;

    // Unpack center and extent (in NDC)
    let v_center = unpack2x16float(splat.pos);
    let v_extent = unpack2x16float(splat.extent);

    // Unpack base color
    let rg = unpack2x16float(splat.color_rg);
    let b_shape = unpack2x16float(splat.color_b_shape);
    out.base_color = vec3<f32>(rg.x, rg.y, b_shape.x);
    out.shape = b_shape.y;

    // Gaussian ID and position (for texture lookup / viewdir)
    out.gauss_id = splat.gauss_id;
    let surfel = surfels[splat.gauss_id];
    out.gauss_xyz = vec3<f32>(surfel.x, surfel.y, surfel.z);

    // Center in pixel coordinates (for rho2d)
    let viewport = camera.viewport;
    out.center_pix = (v_center + 1.0) * viewport * 0.5;

    // Generate quad vertex: expand AABB
    let x = f32(in_vertex_index % 2u == 0u) * 2.0 - 1.0;
    let y = f32(in_vertex_index < 2u) * 2.0 - 1.0;

    // The extent is already in NDC units, scale by a generous margin
    let margin = 1.2;
    let offset = vec2<f32>(x, y) * v_extent * margin;
    out.position = vec4<f32>(v_center + offset, 0.0, 1.0);

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Fragment position in pixel coordinates
    let pix = in.position.xy;

    let Tu = in.Tu;
    let Tv = in.Tv;
    let Tw = in.Tw;

    // Ray-disk intersection: solve for (s.x, s.y) on the surfel disk
    let k = vec3<f32>(pix.x * Tw.x - Tu.x, pix.x * Tw.y - Tu.y, pix.x * Tw.z - Tu.z);
    let l = vec3<f32>(pix.y * Tw.x - Tv.x, pix.y * Tw.y - Tv.y, pix.y * Tw.z - Tv.z);
    let p = cross(k, l);

    if abs(p.z) < 1e-6 {
        discard;
    }

    let s = vec2<f32>(p.x / p.z, p.y / p.z);

    // Compute rho3d (surfel distance) and rho2d (screen-space low-pass)
    let rho3d = dot(s, s);
    let d = in.center_pix - pix;
    let rho2d = FilterInvSquare * dot(d, d);
    let rho = min(rho3d, rho2d);

    let opa = in.opacity;
    let kernel_type = tex_params.kernel_type;
    let shape_val = in.shape;
    var alpha: f32;

    if (kernel_type == 1u || kernel_type == 4u) {
        // Beta kernel: (1 - rho3d/k²)^shape, with Gaussian low-pass handoff
        // kernel_type 1: k²=1 (unit disk), kernel_type 4: k²=9 (3σ scaled)
        let k_sq = select(1.0, 9.0, kernel_type == 4u);
        if rho3d >= k_sq + 1e-6 {
            discard;
        }
        let base = max(0.0, 1.0 - rho3d / k_sq);
        let alpha_beta = pow(base, shape_val);
        let alpha_lp = exp(-rho2d / 2.0);
        let kernel_val = max(alpha_beta, alpha_lp);
        alpha = min(0.99, opa * kernel_val);
    } else if kernel_type == 2u {
        // Flex kernel: modified Gaussian with per-Gaussian beta
        let power = -0.5 * rho;
        if power > 0.0 {
            discard;
        }
        var G = exp(power);
        if shape_val > 0.0 {
            G = (1.0 + shape_val) * G / (1.0 + shape_val * G);
        }
        alpha = min(0.99, opa * G);
    } else if kernel_type == 3u {
        // General kernel: isotropic generalized Gaussian exp(-0.5 * rho^(beta/2))
        let power = -0.5 * pow(rho, shape_val * 0.5);
        if power > 0.0 {
            discard;
        }
        alpha = min(0.99, opa * exp(power));
    } else {
        // Default: standard Gaussian kernel (kernel_type 0)
        let power = -0.5 * rho;
        if power > 0.0 {
            discard;
        }
        alpha = min(0.99, opa * exp(power));
    }

    if alpha < 1.0 / 255.0 {
        discard;
    }

    // Start with SH base color
    var color = in.base_color;

    // Texture residual lookup
    let residual_dim = tex_params.residual_dim;
    if residual_dim > 0u {
        // Map surfel coords to texel coords: s ∈ [-4, 4] → texel ∈ [0, 8)
        // Texel-center convention: sample i at (i+0.5)/8 * 8 - 4 = i + 0.5 - 4 = i - 3.5
        // Inverse: texel = s + 3.5
        let tex_u = clamp(s.x + 3.5, 0.0, 6.999);
        let tex_v = clamp(s.y + 3.5, 0.0, 6.999);

        let u0 = i32(tex_u);
        let v0 = i32(tex_v);
        let fu = tex_u - f32(u0);
        let fv = tex_v - f32(v0);
        let u1 = min(u0 + 1, 7);
        let v1 = min(v0 + 1, 7);

        // Bilinear weights
        let w00 = (1.0 - fu) * (1.0 - fv);
        let w10 = fu * (1.0 - fv);
        let w01 = (1.0 - fu) * fv;
        let w11 = fu * fv;

        let gauss_id = in.gauss_id;

        if residual_dim == 48u {
            // 48D SH residual: per-texel SH coefficients, evaluate at viewdir
            // Texture layout: [N, 8, 8, 48] as FP16, packed as u32 pairs
            // Channel layout within 48D: [R×16, G×16, B×16]
            let tex_stride = 8u * 8u * 48u;  // 3072 f16 values per Gaussian
            let base = gauss_id * tex_stride; // in f16 units

            // Compute view direction at Gaussian center
            let cam_pos = camera.view_inv[3].xyz;
            let dir = normalize(in.gauss_xyz - cam_pos);

            // Precompute SH basis (degree 3, 16 terms)
            let dx = dir.x; let dy = dir.y; let dz = dir.z;
            let xx = dx*dx; let yy = dy*dy; let zz = dz*dz;
            let xy = dx*dy; let yz = dy*dz; let xz = dx*dz;

            var sh_basis: array<f32, 16>;
            sh_basis[0]  = SH_C0;
            sh_basis[1]  = -SH_C1 * dy;
            sh_basis[2]  = SH_C1 * dz;
            sh_basis[3]  = -SH_C1 * dx;
            sh_basis[4]  = SH_C2[0] * xy;
            sh_basis[5]  = SH_C2[1] * yz;
            sh_basis[6]  = SH_C2[2] * (2.0*zz - xx - yy);
            sh_basis[7]  = SH_C2[3] * xz;
            sh_basis[8]  = SH_C2[4] * (xx - yy);
            sh_basis[9]  = SH_C3[0] * dy * (3.0*xx - yy);
            sh_basis[10] = SH_C3[1] * xy * dz;
            sh_basis[11] = SH_C3[2] * dy * (4.0*zz - xx - yy);
            sh_basis[12] = SH_C3[3] * dz * (2.0*zz - 3.0*xx - 3.0*yy);
            sh_basis[13] = SH_C3[4] * dx * (4.0*zz - xx - yy);
            sh_basis[14] = SH_C3[5] * dz * (xx - yy);
            sh_basis[15] = SH_C3[6] * dx * (xx - 3.0*yy);

            // Texel offsets (in f16 units)
            let off00 = base + u32(v0 * 8 + u0) * 48u;
            let off10 = base + u32(v0 * 8 + u1) * 48u;
            let off01 = base + u32(v1 * 8 + u0) * 48u;
            let off11 = base + u32(v1 * 8 + u1) * 48u;

            // Per-channel SH evaluation: channel layout is [R×16, G×16, B×16]
            for (var ch = 0u; ch < 3u; ch++) {
                let ch_off = ch * 16u;
                var result = 0.0;
                for (var k_sh = 0u; k_sh < 16u; k_sh++) {
                    let f16_idx = ch_off + k_sh;
                    // Read FP16 from packed u32 array
                    let c00 = read_f16(off00 + f16_idx);
                    let c10 = read_f16(off10 + f16_idx);
                    let c01 = read_f16(off01 + f16_idx);
                    let c11 = read_f16(off11 + f16_idx);

                    let val = w00 * c00 + w10 * c10 + w01 * c01 + w11 * c11;
                    result += sh_basis[k_sh] * val;
                }
                color[ch] += result;
            }
        } else if residual_dim == 3u {
            // 3D DC residual: direct RGB addition
            let tex_stride = 8u * 8u * 3u;  // 192 f16 values per Gaussian
            let base = gauss_id * tex_stride;

            for (var ch = 0u; ch < 3u; ch++) {
                let c00 = read_f16(base + u32(v0 * 8 + u0) * 3u + ch);
                let c10 = read_f16(base + u32(v0 * 8 + u1) * 3u + ch);
                let c01 = read_f16(base + u32(v1 * 8 + u0) * 3u + ch);
                let c11 = read_f16(base + u32(v1 * 8 + u1) * 3u + ch);

                color[ch] += w00 * c00 + w10 * c10 + w01 * c01 + w11 * c11;
            }
        }
    }

    // Premultiplied alpha output
    return vec4<f32>(color, 1.0) * alpha;
}

// Read a single FP16 value from the packed u32 texture buffer
// f16_index is the index in f16 units (not bytes, not u32s)
fn read_f16(f16_index: u32) -> f32 {
    let u32_index = f16_index / 2u;
    let component = f16_index % 2u;
    return unpack2x16float(textures[u32_index])[component];
}
