// 2DGS Surfel Preprocess (HW raster path) — TRANSMAT-based.
//
// Computes T = transposeᵀ(splat2world) · world2ndc · ndc2pix exactly like
// preprocess_tile_2dgs.wgsl / CUDA's compute_transmat. Stores (Tu, Tv, Tw)
// columns + screen-space AABB (center_pix, extent_pix) + per-Gaussian color
// (SH-clamped + SB) + shape into Splat2DGS for the HW vertex/fragment.

//const MAX_SH_DEG:u32 = <injected>u;

const SH_C0:f32 = 0.28209479177387814;
const SH_C1 = 0.4886025119029199;
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

struct CameraUniforms {
    view: mat4x4<f32>,
    view_inv: mat4x4<f32>,
    proj: mat4x4<f32>,
    proj_inv: mat4x4<f32>,
    viewport: vec2<f32>,
    focal: vec2<f32>
};

struct Surfel {
    x: f32, y: f32, z: f32,
    opacity_shape: u32,
    scale_rot: array<u32, 3>
};

// Splat2DGS layout matches preprocess_tile_2dgs.wgsl: Tu/Tv/Tw transmat columns
// + opacity + center_pix (f16 pair) + extent_pix (f16 pair) + colors + shape
// + gauss_id. Same struct used by HW raster (gaussian_2dgs.wgsl) and compute
// tile raster (tile_raster_2dgs.wgsl).
struct Splat2DGS {
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opacity: f32,
    pos: u32,           // pixel-space center (f16 pair)
    extent: u32,        // pixel-space half-extent (f16 pair)
    color_rg: u32,
    color_b_shape: u32,
    gauss_id: u32,
    depth_u: f32,
    depth_v: f32,
    depth_center: f32,
    _pad: u32,
};

struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_vertex: u32,
    base_instance: u32,
}

struct DispatchIndirect {
    dispatch_x: atomic<u32>,
    dispatch_y: u32,
    dispatch_z: u32,
}

struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct RenderSettings {
    clipping_box_min: vec4<f32>,
    clipping_box_max: vec4<f32>,
    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_spatting: u32,
    kernel_size: f32,
    walltime: f32,
    scene_extend: f32,
    sh_bias: f32,
    compact_mult: f32,
    sb_number: u32,
    kernel_type: u32,
    _pad1: u32,
    _pad2: u32,
    center: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniforms;

@group(1) @binding(0) var<storage,read> surfels : array<Surfel>;
@group(1) @binding(1) var<storage,read> sh_coefs : array<array<u32,24>>;
@group(1) @binding(2) var<storage,read_write> splats_2d : array<Splat2DGS>;
@group(1) @binding(3) var<storage,read> sb_params : array<f32>;

@group(2) @binding(0) var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1) var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2) var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3) var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(2) @binding(4) var<storage, read_write> draw_indirect: DrawIndirect;

@group(3) @binding(0) var<uniform> render_settings: RenderSettings;


fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let a = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 0u) / 2u])[(c_idx * 3u + 0u) % 2u];
    let b = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 1u) / 2u])[(c_idx * 3u + 1u) % 2u];
    let c = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 2u) / 2u])[(c_idx * 3u + 2u) % 2u];
    return vec3<f32>(a, b, c);
}

fn evaluate_sh(dir: vec3<f32>, v_idx: u32, sh_deg: u32, sh_bias: f32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);
    if sh_deg > 0u {
        let x = dir.x; let y = dir.y; let z = dir.z;
        result += -SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);
        if sh_deg > 1u {
            let xx = x*x; let yy = y*y; let zz = z*z;
            let xy = x*y; let yz = y*z; let xz = x*z;
            result += SH_C2[0]*xy*sh_coef(v_idx, 4u) + SH_C2[1]*yz*sh_coef(v_idx, 5u) + SH_C2[2]*(2.0*zz - xx - yy)*sh_coef(v_idx, 6u) + SH_C2[3]*xz*sh_coef(v_idx, 7u) + SH_C2[4]*(xx - yy)*sh_coef(v_idx, 8u);
            if sh_deg > 2u {
                result += SH_C3[0]*y*(3.0*xx - yy)*sh_coef(v_idx, 9u) + SH_C3[1]*xy*z*sh_coef(v_idx, 10u) + SH_C3[2]*y*(4.0*zz - xx - yy)*sh_coef(v_idx, 11u) + SH_C3[3]*z*(2.0*zz - 3.0*xx - 3.0*yy)*sh_coef(v_idx, 12u) + SH_C3[4]*x*(4.0*zz - xx - yy)*sh_coef(v_idx, 13u) + SH_C3[5]*z*(xx - yy)*sh_coef(v_idx, 14u) + SH_C3[6]*x*(xx - 3.0*yy)*sh_coef(v_idx, 15u);
            }
        }
    }
    result += sh_bias;
    return result;
}

fn eval_sb(gauss_id: u32, view_dir: vec3<f32>, K: u32) -> vec3<f32> {
    if K == 0u { return vec3<f32>(0.0); }
    let softplus_scale: f32 = 10.0 * 0.693147180559945;
    var rgb_sum = vec3<f32>(0.0);
    let base = gauss_id * K * 6u;
    for (var k = 0u; k < K; k = k + 1u) {
        let o = base + k * 6u;
        let r = sb_params[o + 0u]; let g = sb_params[o + 1u]; let b = sb_params[o + 2u];
        let theta = sb_params[o + 3u]; let phi = sb_params[o + 4u]; let beta_raw = sb_params[o + 5u];
        let beta = 4.0 * exp(beta_raw);
        let sr  = log(1.0 + exp(softplus_scale * r)) / softplus_scale;
        let sg  = log(1.0 + exp(softplus_scale * g)) / softplus_scale;
        let sb_ = log(1.0 + exp(softplus_scale * b)) / softplus_scale;
        let st = sin(theta); let ct = cos(theta);
        let sp = sin(phi);   let cp = cos(phi);
        let mu = vec3<f32>(st * cp, st * sp, ct);
        let d = dot(mu, view_dir);
        if d > 0.0 {
            let w = pow(d, beta);
            rgb_sum = rgb_sum + vec3<f32>(sr * w, sg * w, sb_ * w);
        }
    }
    return rgb_sum;
}

fn quat_to_rotmat(q: vec4<f32>) -> mat3x3<f32> {
    let qn = q * inverseSqrt(max(dot(q, q), 1e-20));
    let w = qn.x; let x = qn.y; let y = qn.z; let z = qn.w;
    let x2 = x*x; let y2 = y*y; let z2 = z*z;
    let xy = x*y; let xz = x*z; let yz = y*z;
    let wx = w*x; let wy = w*y; let wz = w*z;
    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(y2 + z2), 2.0*(xy + wz),       2.0*(xz - wy)),
        vec3<f32>(2.0*(xy - wz),       1.0 - 2.0*(x2 + z2), 2.0*(yz + wx)),
        vec3<f32>(2.0*(xz + wy),       2.0*(yz - wx),       1.0 - 2.0*(x2 + y2))
    );
}

// Compute screen-space AABB from transmat. Returns (cx, cy, hx, hy) in pixel
// coords (CUDA Y-up convention — pix.y=H is top). hx/hy < 0 → degenerate.
fn compute_aabb(T: mat3x3<f32>, cutoff: f32) -> vec4<f32> {
    let t = vec3<f32>(cutoff * cutoff, cutoff * cutoff, -1.0);
    let d = dot(t, T[2] * T[2]);
    if d == 0.0 {
        return vec4<f32>(0.0, 0.0, -1.0, -1.0);
    }
    let f = (1.0 / d) * t;
    let p = vec2<f32>(dot(f, T[0] * T[2]), dot(f, T[1] * T[2]));
    let h0 = p * p - vec2<f32>(dot(f, T[0] * T[0]), dot(f, T[1] * T[1]));
    let h = sqrt(max(vec2<f32>(1e-4, 1e-4), h0));
    return vec4<f32>(p.x, p.y, h.x, h.y);
}

@compute @workgroup_size(256,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&surfels) {
        return;
    }

    let viewport = camera.viewport;
    let W = viewport.x;
    let H = viewport.y;
    let surfel = surfels[idx];
    let xyz = vec3<f32>(surfel.x, surfel.y, surfel.z);

    let opa_shape = unpack2x16float(surfel.opacity_shape);
    var opacity = opa_shape.x;
    let shape = opa_shape.y;

    if any(xyz < render_settings.clipping_box_min.xyz) || any(xyz > render_settings.clipping_box_max.xyz) {
        return;
    }

    var camspace = camera.view * vec4<f32>(xyz, 1.0);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z_ndc = pos2d.z / pos2d.w;

    if idx == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    if z_ndc <= 0.0 || z_ndc >= 1.0 || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    if opacity < (1.0 / 255.0) {
        return;
    }

    let scale_packed = unpack2x16float(surfel.scale_rot[0]);
    let rot_wx = unpack2x16float(surfel.scale_rot[1]);
    let rot_yz = unpack2x16float(surfel.scale_rot[2]);

    let scaling = render_settings.gaussian_scaling;
    let sx = scale_packed.x * scaling;
    let sy = scale_packed.y * scaling;
    let rot = vec4<f32>(rot_wx.x, rot_wx.y, rot_yz.x, rot_yz.y);

    let R = quat_to_rotmat(rot);
    let L0 = R[0] * sx;
    let L1 = R[1] * sy;
    let depth_u = (camera.view * vec4<f32>(L0, 0.0)).z;
    let depth_v = (camera.view * vec4<f32>(L1, 0.0)).z;

    // T = transposeᵀ(splat2world) · world2ndc · ndc2pix.
    // Undo the wgpu Y-flip baked into camera.proj (VIEWPORT_Y_FLIP) so we
    // build T against the standard (Y-up NDC) projection — same as CUDA.
    let s2w_r0 = vec4<f32>(L0, 0.0);
    let s2w_r1 = vec4<f32>(L1, 0.0);
    let s2w_r2 = vec4<f32>(xyz, 1.0);

    var proj_raw = camera.proj;
    proj_raw[0].y = -proj_raw[0].y;
    proj_raw[1].y = -proj_raw[1].y;
    proj_raw[2].y = -proj_raw[2].y;
    proj_raw[3].y = -proj_raw[3].y;
    // world2ndc = (PV)^T to match CUDA's GLM column-major reindexing of the
    // PyTorch-stored projmatrix. Identical to preprocess_tile_2dgs.wgsl.
    let M = transpose(proj_raw * camera.view);

    let I0 = vec4<f32>(dot(s2w_r0, M[0]), dot(s2w_r0, M[1]), dot(s2w_r0, M[2]), dot(s2w_r0, M[3]));
    let I1 = vec4<f32>(dot(s2w_r1, M[0]), dot(s2w_r1, M[1]), dot(s2w_r1, M[2]), dot(s2w_r1, M[3]));
    let I2 = vec4<f32>(dot(s2w_r2, M[0]), dot(s2w_r2, M[1]), dot(s2w_r2, M[2]), dot(s2w_r2, M[3]));

    let np0 = vec4<f32>(W / 2.0, 0.0, 0.0, (W - 1.0) / 2.0);
    let np1 = vec4<f32>(0.0, H / 2.0, 0.0, (H - 1.0) / 2.0);
    let np2 = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    let T_mat = mat3x3<f32>(
        vec3<f32>(dot(I0, np0), dot(I1, np0), dot(I2, np0)),
        vec3<f32>(dot(I0, np1), dot(I1, np1), dot(I2, np1)),
        vec3<f32>(dot(I0, np2), dot(I1, np2), dot(I2, np2)),
    );

    // Match CUDA bake_render's default aabb_mode=3: opacity-adaptive cutoff.
    // For BetaScaled (k=3), the compact support still clips at rho3d=9 in the
    // fragment; this cutoff only controls bbox/low-pass center calculation.
    var cutoff: f32;
    if render_settings.kernel_type == 4u {
        let k = 3.0;
        let ratio = 1.0 / (255.0 * opacity);
        var r_beta = 0.0;
        let threshold = pow(ratio, 1.0 / max(shape, 1e-20));
        if threshold < 1.0 {
            r_beta = k * sqrt(1.0 - threshold);
        }
        var r_lp = 0.0;
        let log_term = log(255.0 * opacity);
        if log_term > 0.0 {
            r_lp = sqrt(2.0 * log_term);
        }
        cutoff = min(max(r_beta, r_lp), k + 2.0);
    } else if render_settings.kernel_type == 1u {
        let k = 1.0;
        let ratio = 1.0 / (255.0 * opacity);
        var r_beta = 0.0;
        let threshold = pow(ratio, 1.0 / max(shape, 1e-20));
        if threshold < 1.0 {
            r_beta = k * sqrt(1.0 - threshold);
        }
        var r_lp = 0.0;
        let log_term = log(255.0 * opacity);
        if log_term > 0.0 {
            r_lp = sqrt(2.0 * log_term);
        }
        cutoff = min(max(r_beta, r_lp), k + 2.0);
    } else {
        let log_term = log(255.0 * opacity);
        if log_term > 0.0 {
            cutoff = sqrt(2.0 * log_term * render_settings.compact_mult);
        } else {
            cutoff = 0.1;
        }
        cutoff = min(cutoff, 4.0);
    }

    let aabb = compute_aabb(T_mat, cutoff);
    let center_pix = aabb.xy;
    let extent_pix = aabb.zw;
    if extent_pix.x < 0.0 {
        return;
    }

    // SH + SB folded into per-Gaussian color (preprocess time, view-dir is
    // Gaussian-centric so this is correct for the fragment).
    let camera_pos = camera.view_inv[3].xyz;
    let view_dir = normalize(xyz - camera_pos);
    var color = max(vec3<f32>(0.0),
        evaluate_sh(view_dir, idx, render_settings.max_sh_deg, render_settings.sh_bias));
    color = color + eval_sb(idx, view_dir, render_settings.sb_number);

    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);

    // Store columns of T_mat directly — same as preprocess_tile_2dgs.wgsl. The
    // CUDA `k_vec = pix.x * Tw - Tu` cross-product formula expects Tu/Tv/Tw to
    // be glm columns of T (which encode rows of the math transmat — see CUDA
    // `transMats[idx*9 + i*3 + k] = T[i].k`).
    splats_2d[store_idx] = Splat2DGS(
        T_mat[0].x, T_mat[0].y, T_mat[0].z,  // Tu
        T_mat[1].x, T_mat[1].y, T_mat[1].z,  // Tv
        T_mat[2].x, T_mat[2].y, T_mat[2].z,  // Tw
        opacity,
        pack2x16float(center_pix),
        pack2x16float(extent_pix),
        pack2x16float(vec2<f32>(color.r, color.g)),
        pack2x16float(vec2<f32>(color.b, shape)),
        idx,
        depth_u,
        depth_v,
        camspace.z,
        0u,
    );

    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - 1.0);
    sort_depths[store_idx] = bitcast<u32>(zfar - pos2d.z);
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
