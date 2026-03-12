// 2DGS Surfel Preprocessing Shader
// Computes transmat [Tu, Tv, Tw] for ray-disk intersection rendering

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

// 2DGS Surfel: xyz(3×f32) + opacity(f16+pad) + scale(2×f16) + rotation(4×f16)
// Total 28 bytes, matching Gaussian struct size
struct Surfel {
    x: f32, y: f32, z: f32,
    opacity_shape: u32,     // f16 opacity + f16 shape (beta kernel parameter)
    scale_rot: array<u32, 3> // [scale_xy, rot_wx, rot_yz] as packed f16 pairs
};

// 2DGS Splat output: transmat (f32) + AABB + color + ID
// Total 64 bytes (16 u32s). Transmat at f32 to avoid precision loss.
struct Splat2DGS {
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opacity: f32,
    pos: u32,           // NDC center x, y (f16 pair)
    extent: u32,        // NDC extent x, y (f16 pair)
    color_rg: u32,      // R, G (f16 pair)
    color_b_shape: u32, // B, shape (f16 pair)
    gauss_id: u32,      // original Gaussian index
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
    center: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage,read> surfels : array<Surfel>;
@group(1) @binding(1)
var<storage,read> sh_coefs : array<array<u32,24>>;

@group(1) @binding(2)
var<storage,read_write> splats_2d : array<Splat2DGS>;

@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths : array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices : array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(2) @binding(4)
var<storage, read_write> draw_indirect: DrawIndirect;

@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;


/// reads the ith sh coef from the sh buffer
fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let a = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 0u) / 2u])[(c_idx * 3u + 0u) % 2u];
    let b = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 1u) / 2u])[(c_idx * 3u + 1u) % 2u];
    let c = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 2u) / 2u])[(c_idx * 3u + 2u) % 2u];
    return vec3<f32>(a, b, c);
}

// spherical harmonics evaluation with Condon-Shortley phase
fn evaluate_sh(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
    var result = SH_C0 * sh_coef(v_idx, 0u);

    if sh_deg > 0u {
        let x = dir.x;
        let y = dir.y;
        let z = dir.z;

        result += - SH_C1 * y * sh_coef(v_idx, 1u) + SH_C1 * z * sh_coef(v_idx, 2u) - SH_C1 * x * sh_coef(v_idx, 3u);

        if sh_deg > 1u {
            let xx = x * x;
            let yy = y * y;
            let zz = z * z;
            let xy = x * y;
            let yz = y * z;
            let xz = x * z;

            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);

            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    result += 0.5;
    return result;
}

// Build 3x3 rotation matrix from quaternion (w, x, y, z)
fn quat_to_rotmat(q: vec4<f32>) -> mat3x3<f32> {
    let w = q.x; let x = q.y; let y = q.z; let z = q.w;
    let x2 = x * x; let y2 = y * y; let z2 = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2))
    );
}

@compute @workgroup_size(256,1,1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) wgs: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&surfels) {
        return;
    }

    let viewport = camera.viewport;
    let surfel = surfels[idx];
    let xyz = vec3<f32>(surfel.x, surfel.y, surfel.z);

    // Unpack opacity and shape
    let opa_shape = unpack2x16float(surfel.opacity_shape);
    var opacity = opa_shape.x;
    let shape = opa_shape.y;

    // Clipping box culling
    if any(xyz < render_settings.clipping_box_min.xyz) || any(xyz > render_settings.clipping_box_max.xyz) {
        return;
    }

    // Frustum culling
    var camspace = camera.view * vec4<f32>(xyz, 1.0);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z = pos2d.z / pos2d.w;

    if z <= 0.0 || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    // Unpack scale (2D) and rotation (quaternion)
    let scale_packed = unpack2x16float(surfel.scale_rot[0]);
    let rot_wx = unpack2x16float(surfel.scale_rot[1]);
    let rot_yz = unpack2x16float(surfel.scale_rot[2]);

    let scaling = render_settings.gaussian_scaling;

    let sx = scale_packed.x * scaling;
    let sy = scale_packed.y * scaling;
    let rot = vec4<f32>(rot_wx.x, rot_wx.y, rot_yz.x, rot_yz.y);

    // Build rotation matrix and local frame
    let R = quat_to_rotmat(rot);
    // L = R * S where S = diag(sx, sy) — only first two columns
    let L0 = R[0] * sx;  // first column of local frame
    let L1 = R[1] * sy;  // second column of local frame

    // Normal = cross(L0, L1) normalized
    var normal = cross(L0, L1);
    let normal_len = length(normal);
    if normal_len > 1e-7 {
        normal = normal / normal_len;
    }

    // Transform normal to view space for backface culling
    let view_normal = (camera.view * vec4<f32>(normal, 0.0)).xyz;
    let ray_dir = camspace.xyz;
    if dot(view_normal, ray_dir) > 0.0 {
        normal = -normal;
    }

    // Compute transmat T = transpose(splat2world) * world2ndc * ndc2pix
    // Tu, Tv, Tw are COLUMNS of T: pixel_homo = u*Tu + v*Tv + Tw
    //
    // For surfel point: world_pos = p + u*L0 + v*L1
    // clip = mvp * (world_pos, 1) = cp + u*cu + v*cv
    //
    // WebGPU viewport: position = (ndc + 1) / 2 * [W, H]
    //   pixel_x_homo = clip.x * W/2 + clip.w * W/2
    //   pixel_y_homo = clip.y * H/2 + clip.w * H/2
    //   w = clip.w
    //
    // camera.proj has VIEWPORT_Y_FLIP baked in (clip.y already negated),
    // so pixel_y correctly increases downward matching @builtin(position).
    let mvp = camera.proj * camera.view;

    let cp = mvp * vec4<f32>(xyz, 1.0);
    let cu = mvp * vec4<f32>(L0, 0.0);
    let cv = mvp * vec4<f32>(L1, 0.0);

    let half_w = viewport.x / 2.0;
    let half_h = viewport.y / 2.0;

    // Tu, Tv, Tw: columns of the transmat
    // ndc2pix uses +half_h (no additional Y flip — VIEWPORT_Y_FLIP is in proj)
    let Tu = vec3<f32>(
        cu.x * half_w + cu.w * half_w,
        cu.y * half_h + cu.w * half_h,
        cu.w
    );
    let Tv = vec3<f32>(
        cv.x * half_w + cv.w * half_w,
        cv.y * half_h + cv.w * half_h,
        cv.w
    );
    let Tw = vec3<f32>(
        cp.x * half_w + cp.w * half_w,
        cp.y * half_h + cp.w * half_h,
        cp.w
    );

    // Compute AABB from transmat
    let cutoff = 4.0; // 4 sigma cutoff for Gaussian kernel
    let t = vec3<f32>(cutoff * cutoff, cutoff * cutoff, -1.0);
    let T2_sq = Tw * Tw; // element-wise
    let d = dot(t, T2_sq);
    if d == 0.0 {
        return;
    }
    let f = (1.0 / d) * t;

    let p_center = vec2<f32>(
        dot(f, Tu * Tw),
        dot(f, Tv * Tw)
    );

    let h0 = p_center * p_center - vec2<f32>(
        dot(f, Tu * Tu),
        dot(f, Tv * Tv)
    );
    let h = sqrt(max(vec2<f32>(1e-4, 1e-4), h0));

    // Evaluate SH for base color
    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let color = vec4<f32>(
        max(vec3<f32>(0.0), evaluate_sh(dir, idx, render_settings.max_sh_deg)),
        opacity
    );

    // Store to output buffer
    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);

    // Convert pixel-space AABB to NDC for vertex shader output
    // WebGPU viewport: position = (ndc + 1) / 2 * [W, H]
    // Inverse: ndc = position * 2 / [W, H] - 1
    let ndc_center = p_center * 2.0 / viewport - 1.0;
    let ndc_ext = h * 2.0 / viewport;

    splats_2d[store_idx] = Splat2DGS(
        Tu.x, Tu.y, Tu.z,
        Tv.x, Tv.y, Tv.z,
        Tw.x, Tw.y, Tw.z,
        opacity,
        pack2x16float(ndc_center),
        pack2x16float(ndc_ext),
        pack2x16float(vec2<f32>(color.x, color.y)),
        pack2x16float(vec2<f32>(color.z, shape)),
        idx,
        0u,
    );

    // Depth sorting
    let znear = -camera.proj[3][2] / camera.proj[2][2];
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - 1.0);
    sort_depths[store_idx] = bitcast<u32>(zfar - pos2d.z);
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
