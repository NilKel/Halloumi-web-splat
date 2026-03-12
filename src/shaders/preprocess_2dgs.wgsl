// 2DGS Surfel Preprocessing Shader
// Projects surfel axes (L0, L1) to 2D covariance, eigendecomposes,
// and outputs eigenvectors in the same format as 3DGS for rendering.

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

// Splat2DGS: 64 bytes. Repurposed for eigenvector storage:
// tu_x, tu_y = v1 / viewport (eigenvector 1 in NDC half-units)
// tu_z, tv_x = v2 / viewport (eigenvector 2 in NDC half-units)
// Remaining f32 fields unused (zeroed).
struct Splat2DGS {
    tu_x: f32, tu_y: f32, tu_z: f32,
    tv_x: f32, tv_y: f32, tv_z: f32,
    tw_x: f32, tw_y: f32, tw_z: f32,
    opacity: f32,
    pos: u32,           // NDC center x, y (f16 pair)
    extent: u32,        // unused
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

    let focal = camera.focal;
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

    if idx == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    if z <= 0.0 || z >= 1.0 || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
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

    // Build rotation matrix and surfel local frame
    let R = quat_to_rotmat(rot);
    let L0 = R[0] * sx;  // u-direction in world space
    let L1 = R[1] * sy;  // v-direction in world space

    // Compute 3D covariance from surfel axes: Vrk = L0*L0^T + L1*L1^T
    // (rank-2 covariance — flat disk in 3D)
    let Vrk = mat3x3<f32>(
        vec3<f32>(L0.x*L0.x + L1.x*L1.x, L0.y*L0.x + L1.y*L1.x, L0.z*L0.x + L1.z*L1.x),
        vec3<f32>(L0.x*L0.y + L1.x*L1.y, L0.y*L0.y + L1.y*L1.y, L0.z*L0.y + L1.z*L1.y),
        vec3<f32>(L0.x*L0.z + L1.x*L1.z, L0.y*L0.z + L1.y*L1.z, L0.z*L0.z + L1.z*L1.z),
    );

    // Project to 2D covariance (identical to 3DGS path)
    let J = mat3x3<f32>(
        focal.x / camspace.z,
        0.,
        -(focal.x * camspace.x) / (camspace.z * camspace.z),
        0.,
        -focal.y / camspace.z,
        (focal.y * camspace.y) / (camspace.z * camspace.z),
        0.,
        0.,
        0.
    );

    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;
    let cov = transpose(T) * Vrk * T;

    // Eigendecomposition of 2D covariance (identical to 3DGS)
    let kernel_size = render_settings.kernel_size;
    let diagonal1 = cov[0][0] + kernel_size;
    let offDiagonal = cov[0][1];
    let diagonal2 = cov[1][1] + kernel_size;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let radius = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    let lambda1 = mid + radius;
    let lambda2 = max(mid - radius, 0.1);

    let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    let v1 = sqrt(2.0 * lambda1) * diagonalVector;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xyzw / pos2d.w;

    // Compute screen_pos → surfel UV matrix (B = A⁻¹ * [v1, v2])
    // A maps surfel (u,v) to pixel displacement: A = J_wp * [L0, L1]
    // T = W * J = J_wp^T, so T^T = J_wp
    let TT = transpose(T);
    let a0 = (TT * L0).xy;  // pixel displacement of u-direction
    let a1 = (TT * L1).xy;  // pixel displacement of v-direction
    let det_A = a0.x * a1.y - a0.y * a1.x;
    // B maps screen_pos to surfel (u,v) for texture lookup
    var B = mat2x2<f32>(vec2<f32>(0.0), vec2<f32>(0.0));
    if abs(det_A) > 1e-10 {
        let inv_det = 1.0 / det_A;
        let A_inv = mat2x2<f32>(
            vec2<f32>(a1.y * inv_det, -a0.y * inv_det),
            vec2<f32>(-a1.x * inv_det, a0.x * inv_det)
        );
        B = A_inv * mat2x2<f32>(v1, v2);
    }

    // SH color
    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let color = vec4<f32>(
        max(vec3<f32>(0.0), evaluate_sh(dir, idx, render_settings.max_sh_deg)),
        opacity
    );

    // Store to output buffer
    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);

    // Pack into Splat2DGS fields:
    // tu_x, tu_y = v1 / viewport
    // tu_z, tv_x = v2 / viewport
    // tv_y, tv_z, tw_x, tw_y = B matrix (screen_pos → surfel UV)
    let v_scaled = vec4<f32>(v1 / viewport, v2 / viewport);
    splats_2d[store_idx] = Splat2DGS(
        v_scaled.x, v_scaled.y, v_scaled.z,  // v1/viewport, v2.x/viewport
        v_scaled.w, B[0].x, B[0].y,          // v2.y/viewport, B col0
        B[1].x, B[1].y, 0.0,                 // B col1, unused
        opacity,
        pack2x16float(v_center.xy),
        0u,
        pack2x16float(color.rg),
        pack2x16float(vec2<f32>(color.b, shape)),
        idx,
        0u,
    );

    // Depth sorting
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - 1.0);
    sort_depths[store_idx] = bitcast<u32>(zfar - pos2d.z);
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }
}
