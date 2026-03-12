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

    // Build splat2world matrix (3x4)
    // splat2world = [L0, L1, p_orig] as column vectors
    // Then T = transpose(splat2world) * world2ndc * ndc2pix
    let W = camera.proj[0][0]; // These are the full projection matrix entries
    let H_val = viewport.y;
    let W_val = viewport.x;

    // We compute T = transpose(splat2world) * world2ndc * ndc2pix
    // where world2ndc = proj * view (combined), ndc2pix maps NDC to pixel coords
    //
    // Following the CUDA implementation:
    // world2ndc = projmatrix (already view*proj combined in our case? No, proj and view are separate)
    //
    // Actually, the CUDA code uses projmatrix = proj * view (combined MVP).
    // Let's combine them: MVP = proj * view
    let mvp = camera.proj * camera.view;

    // ndc2pix: maps NDC [-1,1] to pixel [0, W-1] x [0, H-1]
    // ndc2pix is a 3x4 matrix:
    // [W/2,  0,   0, (W-1)/2]
    // [0,    H/2, 0, (H-1)/2]
    // [0,    0,   0,  1     ]

    // splat2world (3x4, columns are [L0, L1, p_orig]):
    // Row 0: [L0.x, L1.x, 0, p.x]
    // Row 1: [L0.y, L1.y, 0, p.y]
    // Row 2: [L0.z, L1.z, 0, p.z]
    // (with homogeneous row [0, 0, 0, 1])

    // T = transpose(splat2world) * world2ndc * ndc2pix
    // T is 3x3 where rows are Tu, Tv, Tw

    // Build the combined matrix: world2ndc * ndc2pix
    // world2ndc is 4x4 (mvp), ndc2pix is conceptually 3x4
    // But we want T = splat2world^T * (mvp * ndc2pix)
    // Let's compute it step by step following the CUDA code

    // The CUDA code builds it as:
    // T = transpose(splat2world) * world2ndc * ndc2pix
    // where world2ndc is the GLM column-major proj matrix
    // and ndc2pix maps from clip space to pixel space

    // Let's do it directly:
    // For each row of T (which becomes Tu, Tv, Tw):
    // Tu = L0 dotted through the full transform chain
    // Tv = L1 dotted through the full transform chain
    // Tw = p_orig dotted through the full transform chain

    // The combined transform from world to pixel is:
    // p_pix = ndc2pix * (mvp * p_world)
    // For a 3D point p, the projected pixel position is:
    // clip = mvp * vec4(p, 1)
    // ndc = clip.xyz / clip.w
    // pix.x = (ndc.x + 1) * W/2 = ndc.x * W/2 + (W-1)/2...
    // But actually ndc2pix includes the /w division implicitly through the matrix form

    // Following the CUDA code exactly:
    // The CUDA uses glm column-major matrices. Let me translate carefully.
    //
    // CUDA: world2ndc = glm::mat4(projmatrix[0..15]) where projmatrix = proj*view
    // CUDA: ndc2pix = mat3x4(vec4(W/2,0,0,(W-1)/2), vec4(0,H/2,0,(H-1)/2), vec4(0,0,0,1))
    // CUDA: T = transpose(splat2world) * world2ndc * ndc2pix
    //
    // But glm is column-major, so world2ndc * ndc2pix multiplies as:
    // result[col] = world2ndc * ndc2pix[col]
    // ndc2pix has 3 columns (it's 4x3 in glm notation, or 3x4 in row-major)
    //
    // Actually in the CUDA code, ndc2pix is mat3x4 which in GLM means 3 columns of vec4
    // So ndc2pix * point would give a vec3 result
    // And world2ndc * ndc2pix would be mat3x4 (a 4x3 matrix, 3 columns of vec4)
    //
    // Then T = transpose(splat2world) * (world2ndc * ndc2pix)
    // splat2world is mat3x4 (3 columns of vec4), so transpose is mat4x3 (4 columns of vec3)
    // So T = mat4x3 * mat4x3... hmm that doesn't work
    //
    // Let me re-read the CUDA code more carefully:
    // splat2world is mat3x4: 3 columns of vec4
    //   col0 = vec4(L[0], 0)
    //   col1 = vec4(L[1], 0)
    //   col2 = vec4(p, 1)
    // This represents a 4x3 matrix (4 rows, 3 cols) in math notation
    //
    // world2ndc is mat4 (4x4)
    // ndc2pix is mat3x4: 3 columns of vec4
    //   = 4x3 matrix
    //
    // world2ndc * ndc2pix: mat4 * mat3x4
    // In GLM: mat4 * mat3x4 = mat3x4 (3 columns, each = mat4 * vec4)
    //
    // transpose(splat2world) * (world2ndc * ndc2pix):
    // transpose of mat3x4 is... in GLM, transpose(mat3x4) should give mat4x3
    // mat4x3 has 4 rows, 3 cols
    // But wait: GLM mat3x4 has 3 columns of 4 components
    // So splat2world^T should be a matrix with rows = former columns
    // In GLM: glm::transpose(mat<3,4>) = mat<4,3> (4 columns of 3 components)
    //
    // So T = mat4x3^T... no wait.
    // T = transpose(splat2world) * M where M = world2ndc * ndc2pix (mat3x4, 3 cols of vec4)
    // transpose(splat2world) is mat4x3 (4 cols of vec3)
    // mat4x3 * mat3x4... hmm dimensions don't match for standard matrix multiply
    //
    // Actually I think the CUDA code computes this differently. Let me look again:
    // T = glm::transpose(splat2world) * world2ndc * ndc2pix;
    //
    // This is evaluated right-to-left in GLM:
    // Step 1: temp = world2ndc * ndc2pix → mat4 * mat3x4 → mat3x4 (4x3 in math)
    // Step 2: T = glm::transpose(splat2world) * temp → ? * mat3x4
    //
    // glm::transpose(mat3x4) = mat4x3 (per GLM convention)
    // mat4x3 * mat3x4... in GLM, mat<C1, R> * mat<C2, C1> = mat<C2, R>
    // mat4x3 has C=4 cols, R=3 rows
    // mat3x4 has C=3 cols, R=4 rows
    // So mat4x3 * mat3x4 = mat3x3. (3 cols of vec3)
    //
    // Wait no. In GLM, mat<C,R> means C columns, R rows per column.
    // Multiplication: mat<C1,R1> * mat<C2,R2> requires R2 = C1
    // Result is mat<C2, R1>
    //
    // transpose(splat2world) = transpose(mat<3,4>) = mat<4,3>
    //   → 4 columns, 3 rows
    // world2ndc * ndc2pix = mat<4,4> * mat<3,4> = mat<3,4>
    //   → 3 columns, 4 rows
    //
    // T = mat<4,3> * mat<3,4>: R2=4, C1=4... hmm 4 != 4 → R2 must equal C1
    // Actually mat<4,3> has 4 columns and 3 rows per column. So it's a 3x4 matrix (3 rows, 4 cols).
    // mat<3,4> has 3 columns and 4 rows per column. So it's a 4x3 matrix (4 rows, 3 cols).
    // Matrix multiply: (3x4) * (4x3) = 3x3. ✓
    // In GLM terms: mat<4,3> * mat<3,4> → C1=4, R1=3, C2=3, R2=4. Need R2=C1 → 4=4 ✓
    // Result: mat<C2, R1> = mat<3, 3>. ✓
    //
    // So T is mat3x3 (3 columns of vec3), which gives us Tu, Tv, Tw as rows.
    // T[0] = column 0 = vec3, T[1] = column 1, T[2] = column 2
    // But in the CUDA code, Tu = T[0], Tv = T[1], Tw = T[2] accessed as columns.

    // OK let me just compute it directly in WGSL.
    // I'll compute each element of the 3x3 T matrix.

    // splat2world as a 4x3 matrix (rows are x,y,z,w; cols are u,v,w):
    // [L0.x  L1.x  p.x]
    // [L0.y  L1.y  p.y]
    // [L0.z  L1.z  p.z]
    // [0     0     1  ]

    // Combined world-to-pixel transform M = mvp then ndc2pix:
    // For a world point p, clip = mvp * vec4(p, 1)
    // pixel.x = clip.x / clip.w * W/2 + (W-1)/2
    // pixel.y = clip.y / clip.w * H/2 + (H-1)/2
    // pixel.z = clip.w (for depth)
    //
    // But we need the linear (pre-division) version for the transmat computation.
    // The transmat maps from surfel local coords (u, v, 1) to pixel homogeneous coords.
    //
    // T * [u, v, 1]^T = [pixel_x * w, pixel_y * w, w]
    // where w = clip.w
    //
    // For surfel point: world_pos = p + u * L0 + v * L1
    // clip = mvp * vec4(world_pos, 1) = mvp * vec4(p, 1) + u * mvp * vec4(L0, 0) + v * mvp * vec4(L1, 0)
    //
    // Let cp = mvp * vec4(p, 1)    → vec4
    // Let cu = mvp * vec4(L0, 0)   → vec4
    // Let cv = mvp * vec4(L1, 0)   → vec4
    //
    // clip = cp + u * cu + v * cv
    //
    // pixel_x_homo = clip.x * W/2 + clip.w * (W-1)/2
    // pixel_y_homo = clip.y * H/2 + clip.w * (H-1)/2
    // w = clip.w
    //
    // So:
    // pixel_x_homo = (cp.x + u*cu.x + v*cv.x) * W/2 + (cp.w + u*cu.w + v*cv.w) * (W-1)/2
    //              = u * (cu.x * W/2 + cu.w * (W-1)/2) + v * (cv.x * W/2 + cv.w * (W-1)/2) + (cp.x * W/2 + cp.w * (W-1)/2)
    //
    // This gives us T as:
    // Tu = (cu.x * W/2 + cu.w * Wcx,  cu.y * H/2 + cu.w * Hcy,  cu.w)
    // Tv = (cv.x * W/2 + cv.w * Wcx,  cv.y * H/2 + cv.w * Hcy,  cv.w)
    // Tw = (cp.x * W/2 + cp.w * Wcx,  cp.y * H/2 + cp.w * Hcy,  cp.w)
    // where Wcx = (W-1)/2, Hcy = (H-1)/2

    let cp = mvp * vec4<f32>(xyz, 1.0);
    let cu = mvp * vec4<f32>(L0, 0.0);
    let cv = mvp * vec4<f32>(L1, 0.0);

    let half_w = W_val / 2.0;
    let half_h = H_val / 2.0;
    let cx = W_val / 2.0;
    let cy = H_val / 2.0;

    // Tu, Tv, Tw are ROWS of the pixel_homo mapping matrix, where pixel coords
    // match @builtin(position) in the fragment shader.
    //
    // WebGPU viewport transform:
    //   position.x = (ndc.x + 1) * W/2       → pixel_x_homo = clip.x * W/2 + clip.w * W/2
    //   position.y = (1 - ndc.y) * H/2        → pixel_y_homo = -clip.y * H/2 + clip.w * H/2
    //
    // camera.proj includes VIEWPORT_Y_FLIP (negates clip.y), so the WebGPU viewport
    // Y flip and the projection Y flip combine to give the correct screen position.
    // But for the transmat we need pixel coords matching @builtin(position), hence -half_h.
    let Tu = vec3<f32>(
        cu.x * half_w + cu.w * cx,
        cu.y * (-half_h) + cu.w * cy,
        cu.w
    );
    let Tv = vec3<f32>(
        cv.x * half_w + cv.w * cx,
        cv.y * (-half_h) + cv.w * cy,
        cv.w
    );
    let Tw = vec3<f32>(
        cp.x * half_w + cp.w * cx,
        cp.y * (-half_h) + cp.w * cy,
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

    // DEBUG: use simple NDC projection (same as 3DGS) to test if data is valid
    let ndc_pos = pos2d.xy / pos2d.w;
    let ndc_ext = vec2<f32>(0.01, 0.01);  // small fixed extent in NDC

    splats_2d[store_idx] = Splat2DGS(
        Tu.x, Tu.y, Tu.z,
        Tv.x, Tv.y, Tv.z,
        Tw.x, Tw.y, Tw.z,
        opacity,
        pack2x16float(ndc_pos),
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
