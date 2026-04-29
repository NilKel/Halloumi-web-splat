// 2DGS Surfel preprocessing for tile-based compute rasterizer.
// Computes transmat (Tu, Tv, Tw) for ray-disk intersection, tile AABB, and tile binning.
// Matches CUDA compute_transmat + compute_aabb from diff_surfel_bake_render.

//const MAX_SH_DEG:u32 = <injected>u;

//const TILE_SIZE: u32 = <injected>u;
const FILTER_SIZE: f32 = 0.707106; // sqrt(2) / 2
const FILTER_INV_SQUARE: f32 = 2.0; // 1 / (2 * 0.3^2) ... actually = 2.0 per CUDA

const SH_C0: f32 = 0.28209479177387814;
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

// 2DGS Surfel: xyz(3×f32) + opacity(f16)+shape(f16) + scale(2×f16) + rotation(4×f16)
struct Surfel {
    x: f32, y: f32, z: f32,
    opacity_shape: u32,     // f16 opacity + f16 shape
    scale_rot: array<u32, 3> // [scale_xy, rot_wx, rot_yz] as packed f16 pairs
};

// Splat2DGS: 64 bytes. Stores transmat (Tu, Tv, Tw) for ray-disk intersection.
struct Splat2DGS {
    tu_x: f32, tu_y: f32, tu_z: f32,  // Tu (transmat column 0)
    tv_x: f32, tv_y: f32, tv_z: f32,  // Tv (transmat column 1)
    tw_x: f32, tw_y: f32, tw_z: f32,  // Tw (transmat column 2)
    opacity: f32,
    pos: u32,           // pixel-space center x, y (f16 pair)
    extent: u32,        // unused
    color_rg: u32,      // R, G (f16 pair)
    color_b_shape: u32, // B, shape (f16 pair)
    gauss_id: u32,      // original Gaussian index
    depth_u: f32,
    depth_v: f32,
    depth_center: f32,
    _pad: u32,
};

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
    sh_bias: f32,       // additive SH bias before ReLU (CUDA d_sh_bias)
    compact_mult: f32,  // FastGS Compact Box multiplier on AdR r_lp (CUDA d_compact_mult)
    sb_number: u32,     // number of SB lobes per Gaussian (0 = SB disabled)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    // scene_center lands at offset 80 (16-byte aligned) — vec4 alignment requirement.
    center: vec4<f32>,
}

struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_vertex: u32,
    base_instance: u32,
}

struct TileInfo {
    tiles_x: u32,
    tiles_y: u32,
    total_tiles: u32,
    _pad: u32,
}

// Group 0: Camera
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

// Group 1: Surfel data + SH + output splats (same layout as PointCloud::bind_group_layout)
@group(1) @binding(0)
var<storage, read> surfels: array<Surfel>;
@group(1) @binding(1)
var<storage, read> sh_coefs: array<array<u32, 24>>;
@group(1) @binding(2)
var<storage, read_write> splats_2d: array<Splat2DGS>;

// Group 2: Sort structures
@group(2) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(2) @binding(1)
var<storage, read_write> sort_depths: array<u32>;
@group(2) @binding(2)
var<storage, read_write> sort_indices: array<u32>;
@group(2) @binding(3)
var<storage, read_write> sort_dispatch: DispatchIndirect;
@group(2) @binding(4)
var<storage, read_write> draw_indirect: DrawIndirect;

// Group 3: Render settings + tile binning outputs
@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;
@group(3) @binding(1)
var<storage, read_write> tiles_touched: array<u32>;
@group(3) @binding(2)
var<storage, read_write> rect_data: array<vec4<u32>>;
@group(3) @binding(3)
var<storage, read_write> depth_16: array<u32>;
@group(3) @binding(4)
var<uniform> tile_info: TileInfo;
// Spherical-Beta lobes: flat array of f32, laid out as [N, sb_number, 6]
// (per lobe: r, g, b, theta, phi, beta_raw). Bound as a storage buffer; even
// when sb_number == 0, a dummy 16-byte buffer is provided so the binding
// is always valid.
@group(3) @binding(5)
var<storage, read> sb_params: array<f32>;

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
            let xx = x * x; let yy = y * y; let zz = z * z;
            let xy = x * y; let yz = y * z; let xz = x * z;
            result += SH_C2[0] * xy * sh_coef(v_idx, 4u) + SH_C2[1] * yz * sh_coef(v_idx, 5u) + SH_C2[2] * (2.0 * zz - xx - yy) * sh_coef(v_idx, 6u) + SH_C2[3] * xz * sh_coef(v_idx, 7u) + SH_C2[4] * (xx - yy) * sh_coef(v_idx, 8u);
            if sh_deg > 2u {
                result += SH_C3[0] * y * (3.0 * xx - yy) * sh_coef(v_idx, 9u) + SH_C3[1] * xy * z * sh_coef(v_idx, 10u) + SH_C3[2] * y * (4.0 * zz - xx - yy) * sh_coef(v_idx, 11u) + SH_C3[3] * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * sh_coef(v_idx, 12u) + SH_C3[4] * x * (4.0 * zz - xx - yy) * sh_coef(v_idx, 13u) + SH_C3[5] * z * (xx - yy) * sh_coef(v_idx, 14u) + SH_C3[6] * x * (xx - 3.0 * yy) * sh_coef(v_idx, 15u);
            }
        }
    }
    // NOTE: the final clamp(>=0) is applied by the caller, matching
    // computeColorFromSH in CUDA (result + d_sh_bias, then max(0)).
    result += sh_bias;
    return result;
}

// Spherical-Beta (SB) view-dependent lobes. Per lobe: (r, g, b, theta, phi,
// beta_raw). Matches CUDA eval_sb in diff_surfel_bake_render/forward.cu.
// Returns a non-negative RGB lobe sum (softplus-activated colors, gated on
// dot(mu, view) > 0, so output is always >= 0 and only adds to SH color).
fn eval_sb(gauss_id: u32, view_dir: vec3<f32>) -> vec3<f32> {
    let K = render_settings.sb_number;
    if K == 0u {
        return vec3<f32>(0.0);
    }
    let softplus_scale: f32 = 10.0 * 0.693147180559945;  // 10 * ln(2)
    var rgb_sum = vec3<f32>(0.0);
    let base = gauss_id * K * 6u;
    for (var k = 0u; k < K; k = k + 1u) {
        let o = base + k * 6u;
        let r        = sb_params[o + 0u];
        let g        = sb_params[o + 1u];
        let b        = sb_params[o + 2u];
        let theta    = sb_params[o + 3u];
        let phi      = sb_params[o + 4u];
        let beta_raw = sb_params[o + 5u];
        let beta = 4.0 * exp(beta_raw);
        // Softplus activation on RGB (steep, matches CUDA).
        let sr = log(1.0 + exp(softplus_scale * r)) / softplus_scale;
        let sg = log(1.0 + exp(softplus_scale * g)) / softplus_scale;
        let sb_ = log(1.0 + exp(softplus_scale * b)) / softplus_scale;
        let st = sin(theta);
        let ct = cos(theta);
        let sp = sin(phi);
        let cp = cos(phi);
        let mu = vec3<f32>(st * cp, st * sp, ct);
        let d = dot(mu, view_dir);
        if d > 0.0 {
            let w = pow(d, beta);
            rgb_sum = rgb_sum + vec3<f32>(sr * w, sg * w, sb_ * w);
        }
    }
    return rgb_sum;
}

// Build 3x3 rotation matrix from quaternion (w, x, y, z)
fn quat_to_rotmat(q: vec4<f32>) -> mat3x3<f32> {
    let qn = q * inverseSqrt(max(dot(q, q), 1e-20));
    let w = qn.x; let x = qn.y; let y = qn.z; let z = qn.w;
    let x2 = x * x; let y2 = y * y; let z2 = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;

    return mat3x3<f32>(
        vec3<f32>(1.0 - 2.0 * (y2 + z2), 2.0 * (xy + wz), 2.0 * (xz - wy)),
        vec3<f32>(2.0 * (xy - wz), 1.0 - 2.0 * (x2 + z2), 2.0 * (yz + wx)),
        vec3<f32>(2.0 * (xz + wy), 2.0 * (yz - wx), 1.0 - 2.0 * (x2 + y2))
    );
}

// Compute AABB from transmat (matches CUDA compute_aabb)
fn compute_aabb(T: mat3x3<f32>, cutoff: f32) -> vec4<f32> {
    // T[0] = Tu, T[1] = Tv, T[2] = Tw (columns)
    let t = vec3<f32>(cutoff * cutoff, cutoff * cutoff, -1.0);
    let d = dot(t, T[2] * T[2]);
    if d == 0.0 {
        return vec4<f32>(0.0, 0.0, -1.0, -1.0); // invalid
    }
    let f = (1.0 / d) * t;

    let p = vec2<f32>(
        dot(f, T[0] * T[2]),
        dot(f, T[1] * T[2])
    );

    let h0 = p * p - vec2<f32>(
        dot(f, T[0] * T[0]),
        dot(f, T[1] * T[1])
    );

    let h = sqrt(max(vec2<f32>(1e-4, 1e-4), h0));
    return vec4<f32>(p.x, p.y, h.x, h.y); // center_x, center_y, extent_x, extent_y
}

@compute @workgroup_size(256, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&surfels) {
        return;
    }

    let viewport = camera.viewport;
    let W = viewport.x;
    let H = viewport.y;
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

    // Unpack scale and rotation
    let scale_packed = unpack2x16float(surfel.scale_rot[0]);
    let rot_wx = unpack2x16float(surfel.scale_rot[1]);
    let rot_yz = unpack2x16float(surfel.scale_rot[2]);

    let scaling = render_settings.gaussian_scaling;
    let sx = scale_packed.x * scaling;
    let sy = scale_packed.y * scaling;
    let rot = vec4<f32>(rot_wx.x, rot_wx.y, rot_yz.x, rot_yz.y);

    // Build rotation matrix and scaled local frame
    let R = quat_to_rotmat(rot);
    let L0 = R[0] * sx;  // u-direction in world space
    let L1 = R[1] * sy;  // v-direction in world space
    let depth_u = (camera.view * vec4<f32>(L0, 0.0)).z;
    let depth_v = (camera.view * vec4<f32>(L1, 0.0)).z;

    // Compute transmat: T = transpose(s2w) * world2ndc * ndc2pix
    // Using explicit vec4 math to avoid mat3x4/mat4x3 types (naga/Metal compatibility).
    //
    // splat2world columns: c0 = (L0, 0), c1 = (L1, 0), c2 = (xyz, 1)
    // transpose(s2w) has rows = s2w columns:
    //   row0 = (L0.x, L0.y, L0.z, 0)
    //   row1 = (L1.x, L1.y, L1.z, 0)
    //   row2 = (xyz.x, xyz.y, xyz.z, 1)
    let s2w_r0 = vec4<f32>(L0, 0.0);
    let s2w_r1 = vec4<f32>(L1, 0.0);
    let s2w_r2 = vec4<f32>(xyz, 1.0);

    // world2ndc = transpose(P * V) — matches CUDA's GLM reindexing convention.
    // In CUDA, projmatrix stores (P*V)^T from PyTorch. GLM mat4 constructor reindexes
    // this flat array into a column-major matrix whose columns are ROWS of P*V.
    // So CUDA's world2ndc columns[j] = (PV)_row_j, meaning world2ndc = (PV)^T as a transform.
    //
    // camera.proj includes VIEWPORT_Y_FLIP (diag(1,-1,1,1)) for WebGPU rendering.
    // Undo Y-flip by negating the y-component of each column (= negating row 1).
    var proj_raw = camera.proj;
    proj_raw[0].y = -proj_raw[0].y;
    proj_raw[1].y = -proj_raw[1].y;
    proj_raw[2].y = -proj_raw[2].y;
    proj_raw[3].y = -proj_raw[3].y;
    let M = proj_raw * camera.view;  // world2ndc = (VP)^T

    // Compute intermediate: I = transpose(s2w) * M  (3 rows × 4 cols, stored as 3 vec4 rows)
    // I[i][j] = dot(s2w_row_i, M_col_j). M[j] = column j of (PV)^T = row j of PV.
    let I0 = vec4<f32>(dot(s2w_r0, M[0]), dot(s2w_r0, M[1]), dot(s2w_r0, M[2]), dot(s2w_r0, M[3]));
    let I1 = vec4<f32>(dot(s2w_r1, M[0]), dot(s2w_r1, M[1]), dot(s2w_r1, M[2]), dot(s2w_r1, M[3]));
    let I2 = vec4<f32>(dot(s2w_r2, M[0]), dot(s2w_r2, M[1]), dot(s2w_r2, M[2]), dot(s2w_r2, M[3]));

    // ndc2pix columns (as vec4): c0 = (W/2, 0, 0, (W-1)/2), c1 = (0, H/2, 0, (H-1)/2), c2 = (0, 0, 0, 1)
    let np0 = vec4<f32>(W / 2.0, 0.0, 0.0, (W - 1.0) / 2.0);
    let np1 = vec4<f32>(0.0, H / 2.0, 0.0, (H - 1.0) / 2.0);
    let np2 = vec4<f32>(0.0, 0.0, 0.0, 1.0);

    // T = I * ndc2pix  (3×4 * 4×3 = 3×3)
    // T[row][col] = dot(I_row, ndc2pix_col)
    let T_mat = mat3x3<f32>(
        vec3<f32>(dot(I0, np0), dot(I1, np0), dot(I2, np0)),  // column 0 = Tu
        vec3<f32>(dot(I0, np1), dot(I1, np1), dot(I2, np1)),  // column 1 = Tv
        vec3<f32>(dot(I0, np2), dot(I1, np2), dot(I2, np2)),  // column 2 = Tw
    );

    // Compute cutoff radius based on AABB_MODE (injected at compile time)
    // AABB_MODE 0: Square AABB + fixed cutoff
    // AABB_MODE 2: Rectangular AABB + fixed cutoff
    // AABB_MODE 3: Rectangular AABB + AdR (opacity-adaptive) cutoff
    var cutoff: f32;
    if opacity < (1.0 / 255.0) {
        return;
    }

    // AABB_MODE: 0 = square + fixed cutoff, 1 = square + AdR, 2 = rect + fixed, 3 = rect + AdR.
    // Mirrors the CUDA compact-box logic: r_lp gets multiplied by d_compact_mult.
    let use_adr = (AABB_MODE == 1u || AABB_MODE == 3u);
    let compact_mult = render_settings.compact_mult;

    if use_adr {
        if KERNEL_TYPE == 4u {
            // BetaScaled kernel (k²=9, k=3): r_beta from kernel threshold, r_lp from low-pass.
            // NOTE: training's beta-AdR path (diff_surfel_3D_sh_res forward.cu ~L663)
            // leaves r_lp unmodified — compact_mult is only applied to the pure-Gaussian
            // AdR branch. Applying it here instead would make beta kernels render dimmer
            // than training.
            let k = 3.0;
            let ratio = 1.0 / (255.0 * opacity);
            var r_beta = 0.0;
            let threshold = pow(ratio, 1.0 / shape);
            if threshold < 1.0 {
                r_beta = k * sqrt(1.0 - threshold);
            }
            var r_lp = 0.0;
            let log_term = log(255.0 * opacity);
            if log_term > 0.0 {
                r_lp = sqrt(2.0 * log_term);
            }
            cutoff = max(r_beta, r_lp);
            cutoff = min(cutoff, k + 2.0);
        } else if KERNEL_TYPE == 1u {
            // Beta kernel (k²=1, k=1): same structure as BetaScaled, tighter k.
            // Same rule — no compact_mult on r_lp.
            let k = 1.0;
            let ratio = 1.0 / (255.0 * opacity);
            var r_beta = 0.0;
            let threshold = pow(ratio, 1.0 / shape);
            if threshold < 1.0 {
                r_beta = k * sqrt(1.0 - threshold);
            }
            var r_lp = 0.0;
            let log_term = log(255.0 * opacity);
            if log_term > 0.0 {
                r_lp = sqrt(2.0 * log_term);
            }
            cutoff = max(r_beta, r_lp);
            cutoff = min(cutoff, k + 2.0);
        } else {
            // Gaussian: solve opacity * exp(-r²/2) = 1/255.
            let log_term = log(255.0 * opacity);
            if log_term > 0.0 {
                cutoff = sqrt(2.0 * log_term * compact_mult);
            } else {
                cutoff = 0.1;
            }
            cutoff = min(cutoff, 4.0);
        }
    } else {
        // Mode 0 (square) / mode 2 (rect): fixed 4σ (2DGS default) for ANY kernel.
        // We do NOT have a WGSL equivalent of CUDA mode 4 (use_beta_fixed) —
        // if you wanted the tight beta-compact path, run under AdR (mode 1/3).
        // The previous "KERNEL_TYPE==4 → 3.3" fallback caused ~16% dimming vs training.
        cutoff = 4.0;
    }

    // Compute AABB from transmat
    let aabb = compute_aabb(T_mat, cutoff);
    let center_pix = aabb.xy;  // pixel-space center
    let extent_pix = aabb.zw;  // pixel-space extent

    if extent_pix.x < 0.0 {
        return; // invalid AABB
    }

    // Compute tile bounding box
    let use_rect = (AABB_MODE >= 2u);
    let filter_r = cutoff * FILTER_SIZE;
    var rect_min_x: u32;
    var rect_min_y: u32;
    var rect_max_x: u32;
    var rect_max_y: u32;

    if use_rect {
        // Rectangular AABB: separate X and Y radii
        let rx = ceil(max(extent_pix.x, filter_r));
        let ry = ceil(max(extent_pix.y, filter_r));
        rect_min_x = max(0u, u32(max(0.0, floor((center_pix.x - rx) / f32(TILE_SIZE)))));
        rect_min_y = max(0u, u32(max(0.0, floor((center_pix.y - ry) / f32(TILE_SIZE)))));
        rect_max_x = min(tile_info.tiles_x, u32(ceil((center_pix.x + rx) / f32(TILE_SIZE))));
        rect_max_y = min(tile_info.tiles_y, u32(ceil((center_pix.y + ry) / f32(TILE_SIZE))));
    } else {
        // Square AABB: max(x, y) as scalar radius
        let radius = ceil(max(max(extent_pix.x, extent_pix.y), filter_r));
        rect_min_x = max(0u, u32(max(0.0, floor((center_pix.x - radius) / f32(TILE_SIZE)))));
        rect_min_y = max(0u, u32(max(0.0, floor((center_pix.y - radius) / f32(TILE_SIZE)))));
        rect_max_x = min(tile_info.tiles_x, u32(ceil((center_pix.x + radius) / f32(TILE_SIZE))));
        rect_max_y = min(tile_info.tiles_y, u32(ceil((center_pix.y + radius) / f32(TILE_SIZE))));
    }

    let n_tiles = (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y);
    if n_tiles == 0u {
        return;
    }

    // SH color evaluation: clamp(SH_eval + sh_bias, 0). Mirrors CUDA computeColorFromSH.
    // SB view-dep lobes are added in unclamped per the CUDA ordering:
    //   feat = max(0, SH + sh_bias) + residual + SB + res_bias  (ReLU at the end).
    // Residual and res_bias are applied in tile_raster; SH + SB fold in here.
    let camera_pos = camera.view_inv[3].xyz;
    let view_dir = normalize(xyz - camera_pos);
    var color = max(vec3<f32>(0.0), evaluate_sh(view_dir, idx, render_settings.max_sh_deg, render_settings.sh_bias));
    color = color + eval_sb(idx, view_dir);

    // Store to output
    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);

    splats_2d[store_idx] = Splat2DGS(
        T_mat[0].x, T_mat[1].x, T_mat[2].x,
        T_mat[0].y, T_mat[1].y, T_mat[2].y,
        T_mat[0].z, T_mat[1].z, T_mat[2].z,
        opacity,
        pack2x16float(center_pix),             // pixel-space center
        0u,
        pack2x16float(vec2<f32>(color.r, color.g)),
        pack2x16float(vec2<f32>(color.b, shape)),
        idx,
        depth_u,
        depth_v,
        camspace.z,
        0u,
    );

    // Depth for sorting
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - 1.0);
    sort_depths[store_idx] = bitcast<u32>(zfar - pos2d.z);
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    // Tile binning
    tiles_touched[store_idx] = n_tiles;
    rect_data[store_idx] = vec4<u32>(rect_min_x, rect_min_y, rect_max_x, rect_max_y);

    // Quantize depth to 16 bits for packed tile|depth key
    let depth_norm = clamp(pos2d.z / zfar, 0.0, 1.0);
    depth_16[store_idx] = u32(depth_norm * 65535.0);
}
