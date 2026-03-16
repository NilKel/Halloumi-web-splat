// Tile-based preprocess: same as standard preprocess + tile AABB + tiles_touched count
// This shader extends the standard preprocess to compute per-Gaussian tile overlap info
// for the compute rasterizer's tile binning pipeline.

const CUTOFF: f32 = 2.3539888583335364; // sqrt(log(255))
//const TILE_SIZE: u32 = <injected>u;

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

struct Gaussian {
    x: f32, y: f32, z: f32,
    opacity: u32,
    cov: array<u32, 3>
};

struct Splat {
    v_0: u32, v_1: u32,
    pos: u32,
    color_0: u32, color_1: u32,
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
    center: vec3<f32>,
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

// Standard preprocess bindings (same as preprocess.wgsl)
@group(0) @binding(0)
var<uniform> camera: CameraUniforms;

@group(1) @binding(0)
var<storage, read> gaussians: array<Gaussian>;
@group(1) @binding(1)
var<storage, read> sh_coefs: array<array<u32, 24>>;
@group(1) @binding(2)
var<storage, read_write> points_2d: array<Splat>;

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

@group(3) @binding(0)
var<uniform> render_settings: RenderSettings;

// Additional tile binning outputs
@group(3) @binding(1)
var<storage, read_write> tiles_touched: array<u32>;
@group(3) @binding(2)
var<storage, read_write> rect_data: array<vec4<u32>>; // (rect_min.x, rect_min.y, rect_max.x, rect_max.y)
@group(3) @binding(3)
var<storage, read_write> depth_16: array<u32>; // quantized 16-bit depth per visible Gaussian
@group(3) @binding(4)
var<uniform> tile_info: TileInfo;

fn sh_coef(splat_idx: u32, c_idx: u32) -> vec3<f32> {
    let a = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 0u) / 2u])[(c_idx * 3u + 0u) % 2u];
    let b = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 1u) / 2u])[(c_idx * 3u + 1u) % 2u];
    let c = unpack2x16float(sh_coefs[splat_idx][(c_idx * 3u + 2u) / 2u])[(c_idx * 3u + 2u) % 2u];
    return vec3<f32>(a, b, c);
}

fn evaluate_sh(dir: vec3<f32>, v_idx: u32, sh_deg: u32) -> vec3<f32> {
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
    result += 0.5;
    return result;
}

fn cov_coefs(v_idx: u32) -> array<f32, 6> {
    let a = unpack2x16float(gaussians[v_idx].cov[0]);
    let b = unpack2x16float(gaussians[v_idx].cov[1]);
    let c = unpack2x16float(gaussians[v_idx].cov[2]);
    return array<f32, 6>(a.x, a.y, b.x, b.y, c.x, c.y);
}

@compute @workgroup_size(256, 1, 1)
fn preprocess(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= arrayLength(&gaussians) {
        return;
    }

    let focal = camera.focal;
    let viewport = camera.viewport;
    let vertex = gaussians[idx];
    let a = unpack2x16float(vertex.opacity);
    let xyz = vec3<f32>(vertex.x, vertex.y, vertex.z);
    var opacity = a.x;

    if any(xyz < render_settings.clipping_box_min.xyz) || any(xyz > render_settings.clipping_box_max.xyz) {
        return;
    }

    var camspace = camera.view * vec4<f32>(xyz, 1.);
    let pos2d = camera.proj * camspace;
    let bounds = 1.2 * pos2d.w;
    let z = pos2d.z / pos2d.w;

    if idx == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    if z <= 0. || z >= 1. || pos2d.x < -bounds || pos2d.x > bounds || pos2d.y < -bounds || pos2d.y > bounds {
        return;
    }

    let cov_sparse = cov_coefs(idx);

    let walltime = render_settings.walltime;
    var scale_mod = 0.;
    let dd = 5. * distance(render_settings.center, xyz) / render_settings.scene_extend;
    if walltime > dd {
        scale_mod = smoothstep(0., 1., (walltime - dd));
    }

    let scaling = render_settings.gaussian_scaling * scale_mod;
    let Vrk = mat3x3<f32>(
        cov_sparse[0], cov_sparse[1], cov_sparse[2],
        cov_sparse[1], cov_sparse[3], cov_sparse[4],
        cov_sparse[2], cov_sparse[4], cov_sparse[5]
    ) * scaling * scaling;
    let J = mat3x3<f32>(
        focal.x / camspace.z, 0., -(focal.x * camspace.x) / (camspace.z * camspace.z),
        0., -focal.y / camspace.z, (focal.y * camspace.y) / (camspace.z * camspace.z),
        0., 0., 0.
    );

    let W = transpose(mat3x3<f32>(camera.view[0].xyz, camera.view[1].xyz, camera.view[2].xyz));
    let T = W * J;
    let cov2d = transpose(T) * Vrk * T;

    let kernel_size = render_settings.kernel_size;
    if bool(render_settings.mip_spatting) {
        let det_0 = max(1e-6, cov2d[0][0] * cov2d[1][1] - cov2d[0][1] * cov2d[0][1]);
        let det_1 = max(1e-6, (cov2d[0][0] + kernel_size) * (cov2d[1][1] + kernel_size) - cov2d[0][1] * cov2d[0][1]);
        var coef = sqrt(det_0 / (det_1 + 1e-6) + 1e-6);
        if det_0 <= 1e-6 || det_1 <= 1e-6 {
            coef = 0.0;
        }
        opacity *= coef;
    }

    let diagonal1 = cov2d[0][0] + kernel_size;
    let offDiagonal = cov2d[0][1];
    let diagonal2 = cov2d[1][1] + kernel_size;

    let mid = 0.5 * (diagonal1 + diagonal2);
    let eigrad = length(vec2<f32>((diagonal1 - diagonal2) / 2.0, offDiagonal));
    let lambda1 = mid + eigrad;
    let lambda2 = max(mid - eigrad, 0.1);

    let diagonalVector = normalize(vec2<f32>(offDiagonal, lambda1 - diagonal1));
    let v1 = sqrt(2.0 * lambda1) * diagonalVector;
    let v2 = sqrt(2.0 * lambda2) * vec2<f32>(diagonalVector.y, -diagonalVector.x);

    let v_center = pos2d.xyzw / pos2d.w;

    let camera_pos = camera.view_inv[3].xyz;
    let dir = normalize(xyz - camera_pos);
    let color = vec4<f32>(
        max(vec3<f32>(0.), evaluate_sh(dir, idx, render_settings.max_sh_deg)),
        opacity
    );

    let store_idx = atomicAdd(&sort_infos.keys_size, 1u);
    let v = vec4<f32>(v1 / viewport, v2 / viewport);
    points_2d[store_idx] = Splat(
        pack2x16float(v.xy), pack2x16float(v.zw),
        pack2x16float(v_center.xy),
        pack2x16float(color.rg), pack2x16float(color.ba),
    );

    // Depth for sorting (same as standard preprocess)
    let znear = -camera.proj[3][2] / camera.proj[2][2];
    let zfar = -camera.proj[3][2] / (camera.proj[2][2] - 1.);
    sort_depths[store_idx] = bitcast<u32>(zfar - pos2d.z);
    sort_indices[store_idx] = store_idx;

    let keys_per_wg = 256u * 15u;
    if (store_idx % keys_per_wg) == 0u {
        atomicAdd(&sort_dispatch.dispatch_x, 1u);
    }

    // === TILE BINNING: compute tile AABB ===
    // v1, v2 are screen-space eigenvectors (in pixels)
    // The splat covers CUTOFF * eigenvector radius in each direction
    let radius_px = CUTOFF * vec2<f32>(length(v1), length(v2));
    // Conservative: use max radius for both axes to get a bounding circle
    let max_radius_px = max(radius_px.x, radius_px.y);

    // NDC center to pixel center
    let center_pix = (v_center.xy + 1.0) * 0.5 * viewport;

    // Tile bounding box (in tile coordinates)
    let rect_min_x = max(0u, u32(max(0.0, floor((center_pix.x - max_radius_px) / f32(TILE_SIZE)))));
    let rect_min_y = max(0u, u32(max(0.0, floor((center_pix.y - max_radius_px) / f32(TILE_SIZE)))));
    let rect_max_x = min(tile_info.tiles_x, u32(ceil((center_pix.x + max_radius_px) / f32(TILE_SIZE))));
    let rect_max_y = min(tile_info.tiles_y, u32(ceil((center_pix.y + max_radius_px) / f32(TILE_SIZE))));

    let n_tiles = (rect_max_x - rect_min_x) * (rect_max_y - rect_min_y);
    tiles_touched[store_idx] = n_tiles;
    rect_data[store_idx] = vec4<u32>(rect_min_x, rect_min_y, rect_max_x, rect_max_y);

    // Quantize depth to 16 bits for packed tile|depth key
    let depth_norm = clamp(pos2d.z / zfar, 0.0, 1.0);
    depth_16[store_idx] = u32(depth_norm * 65535.0);
}
