// Duplicate keys: for each visible Gaussian, emit (tile_id << 16 | depth_16) key pairs
// into the tile_keys/tile_payloads arrays, one entry per tile the Gaussian overlaps.
//
// Input: tiles_touched[] (original counts), tile_offsets[] (inclusive prefix sum),
//        rect_data[] (tile AABBs), depth_16[] (quantized depth)
// Output: tile_keys[], tile_payloads[]

struct SortInfos {
    keys_size: u32,  // = num_visible (written by preprocess)
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct DuplicateInfo {
    num_visible: u32,  // unused (kept for uniform alignment), actual count from sort_infos
    tiles_x: u32,
    tile_size: u32,    // pixel size of one tile (8/16/32) for AccuTile pixel-coord math
    _pad1: u32,
}

// AccuTile per-Gaussian conic data. Mirrors EllipseConic in
// preprocess_tile_2dgs.wgsl and EllipseConicData in tile_raster.rs.
// abet = (A, B, E, t) of Q(d) = A·dx² + 2B·dx·dy + E·dy² ≤ t (centered at p).
// abet.w (= t) ≤ 0 → conic degenerate; fall back to rect-AABB enumeration.
struct EllipseConic {
    abet: vec4<f32>,
    p: vec2<f32>,
    _pad: vec2<f32>,
}

@group(0) @binding(0)
var<storage, read> tile_offsets: array<u32>;     // inclusive prefix sum of tiles_touched

@group(0) @binding(1)
var<storage, read> rect_data: array<vec4<u32>>;  // (rect_min.x, rect_min.y, rect_max.x, rect_max.y)

@group(0) @binding(2)
var<storage, read> depth_vals: array<u32>;       // quantized 16-bit depth per Gaussian

@group(0) @binding(3)
var<storage, read_write> tile_keys: array<u32>;  // output: (tile_id << 16) | depth_16

@group(0) @binding(4)
var<storage, read_write> tile_payloads: array<u32>; // output: splat index

@group(0) @binding(5)
var<uniform> info: DuplicateInfo;

@group(0) @binding(6)
var<storage, read> sort_infos: SortInfos;  // reads keys_size = actual num_visible

@group(0) @binding(7)
var<storage, read> conic_data: array<EllipseConic>;

// AccuTile: solve A·dx² + 2B·dx·dy + E·dy² = t for the orthogonal axis at
// fixed `coord` (pixel coord on the chosen axis). Returns the two roots —
// the ellipse boundary's min/max on that line. Exact port of
// auxiliary.h::computeEllipseIntersection (Y-axis branch only — we always
// scan over rows here, so isY = true).
//
// disc = B*B - A*E (must be < 0 for a real ellipse).
// h = coord - p.y;  radicand = disc·h² + t·A;  sqrt_term = sqrt(max(0, radicand)).
// roots = (-B·h ± sqrt_term) / A + p.x   ← x-extent at fixed y = coord.
fn ellipse_x_extent_at_y(abet: vec4<f32>, disc: f32, p: vec2<f32>, y_coord: f32) -> vec2<f32> {
    let A_ = abet.x;
    let B_ = abet.y;
    let t_ = abet.w;
    let h = y_coord - p.y;
    let radicand = disc * h * h + t_ * A_;
    let sqrt_term = sqrt(max(radicand, 0.0));
    let neg_Bh = -B_ * h;
    return vec2<f32>(
        (neg_Bh - sqrt_term) / A_ + p.x,
        (neg_Bh + sqrt_term) / A_ + p.x,
    );
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_visible = sort_infos.keys_size;
    if idx >= num_visible {
        return;
    }

    let rect = rect_data[idx];
    let rect_min_x = rect.x;
    let rect_min_y = rect.y;
    let rect_max_x = rect.z;
    let rect_max_y = rect.w;

    // depth_vals layout (set in preprocess_tile_2dgs.wgsl):
    //   bits  0..15: depth_quant_16 (normalized z * 65535)
    //   bits 16..31: idx_lsb_16     (low 16 bits of original splat_idx)
    // We pack into the 32-bit sort key as (tile_16, depth_10, idx_lsb_6).
    // 1024 depth buckets across [0, zfar] = ~0.05 unit precision at zfar=50,
    // fine enough for almost all overlapping splats to land in distinct buckets.
    // The 6-bit idx_lsb tiebreaker is deterministic when (tile, depth) ties.
    let dval = depth_vals[idx];
    let depth_quant_10 = (dval >> 6u) & 0x3FFu;
    let idx_lsb_6      = (dval >> 16u) & 0x3Fu;

    // Get write offset from prefix sum (inclusive → exclusive by subtracting original count)
    var write_offset: u32;
    if idx == 0u {
        write_offset = 0u;
    } else {
        write_offset = tile_offsets[idx - 1u];
    }

    // Pull conic. abet.w (= t) ≤ 0 means degenerate; just enumerate the
    // rect-AABB tiles like before. Otherwise AccuTile per-row prunes tiles
    // outside the actual ellipse and emits sentinel keys for the rest.
    let ec = conic_data[idx];
    let abet = ec.abet;
    let p = ec.p;
    let A_ = abet.x;
    let B_ = abet.y;
    let E_ = abet.z;
    let t_ = abet.w;
    let disc = B_ * B_ - A_ * E_;
    // Temporarily force OFF to verify tile-binning correctness without
    // ellipse-intersection pruning. Set back to the conjuction once the
    // raw rect-AABB path renders cleanly.
    let use_accutile = false;

    let ts_f = f32(info.tile_size);

    // Sentinel for culled / out-of-ellipse tiles. tile_id = 0xFFFF is past
    // any valid `tiles_x*tiles_y` for typical resolutions, so identify_ranges
    // never queries it. Sorts to the very end of the keys buffer.
    let SENTINEL: u32 = 0xFFFFFFFFu;

    var j = 0u;
    for (var ty = rect_min_y; ty < rect_max_y; ty = ty + 1u) {
        // Default: keep all tiles in this row (rect-AABB fallback).
        var tx_lo = rect_min_x;
        var tx_hi = rect_max_x;

        if use_accutile {
            // Compute the x-extent of the ellipse at this row's top and bottom
            // edges in pixel space, then take the union: (xmin, xmax) is the
            // full pixel-x range any point of the ellipse can occupy in this
            // row. Convert to tile indices.
            let y0 = f32(ty) * ts_f;
            let y1 = f32(ty + 1u) * ts_f;
            let xs0 = ellipse_x_extent_at_y(abet, disc, p, y0);
            let xs1 = ellipse_x_extent_at_y(abet, disc, p, y1);

            // Also widen to include the ellipse's argmax-y extreme if it
            // falls inside this row (the ellipse can bulge past the top/bottom
            // edge intersections in between).
            // y_term = sqrt(-B²·t / (disc·E)) — distance from p.y to the
            // y-extreme; sign per CUDA's `(B < 0) ? +y_term : -y_term`.
            let B2t = B_ * B_ * t_;
            let y_term_sq = -B2t / (disc * E_);
            let y_term = sqrt(max(y_term_sq, 0.0));
            // x-extreme y position (where ∂Q/∂y = 0 along ellipse boundary).
            let arg_y_lo = select(p.y + y_term, p.y - y_term, B_ < 0.0);
            let arg_y_hi = select(p.y - y_term, p.y + y_term, B_ < 0.0);

            var xmin = min(xs0.x, xs1.x);
            var xmax = max(xs0.y, xs1.y);
            if (arg_y_lo >= y0) && (arg_y_lo < y1) {
                let xs_arg = ellipse_x_extent_at_y(abet, disc, p, arg_y_lo);
                xmin = min(xmin, xs_arg.x);
            }
            if (arg_y_hi >= y0) && (arg_y_hi < y1) {
                let xs_arg = ellipse_x_extent_at_y(abet, disc, p, arg_y_hi);
                xmax = max(xmax, xs_arg.y);
            }

            if xmax >= xmin {
                let tx_lo_f = floor(xmin / ts_f);
                let tx_hi_f = ceil(xmax / ts_f);
                tx_lo = max(rect_min_x, u32(max(0.0, tx_lo_f)));
                tx_hi = min(rect_max_x, u32(max(0.0, tx_hi_f)));
            } else {
                // Ellipse doesn't intersect this row at all → emit sentinels.
                tx_lo = rect_min_x;
                tx_hi = rect_min_x;
            }
        }

        for (var tx = rect_min_x; tx < rect_max_x; tx = tx + 1u) {
            let off = write_offset + j;
            if (tx >= tx_lo) && (tx < tx_hi) {
                let tile_id = ty * info.tiles_x + tx;
                // 32-bit packed key: (tile_16 << 16) | (depth_10 << 6) | idx_lsb_6.
                // tile_id primary, depth secondary, splat_idx_lsb is the
                // deterministic tiebreak (eliminates flicker from atomicAdd
                // store_idx ordering).
                tile_keys[off] = (tile_id << 16u) | (depth_quant_10 << 6u) | idx_lsb_6;
                tile_payloads[off] = idx;
            } else {
                tile_keys[off] = SENTINEL;
                tile_payloads[off] = idx;  // payload unused for sentinels
            }
            j = j + 1u;
        }
    }
}
