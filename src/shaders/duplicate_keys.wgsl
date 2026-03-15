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
    _pad0: u32,
    _pad1: u32,
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

    let d16 = depth_vals[idx] & 0xFFFFu; // ensure 16-bit

    // Get write offset from prefix sum (inclusive → exclusive by subtracting original count)
    var write_offset: u32;
    if idx == 0u {
        write_offset = 0u;
    } else {
        write_offset = tile_offsets[idx - 1u];
    }

    // Emit one key-payload pair per overlapping tile
    var j = 0u;
    for (var ty = rect_min_y; ty < rect_max_y; ty++) {
        for (var tx = rect_min_x; tx < rect_max_x; tx++) {
            let tile_id = ty * info.tiles_x + tx;
            tile_keys[write_offset + j] = (tile_id << 16u) | d16;
            tile_payloads[write_offset + j] = idx;
            j++;
        }
    }
}
