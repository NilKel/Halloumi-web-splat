// Identify tile ranges: scan sorted tile keys to find per-tile start/end indices.
// After radix sort, tile_keys is sorted by (tile_id << 16 | depth).
// This shader finds where each tile_id starts and ends in the sorted array.
//
// Output: tile_ranges[tile_id] = vec2<u32>(start, end)

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0)
var<storage, read> sorted_keys: array<u32>;

@group(0) @binding(1)
var<storage, read_write> tile_ranges: array<vec2<u32>>;  // (start, end) per tile

@group(0) @binding(2)
var<storage, read> sort_infos: SortInfos;  // reads keys_size = actual total entries

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let num_entries = sort_infos.keys_size;
    if idx >= num_entries || num_entries == 0u {
        return;
    }

    let curr_tile = sorted_keys[idx] >> 16u;

    if idx == 0u {
        // First element starts its tile
        tile_ranges[curr_tile].x = 0u;
    } else {
        let prev_tile = sorted_keys[idx - 1u] >> 16u;
        if curr_tile != prev_tile {
            // Tile boundary: end the previous tile, start the new one
            tile_ranges[prev_tile].y = idx;
            tile_ranges[curr_tile].x = idx;
        }
    }

    // Last element ends its tile
    if idx == num_entries - 1u {
        tile_ranges[curr_tile].y = num_entries;
    }
}
