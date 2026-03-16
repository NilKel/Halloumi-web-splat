// Identify tile ranges: scan sorted tile keys to find per-tile start/end indices.
// After radix sort, tile_keys is sorted by (tile_id << 16 | depth).
// This shader finds where each tile_id starts and ends in the sorted array.
//
// Uses a single-threaded serial scan to avoid race conditions on Metal
// where concurrent writes to .x and .y of the same vec2<u32> can tear.
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

@compute @workgroup_size(1, 1, 1)
fn main() {
    let num_entries = sort_infos.keys_size;
    if num_entries == 0u {
        return;
    }

    var prev_tile = sorted_keys[0] >> 16u;
    tile_ranges[prev_tile] = vec2<u32>(0u, 0u);

    for (var i = 1u; i < num_entries; i++) {
        let curr_tile = sorted_keys[i] >> 16u;
        if curr_tile != prev_tile {
            tile_ranges[prev_tile].y = i;
            tile_ranges[curr_tile].x = i;
            prev_tile = curr_tile;
        }
    }

    tile_ranges[prev_tile].y = num_entries;
}
