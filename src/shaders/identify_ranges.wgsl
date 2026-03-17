// Identify tile ranges: scan sorted tile keys to find per-tile start/end indices.
// After radix sort, tile_keys is sorted by (tile_id << 16 | depth).
// This shader finds where each tile_id starts and ends in the sorted array.
//
// Uses two separate array<u32> buffers (tile_starts, tile_ends) instead of
// array<vec2<u32>> to avoid Metal MSL race conditions where concurrent writes
// to .x and .y of the same vec2<u32> can tear.
//
// Output: tile_starts[tile_id] = first index, tile_ends[tile_id] = past-end index

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0)
var<storage, read> sorted_keys: array<u32>;

@group(0) @binding(1)
var<storage, read_write> tile_starts: array<u32>;

@group(0) @binding(2)
var<storage, read_write> tile_ends: array<u32>;

@group(0) @binding(3)
var<storage, read> sort_infos: SortInfos;  // reads keys_size = actual total entries

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let num_entries = sort_infos.keys_size;
    let idx = (wid.x + wid.y * nwg.x) * 256u + lid.x;

    if idx >= num_entries {
        return;
    }

    let curr_tile = sorted_keys[idx] >> 16u;

    if idx == 0u {
        // First entry: this tile starts at 0
        tile_starts[curr_tile] = 0u;
    } else {
        let prev_tile = sorted_keys[idx - 1u] >> 16u;
        if curr_tile != prev_tile {
            tile_ends[prev_tile] = idx;
            tile_starts[curr_tile] = idx;
        }
    }

    if idx == num_entries - 1u {
        // Last entry: this tile ends at num_entries
        tile_ends[curr_tile] = num_entries;
    }
}
