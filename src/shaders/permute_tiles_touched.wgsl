// Splatshop step: permute tiles_touched into depth-sorted order, so the
// prefix-sum result indexes by sorted_pos (not store_idx). After this:
//   tiles_touched_perm[i] = tiles_touched[sort_indices[i]]
// where sort_indices[i] = store_idx of the i-th depth-sorted visible splat.
//
// duplicate_keys then iterates sorted_pos, reads tile_offsets_perm[idx-1]
// for write position, and emits tile keys in depth-sorted order. The tile
// sort that follows uses tile_id-only keys; stable sort preserves the
// depth ordering established here.

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0) var<storage, read> sort_indices: array<u32>;
@group(0) @binding(1) var<storage, read> tiles_touched: array<u32>;
@group(0) @binding(2) var<storage, read_write> tiles_touched_perm: array<u32>;
@group(0) @binding(3) var<storage, read> sort_infos: SortInfos;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = arrayLength(&tiles_touched_perm);
    if idx >= n {
        return;
    }
    let num_visible = sort_infos.keys_size;
    if idx < num_visible {
        let store_idx = sort_indices[idx];
        tiles_touched_perm[idx] = tiles_touched[store_idx];
    } else {
        tiles_touched_perm[idx] = 0u;
    }
}
