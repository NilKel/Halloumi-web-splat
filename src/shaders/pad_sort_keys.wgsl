// Fill the padding region [keys_size, keys_size + keys_per_wg) of the sort
// keys buffer with 0xFFFFFFFF sentinels. Replaces a 4 MB-per-frame full-
// buffer reset with a 256-thread × 15-workgroup compute pass (~15 KB written).
//
// Why this works: dispatch_x for the radix sort is set inside preprocess
// via `if (store_idx % keys_per_wg) == 0u { atomicAdd(dispatch_x, 1) }`,
// giving dispatch_x = ceil(keys_size / keys_per_wg). The sort then reads
// dispatch_x * keys_per_wg slots = at most keys_size + keys_per_wg − 1
// slots. Slots past that range are never read by the sort and remain
// whatever stale data the previous frame left there — harmless.
//
// Must run AFTER preprocess (which writes keys_size into sort_infos) and
// BEFORE the radix sort itself.

struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

@group(0) @binding(0) var<storage, read> sort_infos: SortInfos;
@group(0) @binding(1) var<storage, read_write> sort_keys: array<u32>;

@compute @workgroup_size(256, 1, 1)
fn pad(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = sort_infos.keys_size + gid.x;
    if idx < arrayLength(&sort_keys) {
        sort_keys[idx] = 0xFFFFFFFFu;
    }
}
