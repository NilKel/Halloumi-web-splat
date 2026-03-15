// Workgroup-level inclusive prefix sum with multi-level reduction.
// This implements a two-pass approach:
//   Pass 1 (reduce): Compute per-workgroup sums and workgroup-level prefix sums
//   Pass 2 (propagate): Add block offsets to get global prefix sums
//
// For simplicity, we use a single-dispatch approach that works for up to
// 256 * 256 = 65536 elements (enough for most scenes).
// For larger arrays, the Rust code chains multiple dispatches.

const WG_SIZE: u32 = 256u;

struct PrefixSumInfo {
    num_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0)
var<storage, read_write> data: array<u32>;

@group(0) @binding(1)
var<storage, read_write> block_sums: array<u32>;

@group(0) @binding(2)
var<uniform> info: PrefixSumInfo;

var<workgroup> shared_data: array<u32, 256>;

// Pass 1: Compute per-workgroup inclusive prefix sums and write block totals
@compute @workgroup_size(256, 1, 1)
fn reduce(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let global_idx = wid.x * WG_SIZE + lid.x;

    // Load data into shared memory
    if global_idx < info.num_elements {
        shared_data[lid.x] = data[global_idx];
    } else {
        shared_data[lid.x] = 0u;
    }
    workgroupBarrier();

    // Hillis-Steele inclusive scan (up-sweep)
    var offset = 1u;
    for (var d = WG_SIZE >> 1u; d > 0u; d >>= 1u) {
        if lid.x < d {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            shared_data[bi] += shared_data[ai];
        }
        offset <<= 1u;
        workgroupBarrier();
    }

    // Store block sum before down-sweep clears it
    if lid.x == 0u {
        block_sums[wid.x] = shared_data[WG_SIZE - 1u];
        shared_data[WG_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    // Down-sweep for exclusive scan
    for (var d = 1u; d < WG_SIZE; d <<= 1u) {
        offset >>= 1u;
        if lid.x < d {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            let tmp = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += tmp;
        }
        workgroupBarrier();
    }

    // Write exclusive scan result back; convert to inclusive by adding original value
    if global_idx < info.num_elements {
        let original = data[global_idx];
        data[global_idx] = shared_data[lid.x] + original; // inclusive prefix sum within block
    }
}

// Pass 2: Scan block sums (run on a single workgroup)
@compute @workgroup_size(256, 1, 1)
fn scan_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let num_blocks = (info.num_elements + WG_SIZE - 1u) / WG_SIZE;

    if lid.x < num_blocks {
        shared_data[lid.x] = block_sums[lid.x];
    } else {
        shared_data[lid.x] = 0u;
    }
    workgroupBarrier();

    // Blelloch scan on block sums
    var offset = 1u;
    for (var d = WG_SIZE >> 1u; d > 0u; d >>= 1u) {
        if lid.x < d {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            shared_data[bi] += shared_data[ai];
        }
        offset <<= 1u;
        workgroupBarrier();
    }

    if lid.x == 0u {
        shared_data[WG_SIZE - 1u] = 0u;
    }
    workgroupBarrier();

    for (var d = 1u; d < WG_SIZE; d <<= 1u) {
        offset >>= 1u;
        if lid.x < d {
            let ai = offset * (2u * lid.x + 1u) - 1u;
            let bi = offset * (2u * lid.x + 2u) - 1u;
            let tmp = shared_data[ai];
            shared_data[ai] = shared_data[bi];
            shared_data[bi] += tmp;
        }
        workgroupBarrier();
    }

    // Write exclusive prefix sum of block sums
    if lid.x < num_blocks {
        block_sums[lid.x] = shared_data[lid.x];
    }
}

// Pass 3: Add block offsets to each element to get global inclusive prefix sum
@compute @workgroup_size(256, 1, 1)
fn propagate(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let global_idx = wid.x * WG_SIZE + lid.x;
    if global_idx < info.num_elements && wid.x > 0u {
        data[global_idx] += block_sums[wid.x];
    }
}
