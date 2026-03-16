// Fullscreen copy: reads packed RGBA8 u32 values from a storage buffer
// and outputs them to the render target via a fullscreen triangle strip quad.

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
};

struct ViewportInfo {
    width: u32,
    height: u32,
}

@group(0) @binding(0)
var<storage, read> pixels: array<u32>;

@group(0) @binding(1)
var<uniform> viewport: ViewportInfo;

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    // Triangle strip: 4 vertices covering the full screen
    let x = f32(idx % 2u) * 2.0 - 1.0;
    let y = f32(idx / 2u) * 2.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

@fragment
fn fs_main(@builtin(position) pos: vec4<f32>) -> @location(0) vec4<f32> {
    // DEBUG: test if we can read from the storage buffer at all
    let val = pixels[0];
    if val > 0u {
        return vec4<f32>(1.0, 0.0, 0.0, 1.0); // RED = buffer has data
    } else {
        return vec4<f32>(0.0, 0.0, 1.0, 1.0); // BLUE = buffer is zero
    }
}
