use crate::gpu_rs::GPURSSorter;
use crate::pointcloud::PointCloud;
use crate::renderer::{CameraUniform, SplattingArgs, SplattingArgsUniform};
use crate::uniform::UniformBuffer;

use crate::camera::Camera;

/// Uniform for tile info (used by preprocess_tile and tile_raster)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TileInfoUniform {
    tiles_x: u32,
    tiles_y: u32,
    total_tiles: u32,
    _pad: u32,
}

/// Uniform for tile raster info
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TileRasterInfoUniform {
    tiles_x: u32,
    tiles_y: u32,
    viewport_w: u32,
    viewport_h: u32,
    bg_r: f32,
    bg_g: f32,
    bg_b: f32,
    _pad: u32,
}

/// Uniform for prefix sum
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct PrefixSumInfoUniform {
    num_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Uniform for duplicate keys
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DuplicateInfoUniform {
    num_visible: u32,
    tiles_x: u32,
    _pad0: u32,
    _pad1: u32,
}

/// Uniform for viewport info (fullscreen copy)
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct ViewportInfoUniform {
    width: u32,
    height: u32,
}

const TILE_SIZE: u32 = 16;
const MAX_TILES_PER_GAUSSIAN: u32 = 16;

pub struct TileRasterPipeline {
    // Compute pipelines
    prefix_sum_reduce: wgpu::ComputePipeline,
    prefix_sum_scan_blocks: wgpu::ComputePipeline,
    prefix_sum_propagate: wgpu::ComputePipeline,
    duplicate_keys_pipeline: wgpu::ComputePipeline,
    identify_ranges_pipeline: wgpu::ComputePipeline,
    tile_raster_pipeline: wgpu::ComputePipeline,
    update_sort_info_pipeline: wgpu::ComputePipeline,
    preprocess_tile_pipeline: wgpu::ComputePipeline,

    // Render pipeline for fullscreen copy
    fullscreen_copy_pipeline: wgpu::RenderPipeline,

    // Bind group layouts (needed for recreation on resize)
    prefix_sum_bgl: wgpu::BindGroupLayout,
    duplicate_keys_bgl: wgpu::BindGroupLayout,
    identify_ranges_bgl: wgpu::BindGroupLayout,
    tile_raster_bgl: wgpu::BindGroupLayout,
    fullscreen_copy_bgl: wgpu::BindGroupLayout,
    update_sort_info_bgl: wgpu::BindGroupLayout,
    preprocess_tile_bg3_layout: wgpu::BindGroupLayout,

    // Uniforms
    tile_info: UniformBuffer<TileInfoUniform>,
    tile_raster_info: UniformBuffer<TileRasterInfoUniform>,
    prefix_sum_info: UniformBuffer<PrefixSumInfoUniform>,
    duplicate_info: UniformBuffer<DuplicateInfoUniform>,
    viewport_info: UniformBuffer<ViewportInfoUniform>,

    // Camera + render settings (shared with hardware path)
    camera: UniformBuffer<CameraUniform>,
    render_settings: UniformBuffer<SplattingArgsUniform>,

    // Preprocess tile bind group (group 3)
    preprocess_tile_bg3: wgpu::BindGroup,

    // Buffers
    tiles_touched_buf: wgpu::Buffer,
    rect_data_buf: wgpu::Buffer,
    depth_16_buf: wgpu::Buffer,
    prefix_sum_block_sums: wgpu::Buffer,
    tile_starts_buf: wgpu::Buffer,
    tile_ends_buf: wgpu::Buffer,
    output_buf: wgpu::Buffer,

    // Tile sort buffers
    tile_keys_a: wgpu::Buffer,
    tile_keys_b: wgpu::Buffer,
    tile_payloads_a: wgpu::Buffer,
    tile_payloads_b: wgpu::Buffer,
    tile_sort_internal: wgpu::Buffer,
    tile_sort_uni: wgpu::Buffer,
    tile_sort_dis: wgpu::Buffer,

    // Bind groups
    prefix_sum_bg: wgpu::BindGroup,
    duplicate_keys_bg: wgpu::BindGroup,
    identify_ranges_bg: wgpu::BindGroup,
    tile_raster_bg: wgpu::BindGroup,
    fullscreen_copy_bg: wgpu::BindGroup,
    update_sort_info_bg: wgpu::BindGroup,
    tile_sort_bg: wgpu::BindGroup,

    // Preprocess pipeline bind groups (groups 0-2 shared with hardware path)
    preprocess_sort_bg: wgpu::BindGroup,
    preprocess_sort_uni: wgpu::Buffer,
    preprocess_sort_dis: wgpu::Buffer,

    // Cached state
    cached_width: u32,
    cached_height: u32,
    cached_num_points: u32,
    max_tile_entries: u32,
    kernel_type: u32,
    use_shared_mem: bool,
}

impl TileRasterPipeline {
    pub fn new(
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        sorter: &GPURSSorter,
        num_points: u32,
        width: u32,
        height: u32,
        is_2dgs: bool,
        kernel_type: u32,
        use_shared_mem: bool,
    ) -> Self {
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        let total_tiles = tiles_x * tiles_y;
        let max_tile_entries = num_points * MAX_TILES_PER_GAUSSIAN;

        // ========== Create uniforms ==========
        let tile_info = UniformBuffer::new(
            device,
            TileInfoUniform {
                tiles_x,
                tiles_y,
                total_tiles,
                _pad: 0,
            },
            Some("tile info uniform"),
        );

        let tile_raster_info = UniformBuffer::new(
            device,
            TileRasterInfoUniform {
                tiles_x,
                tiles_y,
                viewport_w: width,
                viewport_h: height,
                bg_r: 0.0,
                bg_g: 0.0,
                bg_b: 0.0,
                _pad: 0,
            },
            Some("tile raster info uniform"),
        );

        let prefix_sum_info = UniformBuffer::new(
            device,
            PrefixSumInfoUniform {
                num_elements: num_points,
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            },
            Some("prefix sum info uniform"),
        );

        let duplicate_info = UniformBuffer::new(
            device,
            DuplicateInfoUniform {
                num_visible: num_points,
                tiles_x,
                _pad0: 0,
                _pad1: 0,
            },
            Some("duplicate info uniform"),
        );

        let viewport_info = UniformBuffer::new(
            device,
            ViewportInfoUniform { width, height },
            Some("viewport info uniform"),
        );

        let camera = UniformBuffer::new_default(device, Some("tile raster camera uniform"));
        let render_settings =
            UniformBuffer::new_default(device, Some("tile raster render settings uniform"));

        // ========== Create buffers ==========
        let tiles_touched_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tiles_touched"),
            size: (num_points as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let rect_data_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rect_data"),
            size: (num_points as u64) * 16, // vec4<u32> = 16 bytes
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let depth_16_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("depth_16"),
            size: (num_points as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let num_prefix_blocks = (num_points + 255) / 256;
        let prefix_sum_block_sums = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prefix_sum_block_sums"),
            size: (num_prefix_blocks.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let tile_starts_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_starts"),
            size: (total_tiles.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let tile_ends_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_ends"),
            size: (total_tiles.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let output_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tile_raster_output"),
            size: (width.max(1) * height.max(1)) as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Tile sort buffers — must use create_keyval_buffers for proper padding
        // (radix sort requires buffers padded to count_ru_histo size)
        let (tile_keys_a, tile_keys_b, tile_payloads_a, tile_payloads_b) =
            GPURSSorter::create_keyval_buffers(device, max_tile_entries as usize, 4);

        let tile_sort_internal =
            sorter.create_internal_mem_buffer(device, max_tile_entries as usize);
        let (tile_sort_uni, tile_sort_dis, tile_sort_bg) = sorter.create_bind_group(
            device,
            max_tile_entries as usize,
            &tile_sort_internal,
            &tile_keys_a,
            &tile_keys_b,
            &tile_payloads_a,
            &tile_payloads_b,
        );

        // Preprocess sort buffers (for the preprocess to write sort_infos, depths, indices)
        let preprocess_sort_uni = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess_tile sort_infos"),
            size: 16, // SortInfos: 4 u32s
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let preprocess_sort_dis = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess_tile sort_dispatch"),
            size: 12, // DispatchIndirect: 3 u32s
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::INDIRECT,
            mapped_at_creation: false,
        });
        let preprocess_sort_depths = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess_tile sort_depths"),
            size: (num_points.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let preprocess_sort_indices = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("preprocess_tile sort_indices"),
            size: (num_points.max(1) as u64) * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Dummy draw indirect buffer for preprocess bind group
        let dummy_draw_indirect = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy draw indirect for tile preprocess"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Preprocess sort bind group (group 2) - matches the preprocess_tile.wgsl layout
        let preprocess_sort_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("preprocess tile sort bg layout"),
                entries: &[
                    storage_rw_entry(0), // sort_infos
                    storage_rw_entry(1), // sort_depths
                    storage_rw_entry(2), // sort_indices
                    storage_rw_entry(3), // sort_dispatch
                    storage_rw_entry(4), // draw_indirect
                ],
            });
        let preprocess_sort_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preprocess tile sort bg"),
            layout: &preprocess_sort_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: preprocess_sort_uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preprocess_sort_depths.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: preprocess_sort_indices.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: preprocess_sort_dis.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: dummy_draw_indirect.as_entire_binding(),
                },
            ],
        });

        // ========== Bind group layouts ==========

        // Preprocess tile group 3: render_settings(uniform) + tiles_touched(rw) + rect_data(rw) + depth_16(rw) + tile_info(uniform)
        let preprocess_tile_bg3_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("preprocess tile bg3 layout"),
                entries: &[
                    uniform_entry(0), // render_settings
                    storage_rw_entry(1), // tiles_touched
                    storage_rw_entry(2), // rect_data
                    storage_rw_entry(3), // depth_16
                    uniform_entry(4), // tile_info
                ],
            });

        let preprocess_tile_bg3 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("preprocess tile bg3"),
            layout: &preprocess_tile_bg3_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: render_settings.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tiles_touched_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rect_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: depth_16_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tile_info.buffer().as_entire_binding(),
                },
            ],
        });

        // Prefix sum: data(rw) + block_sums(rw) + info(uniform)
        let prefix_sum_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("prefix sum bg layout"),
            entries: &[
                storage_rw_entry(0),
                storage_rw_entry(1),
                uniform_entry(2),
            ],
        });

        let prefix_sum_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("prefix sum bg"),
            layout: &prefix_sum_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tiles_touched_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: prefix_sum_block_sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: prefix_sum_info.buffer().as_entire_binding(),
                },
            ],
        });

        // Duplicate keys: tile_offsets(read) + rect_data(read) + depth_vals(read) + tile_keys(rw) + tile_payloads(rw) + info(uniform) + sort_infos(read)
        let duplicate_keys_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("duplicate keys bg layout"),
                entries: &[
                    storage_ro_entry(0),
                    storage_ro_entry(1),
                    storage_ro_entry(2),
                    storage_rw_entry(3),
                    storage_rw_entry(4),
                    uniform_entry(5),
                    storage_ro_entry(6), // sort_infos (num_visible)
                ],
            });

        let duplicate_keys_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("duplicate keys bg"),
            layout: &duplicate_keys_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tiles_touched_buf.as_entire_binding(), // after prefix sum = tile_offsets
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rect_data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: depth_16_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_keys_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tile_payloads_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: duplicate_info.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: preprocess_sort_uni.as_entire_binding(), // sort_infos with keys_size
                },
            ],
        });

        // Identify ranges: sorted_keys(read) + tile_starts(rw) + tile_ends(rw) + sort_infos(read)
        let identify_ranges_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("identify ranges bg layout"),
                entries: &[
                    storage_ro_entry(0),  // sorted_keys
                    storage_rw_entry(1),  // tile_starts
                    storage_rw_entry(2),  // tile_ends
                    storage_ro_entry(3),  // sort_infos
                ],
            });

        let identify_ranges_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("identify ranges bg"),
            layout: &identify_ranges_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tile_keys_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tile_starts_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_ends_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_sort_uni.as_entire_binding(),
                },
            ],
        });

        // Tile raster: camera(uniform) + splats(read) + tile_payloads(read) + tile_starts(read) + output_buf(rw) + tile_info(uniform) + tile_ends(read)
        let tile_raster_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("tile raster bg layout"),
            entries: &[
                uniform_entry(0),    // camera
                storage_ro_entry(1), // splats
                storage_ro_entry(2), // tile_payloads
                storage_ro_entry(3), // tile_starts
                storage_rw_entry(4), // output_buf
                uniform_entry(5),    // tile_raster_info
                storage_ro_entry(6), // tile_ends
            ],
        });

        // Tile raster bind group will be created in prepare() since it needs the splat_2d buffer

        // Fullscreen copy: pixels(read) + viewport(uniform)
        let fullscreen_copy_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fullscreen copy bg layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let fullscreen_copy_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fullscreen copy bg"),
            layout: &fullscreen_copy_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: viewport_info.buffer().as_entire_binding(),
                },
            ],
        });

        // Update sort info: tile_offsets(read) + preprocess_sort_uni(read) + tile_sort_uni(rw) + tile_sort_dis(rw)
        let update_sort_info_bgl =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("update sort info bg layout"),
                entries: &[
                    storage_ro_entry(0), // tile_offsets (= tiles_touched after prefix sum)
                    storage_ro_entry(1), // preprocess sort_infos (has num_visible)
                    storage_rw_entry(2), // tile sort uniform
                    storage_rw_entry(3), // tile sort dispatch
                ],
            });

        let update_sort_info_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("update sort info bg"),
            layout: &update_sort_info_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tiles_touched_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: preprocess_sort_uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_sort_uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_sort_dis.as_entire_binding(),
                },
            ],
        });

        // ========== Create pipelines ==========

        // Preprocess tile pipeline (select 3DGS or 2DGS shader)
        let preprocess_tile_shader_src = if is_2dgs {
            format!(
                "const MAX_SH_DEG:u32 = {}u;\nconst KERNEL_TYPE: u32 = {}u;\n{}",
                sh_deg,
                kernel_type,
                include_str!("shaders/preprocess_tile_2dgs.wgsl")
            )
        } else {
            format!(
                "const MAX_SH_DEG:u32 = {}u;\n{}",
                sh_deg,
                include_str!("shaders/preprocess_tile.wgsl")
            )
        };
        let preprocess_tile_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("preprocess tile shader"),
                source: wgpu::ShaderSource::Wgsl(preprocess_tile_shader_src.into()),
            });
        let preprocess_tile_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("preprocess tile pipeline layout"),
                bind_group_layouts: &[
                    &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                    &PointCloud::bind_group_layout(device),
                    &preprocess_sort_bgl,
                    &preprocess_tile_bg3_layout,
                ],
                push_constant_ranges: &[],
            });
        let preprocess_tile_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("preprocess tile pipeline"),
                layout: Some(&preprocess_tile_layout),
                module: &preprocess_tile_shader,
                entry_point: Some("preprocess"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Prefix sum pipelines
        let prefix_sum_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("prefix sum shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/prefix_sum.wgsl").into(),
            ),
        });
        let prefix_sum_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("prefix sum pipeline layout"),
            bind_group_layouts: &[&prefix_sum_bgl],
            push_constant_ranges: &[],
        });
        let prefix_sum_reduce =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix sum reduce"),
                layout: Some(&prefix_sum_layout),
                module: &prefix_sum_shader,
                entry_point: Some("reduce"),
                compilation_options: Default::default(),
                cache: None,
            });
        let prefix_sum_scan_blocks =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix sum scan blocks"),
                layout: Some(&prefix_sum_layout),
                module: &prefix_sum_shader,
                entry_point: Some("scan_blocks"),
                compilation_options: Default::default(),
                cache: None,
            });
        let prefix_sum_propagate =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("prefix sum propagate"),
                layout: Some(&prefix_sum_layout),
                module: &prefix_sum_shader,
                entry_point: Some("propagate"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Duplicate keys pipeline
        let duplicate_keys_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("duplicate keys shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/duplicate_keys.wgsl").into(),
            ),
        });
        let duplicate_keys_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("duplicate keys pipeline layout"),
                bind_group_layouts: &[&duplicate_keys_bgl],
                push_constant_ranges: &[],
            });
        let duplicate_keys_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("duplicate keys pipeline"),
                layout: Some(&duplicate_keys_layout),
                module: &duplicate_keys_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Identify ranges pipeline
        let identify_ranges_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("identify ranges shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/identify_ranges.wgsl").into(),
            ),
        });
        let identify_ranges_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("identify ranges pipeline layout"),
                bind_group_layouts: &[&identify_ranges_bgl],
                push_constant_ranges: &[],
            });
        let identify_ranges_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("identify ranges pipeline"),
                layout: Some(&identify_ranges_layout),
                module: &identify_ranges_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Tile raster pipeline (select 3DGS or 2DGS shader, inject kernel type + shared mem flag)
        let use_shmem_val = if use_shared_mem { 1u32 } else { 0u32 };
        let tile_raster_shader_src = if is_2dgs {
            format!(
                "const KERNEL_TYPE: u32 = {}u;\nconst USE_SHARED_MEM: u32 = {}u;\n{}",
                kernel_type,
                use_shmem_val,
                include_str!("shaders/tile_raster_2dgs.wgsl")
            )
        } else {
            include_str!("shaders/tile_raster.wgsl").to_string()
        };
        log::info!("Tile raster kernel_type = {} ({}), shared_mem = {}",
            kernel_type,
            match kernel_type { 0 => "Gaussian", 1 => "Beta", 4 => "BetaScaled", _ => "Unknown" },
            use_shared_mem);
        let tile_raster_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("tile raster shader"),
            source: wgpu::ShaderSource::Wgsl(tile_raster_shader_src.into()),
        });
        let tile_raster_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("tile raster pipeline layout"),
            bind_group_layouts: &[&tile_raster_bgl],
            push_constant_ranges: &[],
        });
        let tile_raster_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("tile raster pipeline"),
                layout: Some(&tile_raster_layout),
                module: &tile_raster_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Update sort info pipeline (inline shader)
        let update_sort_info_shader =
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("update sort info shader"),
                source: wgpu::ShaderSource::Wgsl(UPDATE_SORT_INFO_SHADER.into()),
            });
        let update_sort_info_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("update sort info pipeline layout"),
                bind_group_layouts: &[&update_sort_info_bgl],
                push_constant_ranges: &[],
            });
        let update_sort_info_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("update sort info pipeline"),
                layout: Some(&update_sort_info_layout),
                module: &update_sort_info_shader,
                entry_point: Some("main"),
                compilation_options: Default::default(),
                cache: None,
            });

        // Fullscreen copy pipeline (render)
        let fullscreen_copy_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fullscreen copy shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("shaders/fullscreen_copy.wgsl").into(),
            ),
        });
        let fullscreen_copy_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("fullscreen copy pipeline layout"),
                bind_group_layouts: &[&fullscreen_copy_bgl],
                push_constant_ranges: &[],
            });
        let fullscreen_copy_pipeline =
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("fullscreen copy pipeline"),
                layout: Some(&fullscreen_copy_layout),
                vertex: wgpu::VertexState {
                    module: &fullscreen_copy_shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &fullscreen_copy_shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    ..Default::default()
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            });

        // Placeholder tile raster bind group (needs splat_2d_buffer from PointCloud)
        let tile_raster_bg = Self::create_tile_raster_bg(
            device,
            &tile_raster_bgl,
            &camera,
            &output_buf,
            &tile_payloads_a,
            &tile_starts_buf,
            &tile_ends_buf,
            &tile_raster_info,
            None,
        );

        Self {
            preprocess_tile_pipeline,
            prefix_sum_reduce,
            prefix_sum_scan_blocks,
            prefix_sum_propagate,
            duplicate_keys_pipeline,
            identify_ranges_pipeline,
            tile_raster_pipeline,
            update_sort_info_pipeline,
            fullscreen_copy_pipeline,

            prefix_sum_bgl,
            duplicate_keys_bgl,
            identify_ranges_bgl,
            tile_raster_bgl,
            fullscreen_copy_bgl,
            update_sort_info_bgl,
            preprocess_tile_bg3_layout,

            tile_info,
            tile_raster_info,
            prefix_sum_info,
            duplicate_info,
            viewport_info,
            camera,
            render_settings,

            preprocess_tile_bg3,

            tiles_touched_buf,
            rect_data_buf,
            depth_16_buf,
            prefix_sum_block_sums,
            tile_starts_buf,
            tile_ends_buf,
            output_buf,

            tile_keys_a,
            tile_keys_b,
            tile_payloads_a,
            tile_payloads_b,
            tile_sort_internal,
            tile_sort_uni,
            tile_sort_dis,

            prefix_sum_bg,
            duplicate_keys_bg,
            identify_ranges_bg,
            tile_raster_bg,
            fullscreen_copy_bg,
            update_sort_info_bg,
            tile_sort_bg,

            preprocess_sort_bg,
            preprocess_sort_uni,
            preprocess_sort_dis,

            cached_width: width,
            cached_height: height,
            cached_num_points: num_points,
            max_tile_entries,
            kernel_type,
            use_shared_mem,
        }
    }

    pub fn kernel_type(&self) -> u32 {
        self.kernel_type
    }

    pub fn use_shared_mem(&self) -> bool {
        self.use_shared_mem
    }

    fn create_tile_raster_bg(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        camera: &UniformBuffer<CameraUniform>,
        output_buf: &wgpu::Buffer,
        tile_payloads: &wgpu::Buffer,
        tile_starts: &wgpu::Buffer,
        tile_ends: &wgpu::Buffer,
        tile_raster_info: &UniformBuffer<TileRasterInfoUniform>,
        splat_2d_buf: Option<&wgpu::Buffer>,
    ) -> wgpu::BindGroup {
        // If no splat buffer yet, create a dummy
        let dummy;
        let splat_resource = if let Some(buf) = splat_2d_buf {
            buf.as_entire_binding()
        } else {
            dummy = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dummy splat buffer"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            });
            dummy.as_entire_binding()
        };

        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tile raster bg"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: splat_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tile_payloads.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tile_starts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: tile_raster_info.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: tile_ends.as_entire_binding(),
                },
            ],
        })
    }

    pub fn update_splat_bind_group(&mut self, device: &wgpu::Device, pc: &PointCloud) {
        self.tile_raster_bg = Self::create_tile_raster_bg(
            device,
            &self.tile_raster_bgl,
            &self.camera,
            &self.output_buf,
            &self.tile_payloads_a,
            &self.tile_starts_buf,
            &self.tile_ends_buf,
            &self.tile_raster_info,
            Some(pc.splat_2d_buffer()),
        );
    }

    pub fn prepare(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        sorter: &GPURSSorter,
        render_settings: SplattingArgs,
    ) {
        self.prepare_inner(encoder, queue, pc, sorter, render_settings, None);
    }

    /// Debug version: submit + poll after each pass to find which one hangs.
    /// Call with device = Some(&device) to enable per-pass sync.
    pub fn prepare_debug(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        sorter: &GPURSSorter,
        render_settings: SplattingArgs,
    ) {
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tile raster debug") });
        self.prepare_inner(&mut encoder, queue, pc, sorter, render_settings, Some(device));
        queue.submit(Some(encoder.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        log::info!("[TILE DEBUG] All passes completed successfully");

        // Readback diagnostics: num_visible and total_tile_entries
        Self::readback_u32(device, queue, &self.preprocess_sort_uni, 0, "num_visible (preprocess sort_infos.keys_size)");
        Self::readback_u32(device, queue, &self.tile_sort_uni, 0, "total_tile_entries (tile sort_infos.keys_size)");
    }

    /// Benchmark version: submits each pass individually with CPU timing + GPU timestamp queries.
    /// Press B to profile per-pass GPU costs.
    pub fn prepare_benchmark(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        sorter: &GPURSSorter,
        render_settings: SplattingArgs,
    ) {
        use std::time::Instant;

        // 8 compute passes we control get GPU timestamp queries (begin+end per pass = 16 timestamps)
        // Radix sort is timed with CPU only (we don't control its internal passes)
        const NUM_TIMED_PASSES: u32 = 8;
        const NUM_TIMESTAMPS: u32 = NUM_TIMED_PASSES * 2;

        let query_set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("benchmark query set"),
            ty: wgpu::QueryType::Timestamp,
            count: NUM_TIMESTAMPS,
        });

        let resolve_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("benchmark resolve"),
            size: (NUM_TIMESTAMPS as u64) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("benchmark readback"),
            size: (NUM_TIMESTAMPS as u64) * 8,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        // --- Uniform updates ---
        let viewport = render_settings.viewport;
        let width = viewport.x;
        let height = viewport.y;
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        let total_tiles = tiles_x * tiles_y;
        let num_points = pc.num_points();

        {
            let cam = self.camera.as_mut();
            let focal = render_settings.camera.projection.focal(render_settings.viewport);
            cam.set_focal(focal);
            cam.set_viewport(viewport.cast().unwrap());
            cam.set_camera(render_settings.camera);
            self.camera.sync(queue);
        }
        {
            let s = self.render_settings.as_mut();
            *s = SplattingArgsUniform::from_args_and_pc(render_settings, pc);
            self.render_settings.sync(queue);
        }
        {
            let t = self.tile_info.as_mut();
            t.tiles_x = tiles_x;
            t.tiles_y = tiles_y;
            t.total_tiles = total_tiles;
            self.tile_info.sync(queue);
        }
        {
            let t = self.tile_raster_info.as_mut();
            t.tiles_x = tiles_x;
            t.tiles_y = tiles_y;
            t.viewport_w = width;
            t.viewport_h = height;
            t.bg_r = render_settings.background_color.r as f32;
            t.bg_g = render_settings.background_color.g as f32;
            t.bg_b = render_settings.background_color.b as f32;
            self.tile_raster_info.sync(queue);
        }
        {
            let p = self.prefix_sum_info.as_mut();
            p.num_elements = num_points;
            self.prefix_sum_info.sync(queue);
        }
        {
            let d = self.duplicate_info.as_mut();
            d.num_visible = num_points;
            d.tiles_x = tiles_x;
            self.duplicate_info.sync(queue);
        }
        {
            let v = self.viewport_info.as_mut();
            v.width = width;
            v.height = height;
            self.viewport_info.sync(queue);
        }

        queue.write_buffer(&self.preprocess_sort_uni, 0, &[0u8; 4]);
        queue.write_buffer(&self.preprocess_sort_dis, 0, &[0u8; 4]);

        // Clear buffers
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("benchmark clear"),
            });
            encoder.clear_buffer(&self.tile_starts_buf, 0, None);
            encoder.clear_buffer(&self.tile_ends_buf, 0, None);
            encoder.clear_buffer(&self.tiles_touched_buf, 0, None);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        let wgs_points = (num_points as f32 / 256.0).ceil() as u32;
        let period = queue.get_timestamp_period();

        // Helper: submit one compute pass with GPU timestamps, poll, return GPU time in ms
        // Uses ComputePassDescriptor::timestamp_writes (works on Metal, unlike encoder.write_timestamp)
        let mut ts_idx: u32 = 0;

        macro_rules! timed_pass {
            ($label:expr, |$pass:ident| $body:block) => {{
                let begin_idx = ts_idx;
                let end_idx = ts_idx + 1;
                ts_idx += 2;
                let cpu_t0 = Instant::now();
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(concat!("bench ", $label)),
                });
                {
                    let mut $pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some($label),
                        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                            query_set: &query_set,
                            beginning_of_pass_write_index: Some(begin_idx),
                            end_of_pass_write_index: Some(end_idx),
                        }),
                    });
                    $body
                }
                queue.submit(Some(encoder.finish()));
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                let cpu_ms = cpu_t0.elapsed().as_secs_f64() * 1000.0;
                cpu_ms
            }};
        }

        // --- Pass 0: Preprocess ---
        let cpu_preprocess = timed_pass!("preprocess", |pass| {
            pass.set_pipeline(&self.preprocess_tile_pipeline);
            pass.set_bind_group(0, self.camera.bind_group(), &[]);
            pass.set_bind_group(1, pc.bind_group(), &[]);
            pass.set_bind_group(2, &self.preprocess_sort_bg, &[]);
            pass.set_bind_group(3, &self.preprocess_tile_bg3, &[]);
            pass.dispatch_workgroups(wgs_points, 1, 1);
        });

        // --- Pass 1: Prefix sum reduce ---
        let cpu_prefix_reduce = timed_pass!("prefix_reduce", |pass| {
            pass.set_pipeline(&self.prefix_sum_reduce);
            pass.set_bind_group(0, &self.prefix_sum_bg, &[]);
            pass.dispatch_workgroups(wgs_points, 1, 1);
        });

        // --- Pass 2: Prefix sum scan blocks ---
        let cpu_prefix_scan = timed_pass!("prefix_scan_blocks", |pass| {
            pass.set_pipeline(&self.prefix_sum_scan_blocks);
            pass.set_bind_group(0, &self.prefix_sum_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        });

        // --- Pass 3: Prefix sum propagate ---
        let cpu_prefix_prop = timed_pass!("prefix_propagate", |pass| {
            pass.set_pipeline(&self.prefix_sum_propagate);
            pass.set_bind_group(0, &self.prefix_sum_bg, &[]);
            pass.dispatch_workgroups(wgs_points, 1, 1);
        });

        // --- Pass 4: Update sort info ---
        let cpu_update = timed_pass!("update_sort_info", |pass| {
            pass.set_pipeline(&self.update_sort_info_pipeline);
            pass.set_bind_group(0, &self.update_sort_info_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        });

        // --- Pass 5: Duplicate keys ---
        let cpu_duplicate = timed_pass!("duplicate_keys", |pass| {
            pass.set_pipeline(&self.duplicate_keys_pipeline);
            pass.set_bind_group(0, &self.duplicate_keys_bg, &[]);
            pass.dispatch_workgroups(wgs_points, 1, 1);
        });

        // --- Pass 6: Radix sort (CPU-timed only, multiple internal passes) ---
        let cpu_sort = {
            let cpu_t0 = Instant::now();
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("bench radix_sort"),
            });
            sorter.record_sort_indirect(&self.tile_sort_bg, &self.tile_sort_dis, &mut encoder);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            cpu_t0.elapsed().as_secs_f64() * 1000.0
        };

        // --- Pass 7: Identify ranges ---
        let cpu_identify = timed_pass!("identify_ranges", |pass| {
            pass.set_pipeline(&self.identify_ranges_pipeline);
            pass.set_bind_group(0, &self.identify_ranges_bg, &[]);
            let wgs = (self.max_tile_entries as f32 / 256.0).ceil() as u32;
            pass.dispatch_workgroups(wgs.max(1), 1, 1);
        });

        // --- Pass 8: Tile raster ---
        let cpu_tile_raster = timed_pass!("tile_raster", |pass| {
            pass.set_pipeline(&self.tile_raster_pipeline);
            pass.set_bind_group(0, &self.tile_raster_bg, &[]);
            pass.dispatch_workgroups(tiles_x, tiles_y, 1);
        });

        // Read back actual tile entry count
        Self::readback_u32(device, queue, &self.preprocess_sort_uni, 0, "num_visible");
        Self::readback_u32(device, queue, &self.tile_sort_uni, 0, "total_tile_entries");

        // Resolve GPU timestamps
        {
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("benchmark resolve"),
            });
            encoder.resolve_query_set(&query_set, 0..ts_idx, &resolve_buf, 0);
            encoder.copy_buffer_to_buffer(&resolve_buf, 0, &readback_buf, 0, (ts_idx as u64) * 8);
            queue.submit(Some(encoder.finish()));
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }

        // Read back GPU timestamps
        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let timestamps: &[u64] = bytemuck::cast_slice(&data);

        let gpu_labels = [
            "preprocess",
            "prefix_reduce",
            "prefix_scan_blocks",
            "prefix_propagate",
            "update_sort_info",
            "duplicate_keys",
            // radix_sort has no GPU timestamps
            "identify_ranges",
            "tile_raster",
        ];
        let cpu_times = [
            cpu_preprocess, cpu_prefix_reduce, cpu_prefix_scan, cpu_prefix_prop,
            cpu_update, cpu_duplicate, cpu_sort, cpu_identify, cpu_tile_raster,
        ];
        let all_labels = [
            "preprocess", "prefix_reduce", "prefix_scan_blocks", "prefix_propagate",
            "update_sort_info", "duplicate_keys", "radix_sort", "identify_ranges", "tile_raster",
        ];

        log::info!("===== COMPUTE RASTER GPU BENCHMARK =====");
        log::info!("  viewport: {}x{}, tiles: {}x{} = {}, points: {}, max_tile_entries: {}",
            width, height, tiles_x, tiles_y, total_tiles, num_points, self.max_tile_entries);
        log::info!("  {:20}   {:>8}   {:>8}", "PASS", "GPU ms", "CPU ms");
        let mut total_cpu = 0.0f64;
        let mut gpu_idx: usize = 0;
        for (i, label) in all_labels.iter().enumerate() {
            let cpu_ms = cpu_times[i];
            total_cpu += cpu_ms;

            if *label == "radix_sort" {
                // No GPU timestamps for radix sort
                log::info!("  {:20} : {:>8}   {:8.2}", label, "N/A", cpu_ms);
            } else {
                let start = timestamps[gpu_idx * 2];
                let end = timestamps[gpu_idx * 2 + 1];
                let gpu_ns = (end.wrapping_sub(start)) as f64 * period as f64;
                let gpu_ms = gpu_ns / 1_000_000.0;
                log::info!("  {:20} : {:8.2}   {:8.2}", label, gpu_ms, cpu_ms);
                gpu_idx += 1;
            }
        }
        log::info!("  {:20} : {:>8}   {:8.2}", "TOTAL", "", total_cpu);
        log::info!("========================================");
    }

    fn readback_u32(device: &wgpu::Device, queue: &wgpu::Queue, src: &wgpu::Buffer, offset: u64, label: &str) {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback staging"),
            size: 4,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("readback") });
        enc.copy_buffer_to_buffer(src, offset, &staging, 0, 4);
        queue.submit(Some(enc.finish()));
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();

        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| { tx.send(r).unwrap(); });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let data = slice.get_mapped_range();
        let val = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        log::info!("[TILE DEBUG] {} = {}", label, val);
    }

    fn prepare_inner(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        sorter: &GPURSSorter,
        render_settings: SplattingArgs,
        debug_device: Option<&wgpu::Device>,
    ) {
        let viewport = render_settings.viewport;
        let width = viewport.x;
        let height = viewport.y;
        let tiles_x = (width + TILE_SIZE - 1) / TILE_SIZE;
        let tiles_y = (height + TILE_SIZE - 1) / TILE_SIZE;
        let total_tiles = tiles_x * tiles_y;
        let num_points = pc.num_points();

        // Update camera uniform
        {
            let cam = self.camera.as_mut();
            let focal = render_settings
                .camera
                .projection
                .focal(render_settings.viewport);
            cam.set_focal(focal);
            cam.set_viewport(viewport.cast().unwrap());
            cam.set_camera(render_settings.camera);
            self.camera.sync(queue);
        }

        // Update render settings
        {
            let s = self.render_settings.as_mut();
            *s = SplattingArgsUniform::from_args_and_pc(render_settings, pc);
            self.render_settings.sync(queue);
        }

        // Update tile info
        {
            let t = self.tile_info.as_mut();
            t.tiles_x = tiles_x;
            t.tiles_y = tiles_y;
            t.total_tiles = total_tiles;
            self.tile_info.sync(queue);
        }

        // Update tile raster info
        {
            let t = self.tile_raster_info.as_mut();
            t.tiles_x = tiles_x;
            t.tiles_y = tiles_y;
            t.viewport_w = width;
            t.viewport_h = height;
            t.bg_r = render_settings.background_color.r as f32;
            t.bg_g = render_settings.background_color.g as f32;
            t.bg_b = render_settings.background_color.b as f32;
            self.tile_raster_info.sync(queue);
        }

        // Update prefix sum info
        {
            let p = self.prefix_sum_info.as_mut();
            p.num_elements = num_points;
            self.prefix_sum_info.sync(queue);
        }

        // Update duplicate info
        {
            let d = self.duplicate_info.as_mut();
            d.num_visible = num_points; // will be overwritten by actual visible count
            d.tiles_x = tiles_x;
            self.duplicate_info.sync(queue);
        }

        // Update viewport info
        {
            let v = self.viewport_info.as_mut();
            v.width = width;
            v.height = height;
            self.viewport_info.sync(queue);
        }

        // Reset preprocess counters
        queue.write_buffer(&self.preprocess_sort_uni, 0, &[0u8; 4]); // keys_size = 0
        queue.write_buffer(&self.preprocess_sort_dis, 0, &[0u8; 4]); // dispatch_x = 0

        // Zero tile starts/ends and tiles_touched
        encoder.clear_buffer(&self.tile_starts_buf, 0, None);
        encoder.clear_buffer(&self.tile_ends_buf, 0, None);
        encoder.clear_buffer(&self.tiles_touched_buf, 0, None);

        // Debug sync helper: submit current encoder, poll, log, create new encoder
        macro_rules! debug_sync {
            ($label:expr, $encoder:ident, $debug_device:expr, $queue:expr) => {
                if let Some(dev) = $debug_device {
                    log::info!("[TILE DEBUG] submitting: {}", $label);
                    let finished = std::mem::replace(
                        $encoder,
                        dev.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                            label: Some("tile raster debug"),
                        }),
                    );
                    $queue.submit(Some(finished.finish()));
                    dev.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                    log::info!("[TILE DEBUG] completed: {}", $label);
                }
            };
        }

        // ========== Pass 1: Tile preprocess ==========
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile preprocess"),
                ..Default::default()
            });
            pass.set_pipeline(&self.preprocess_tile_pipeline);
            pass.set_bind_group(0, self.camera.bind_group(), &[]);
            pass.set_bind_group(1, pc.bind_group(), &[]);
            pass.set_bind_group(2, &self.preprocess_sort_bg, &[]);
            pass.set_bind_group(3, &self.preprocess_tile_bg3, &[]);
            let wgs = (num_points as f32 / 256.0).ceil() as u32;
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        debug_sync!("preprocess", encoder, debug_device, queue);

        // ========== Pass 2: Prefix sum on tiles_touched ==========
        // Note: num_elements for prefix sum = num_visible (from preprocess).
        // For simplicity, we run prefix sum on the full num_points buffer.
        // Elements beyond num_visible will be 0, so this is safe.
        {
            let wgs = (num_points as f32 / 256.0).ceil() as u32;

            // Reduce pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prefix sum reduce"),
                    ..Default::default()
                });
                pass.set_pipeline(&self.prefix_sum_reduce);
                pass.set_bind_group(0, &self.prefix_sum_bg, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            debug_sync!("prefix_sum_reduce", encoder, debug_device, queue);

            // Scan blocks pass (single workgroup)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prefix sum scan blocks"),
                    ..Default::default()
                });
                pass.set_pipeline(&self.prefix_sum_scan_blocks);
                pass.set_bind_group(0, &self.prefix_sum_bg, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            debug_sync!("prefix_sum_scan_blocks", encoder, debug_device, queue);

            // Propagate pass
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("prefix sum propagate"),
                    ..Default::default()
                });
                pass.set_pipeline(&self.prefix_sum_propagate);
                pass.set_bind_group(0, &self.prefix_sum_bg, &[]);
                pass.dispatch_workgroups(wgs, 1, 1);
            }
            debug_sync!("prefix_sum_propagate", encoder, debug_device, queue);
        }

        // ========== Pass 2.5: Update tile sort info ==========
        // Reads prefix sum output + num_visible, writes to tile sort uniform/dispatch
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("update tile sort info"),
                ..Default::default()
            });
            pass.set_pipeline(&self.update_sort_info_pipeline);
            pass.set_bind_group(0, &self.update_sort_info_bg, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        debug_sync!("update_sort_info", encoder, debug_device, queue);

        // ========== Pass 3: Duplicate keys ==========
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("duplicate keys"),
                ..Default::default()
            });
            pass.set_pipeline(&self.duplicate_keys_pipeline);
            pass.set_bind_group(0, &self.duplicate_keys_bg, &[]);
            let wgs = (num_points as f32 / 256.0).ceil() as u32;
            pass.dispatch_workgroups(wgs, 1, 1);
        }
        debug_sync!("duplicate_keys", encoder, debug_device, queue);

        // ========== Pass 4: Radix sort tile keys ==========
        sorter.record_sort_indirect(&self.tile_sort_bg, &self.tile_sort_dis, encoder);
        debug_sync!("radix_sort", encoder, debug_device, queue);

        // ========== Pass 5: Identify tile ranges ==========
        // Parallel: each thread checks one entry. Safe on Metal because tile_starts
        // and tile_ends are separate array<u32> buffers (no vec2 partial-write tearing).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("identify ranges"),
                ..Default::default()
            });
            pass.set_pipeline(&self.identify_ranges_pipeline);
            pass.set_bind_group(0, &self.identify_ranges_bg, &[]);
            // Dispatch enough workgroups for max_tile_entries
            // (actual count is dynamic, shader checks bounds via sort_infos.keys_size)
            let wgs = (self.max_tile_entries as f32 / 256.0).ceil() as u32;
            pass.dispatch_workgroups(wgs.max(1), 1, 1);
        }
        debug_sync!("identify_ranges", encoder, debug_device, queue);

        // ========== Pass 6: Tile raster ==========
        // Note: tile_raster_bg must be set up via update_splat_bind_group() before first prepare()
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tile raster"),
                ..Default::default()
            });
            pass.set_pipeline(&self.tile_raster_pipeline);
            pass.set_bind_group(0, &self.tile_raster_bg, &[]);
            pass.dispatch_workgroups(tiles_x, tiles_y, 1);
        }
        debug_sync!("tile_raster", encoder, debug_device, queue);
    }

    pub fn render<'rpass>(&'rpass self, render_pass: &mut wgpu::RenderPass<'rpass>) {
        render_pass.set_pipeline(&self.fullscreen_copy_pipeline);
        render_pass.set_bind_group(0, &self.fullscreen_copy_bg, &[]);
        render_pass.draw(0..4, 0..1);
    }

    pub fn output_buf(&self) -> &wgpu::Buffer {
        &self.output_buf
    }
}

// Helper functions for bind group layout entries
fn storage_rw_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_ro_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Inline compute shader that bridges prefix sum output to sort parameters.
/// Reads tile_offsets[num_visible - 1] to get total tile entries,
/// then writes keys_size and dispatch_x for the radix sort.
const UPDATE_SORT_INFO_SHADER: &str = r#"
struct SortInfos {
    keys_size: u32,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct DispatchIndirect {
    dispatch_x: u32,
    dispatch_y: u32,
    dispatch_z: u32,
}

@group(0) @binding(0)
var<storage, read> tile_offsets: array<u32>;

@group(0) @binding(1)
var<storage, read> preprocess_infos: SortInfos;

@group(0) @binding(2)
var<storage, read_write> tile_sort_infos: SortInfos;

@group(0) @binding(3)
var<storage, read_write> tile_sort_dispatch: DispatchIndirect;

@compute @workgroup_size(1)
fn main() {
    let num_visible = preprocess_infos.keys_size;
    if num_visible == 0u {
        tile_sort_infos.keys_size = 0u;
        tile_sort_dispatch.dispatch_x = 0u;
        tile_sort_dispatch.dispatch_y = 1u;
        tile_sort_dispatch.dispatch_z = 1u;
        return;
    }

    // Total tile entries = inclusive prefix sum at last visible element
    let total_entries = tile_offsets[num_visible - 1u];
    tile_sort_infos.keys_size = total_entries;

    // Dispatch X for radix sort = ceil(total_entries / (256 * 15))
    let keys_per_wg = 256u * 15u;
    let dispatch_x = (total_entries + keys_per_wg - 1u) / keys_per_wg;
    tile_sort_dispatch.dispatch_x = dispatch_x;
    tile_sort_dispatch.dispatch_y = 1u;
    tile_sort_dispatch.dispatch_z = 1u;
}
"#;
