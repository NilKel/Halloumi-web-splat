use crate::gpu_rs::{GPURSSorter, PointCloudSortStuff};
use crate::pointcloud::Aabb;
use crate::utils::GPUStopwatch;
use crate::{
    camera::{Camera, PerspectiveCamera, VIEWPORT_Y_FLIP},
    pointcloud::PointCloud,
    uniform::UniformBuffer,
};

use std::num::NonZeroU64;
use std::time::Duration;

use wgpu::{Extent3d, MultisampleState, include_wgsl};

use cgmath::{EuclideanSpace, Matrix4, Point3, SquareMatrix, Vector2, Vector4};

/// Uniform buffer for 2DGS texture parameters
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TexParamsUniform {
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub kernel_type: u32,  // 0=Gaussian, 1=Beta, 2=Flex, 3=General, 4=BetaScaled
    pub uv_extent_bits: u32, // f32 reinterpreted as u32 for uniform alignment
}

impl Default for TexParamsUniform {
    fn default() -> Self {
        Self {
            atlas_width: 0,
            atlas_height: 0,
            kernel_type: 0,
            uv_extent_bits: 4.0f32.to_bits(),
        }
    }
}

pub struct GaussianRenderer {
    pipeline: wgpu::RenderPipeline,
    camera: UniformBuffer<CameraUniform>,

    render_settings: UniformBuffer<SplattingArgsUniform>,
    preprocess: PreprocessPipeline,
    copy_count: CopyCountPipeline,

    draw_indirect_buffer: wgpu::Buffer,
    #[allow(dead_code)]
    draw_indirect: wgpu::BindGroup,
    color_format: wgpu::TextureFormat,
    sorter: GPURSSorter,
    sorter_suff: Option<PointCloudSortStuff>,

    is_2dgs: bool,
    // 2DGS-specific: bind group for textures+camera+params (group 3 in render shader)
    tex_params: Option<UniformBuffer<TexParamsUniform>>,
    render_tex_bind_group: Option<wgpu::BindGroup>,
}

impl GaussianRenderer {
    pub async fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        compressed: bool,
    ) -> Self {
        Self::new_with_mode(device, queue, color_format, sh_deg, compressed, false, None).await
    }

    pub async fn new_2dgs(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        pc: &PointCloud,
    ) -> Self {
        Self::new_with_mode(device, queue, color_format, sh_deg, false, true, Some(pc)).await
    }

    async fn new_with_mode(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        sh_deg: u32,
        compressed: bool,
        is_2dgs: bool,
        pc: Option<&PointCloud>,
    ) -> Self {
        let pipeline = if is_2dgs {
            // 2DGS render pipeline:
            // Group 0: splat_2d (binding 2)
            // Group 1: sort indices (binding 4)
            // Group 2: surfels (binding 0)
            // Group 3: textures (binding 0) + camera (binding 1) + tex_params (binding 2)
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("2dgs render pipeline layout"),
                bind_group_layouts: &[
                    &PointCloud::bind_group_layout_render(device),
                    &GPURSSorter::bind_group_layout_rendering(device),
                    &PointCloud::bind_group_layout_surfel_render(device),
                    &Self::bind_group_layout_2dgs_render(device),
                ],
                push_constant_ranges: &[],
            });

            let shader_src = format!(
                "const MAX_SH_DEG:u32 = {:}u;\n{:}",
                sh_deg,
                include_str!("shaders/gaussian_2dgs.wgsl")
            );
            let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("2dgs render shader"),
                source: wgpu::ShaderSource::Wgsl(shader_src.into()),
            });

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("2dgs render pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        } else {
            // Standard 3DGS render pipeline
            let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("render pipeline layout"),
                bind_group_layouts: &[
                    &PointCloud::bind_group_layout_render(device),
                    &GPURSSorter::bind_group_layout_rendering(device),
                ],
                push_constant_ranges: &[],
            });

            let shader = device.create_shader_module(wgpu::include_wgsl!("shaders/gaussian.wgsl"));

            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("render pipeline"),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    topology: wgpu::PrimitiveTopology::TriangleStrip,
                    strip_index_format: None,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: None,
                    polygon_mode: wgpu::PolygonMode::Fill,
                    unclipped_depth: false,
                    conservative: false,
                },
                depth_stencil: None,
                multisample: wgpu::MultisampleState::default(),
                multiview: None,
                cache: None,
            })
        };

        let draw_indirect_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("indirect draw buffer"),
            size: std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64,
            usage: wgpu::BufferUsages::INDIRECT
                | wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let indirect_layout = Self::bind_group_layout(device);
        let draw_indirect = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("draw indirect buffer"),
            layout: &indirect_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: draw_indirect_buffer.as_entire_binding(),
            }],
        });

        let sorter = GPURSSorter::new(device, queue).await;

        let camera = UniformBuffer::new_default(device, Some("camera uniform buffer"));
        let preprocess = PreprocessPipeline::new(device, sh_deg, compressed, is_2dgs);

        // 2DGS-specific setup
        let mut tex_params = None;
        let mut render_tex_bind_group = None;
        if is_2dgs {
            let tp = UniformBuffer::new(
                device,
                TexParamsUniform {
                    atlas_width: pc.map_or(0, |p| p.atlas_width()),
                    atlas_height: pc.map_or(0, |p| p.atlas_height()),
                    kernel_type: pc.map_or(0, |p| p.kernel_type()),
                    uv_extent_bits: pc.map_or(4.0f32, |p| p.uv_extent()).to_bits(),
                },
                Some("tex params uniform buffer"),
            );

            // Create the render bind group for group 3:
            // binding 0: atlas_texture, binding 1: atlas_rects, binding 2: camera, binding 3: tex_params
            let layout = Self::bind_group_layout_2dgs_render(device);

            // Atlas texture buffer (or dummy)
            let dummy_atlas;
            let atlas_resource = if let Some(buf) = pc.and_then(|p| p.atlas_buffer()) {
                buf.as_entire_binding()
            } else {
                dummy_atlas = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dummy atlas buffer"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                dummy_atlas.as_entire_binding()
            };

            // Atlas rects buffer (or dummy)
            let dummy_rects;
            let rects_resource = if let Some(buf) = pc.and_then(|p| p.atlas_rects_buffer()) {
                buf.as_entire_binding()
            } else {
                dummy_rects = device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("dummy rects buffer"),
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
                dummy_rects.as_entire_binding()
            };

            let entries = [
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: atlas_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rects_resource,
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: camera.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tp.buffer().as_entire_binding(),
                },
            ];

            render_tex_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("2dgs render tex bind group"),
                layout: &layout,
                entries: &entries,
            }));

            tex_params = Some(tp);
        }

        let copy_count = CopyCountPipeline::new(device);

        GaussianRenderer {
            pipeline,
            camera,
            preprocess,
            copy_count,
            draw_indirect_buffer,
            draw_indirect,
            color_format,
            sorter,
            sorter_suff: None,
            render_settings: UniformBuffer::new_default(
                device,
                Some("render settings uniform buffer"),
            ),
            is_2dgs,
            tex_params,
            render_tex_bind_group,
        }
    }

    pub(crate) fn camera(&self) -> &UniformBuffer<CameraUniform> {
        &self.camera
    }

    fn preprocess<'a>(
        &'a mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        queue: &wgpu::Queue,
        pc: &'a PointCloud,
        render_settings: SplattingArgs,
    ) {
        let camera = render_settings.camera;
        let uniform = self.camera.as_mut();
        let focal = camera.projection.focal(render_settings.viewport);
        let viewport = render_settings.viewport;
        uniform.set_focal(focal);
        uniform.set_viewport(viewport.cast().unwrap());
        uniform.set_camera(camera);

        self.camera.sync(queue);

        let settings_uniform = self.render_settings.as_mut();
        *settings_uniform = SplattingArgsUniform::from_args_and_pc(render_settings, pc);
        self.render_settings.sync(queue);

        // Set vertex_count=4 for the indirect draw call.
        // instance_count will be set by copy_count shader after preprocess + sort.
        queue.write_buffer(
            &self.draw_indirect_buffer,
            0,
            wgpu::util::DrawIndirectArgs {
                vertex_count: 4,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }
            .as_bytes(),
        );
        let depth_buffer = &self.sorter_suff.as_ref().unwrap().sorter_bg_pre;
        self.preprocess.run(
            encoder,
            pc,
            &self.camera,
            &self.render_settings,
            depth_buffer,
        );
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub async fn num_visible_points(&self, device: &wgpu::Device, queue: &wgpu::Queue) -> u32 {
        let n = {
            let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();

            wgpu::util::DownloadBuffer::read_buffer(
                device,
                queue,
                &self.draw_indirect_buffer.slice(..),
                move |b| {
                    let download = b.unwrap();
                    let data = download.as_ref();
                    let num_points = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                    tx.send(num_points).unwrap();
                },
            );
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            rx.receive().await.unwrap()
        };
        return n;
    }

    pub fn prepare(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pc: &PointCloud,
        render_settings: SplattingArgs,
        stopwatch: &mut Option<GPUStopwatch>,
    ) {
        if self.sorter_suff.is_none()
            || self
                .sorter_suff
                .as_ref()
                .is_some_and(|s| s.num_points != pc.num_points() as usize)
        {
            log::debug!("created sort buffers for {:} points", pc.num_points());
            let mut sort_stuff = self.sorter
                .create_sort_stuff(device, pc.num_points() as usize, &self.draw_indirect_buffer);
            sort_stuff.copy_count_bg = self.copy_count.create_bind_group(
                device,
                &sort_stuff.sorter_uni,
                &self.draw_indirect_buffer,
            );
            self.sorter_suff = Some(sort_stuff);
        }

        log::info!("prepare() called: {} points, viewport {}x{}",
            pc.num_points(),
            render_settings.viewport.x,
            render_settings.viewport.y,
        );

        GPURSSorter::record_reset_indirect_buffer(
            &self.sorter_suff.as_ref().unwrap().sorter_dis,
            &self.sorter_suff.as_ref().unwrap().sorter_uni,
            &queue,
        );

        // convert 3D gaussian splats to 2D gaussian splats
        if let Some(stopwatch) = stopwatch {
            stopwatch.start(encoder, "preprocess").unwrap();
        }

        self.preprocess(encoder, queue, &pc, render_settings);
        if let Some(stopwatch) = stopwatch {
            stopwatch.stop(encoder, "preprocess").unwrap();
        }

        // sort 2d splats
        if let Some(stopwatch) = stopwatch {
            stopwatch.start(encoder, "sorting").unwrap();
        }
        self.sorter.record_sort_indirect(
            &self.sorter_suff.as_ref().unwrap().sorter_bg,
            &self.sorter_suff.as_ref().unwrap().sorter_dis,
            encoder,
        );
        if let Some(stopwatch) = stopwatch {
            stopwatch.stop(encoder, "sorting").unwrap();
        }

        // Copy keys_size → instance_count via 1-thread compute shader.
        // Workaround: copy_buffer_to_buffer and clear_buffer are broken on Metal.
        self.copy_count.run(
            encoder,
            self.sorter_suff.as_ref().unwrap().copy_count_bg.as_ref().unwrap(),
        );
    }

    pub fn render<'rpass>(
        &'rpass self,
        render_pass: &mut wgpu::RenderPass<'rpass>,
        pc: &'rpass PointCloud,
    ) {
        render_pass.set_bind_group(0, pc.render_bind_group(), &[]);
        render_pass.set_bind_group(1, &self.sorter_suff.as_ref().unwrap().sorter_render_bg, &[]);

        if self.is_2dgs {
            // 2DGS: set surfel and texture bind groups
            if let Some(surfel_bg) = pc.surfel_render_bind_group() {
                render_pass.set_bind_group(2, surfel_bg, &[]);
            }
            if let Some(tex_bg) = &self.render_tex_bind_group {
                render_pass.set_bind_group(3, tex_bg, &[]);
            }
        }

        render_pass.set_pipeline(&self.pipeline);
        render_pass.draw_indirect(&self.draw_indirect_buffer, 0);
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("draw indirect"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: Some(
                        NonZeroU64::new(std::mem::size_of::<wgpu::util::DrawIndirectArgs>() as u64)
                            .unwrap(),
                    ),
                },
                count: None,
            }],
        })
    }

    /// Bind group layout for 2DGS render pass group 3:
    /// binding 0: textures (storage, read), binding 1: camera (uniform), binding 2: tex_params (uniform)
    pub fn bind_group_layout_2dgs_render(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("2dgs render bind group layout"),
            entries: &[
                // binding 0: atlas texture (FP16 packed as u32)
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
                // binding 1: atlas rects [N, 4] f32
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 2: camera uniforms
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // binding 3: tex params (atlas dims, kernel_type, uv_extent)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn color_format(&self) -> wgpu::TextureFormat {
        self.color_format
    }

    pub(crate) fn render_settings(&self) -> &UniformBuffer<SplattingArgsUniform> {
        &self.render_settings
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// the cameras view matrix
    pub(crate) view_matrix: Matrix4<f32>,
    /// inverse view matrix
    pub(crate) view_inv_matrix: Matrix4<f32>,

    // the cameras projection matrix
    pub(crate) proj_matrix: Matrix4<f32>,

    // inverse projection matrix
    pub(crate) proj_inv_matrix: Matrix4<f32>,

    pub(crate) viewport: Vector2<f32>,
    pub(crate) focal: Vector2<f32>,
}

impl Default for CameraUniform {
    fn default() -> Self {
        Self {
            view_matrix: Matrix4::identity(),
            view_inv_matrix: Matrix4::identity(),
            proj_matrix: Matrix4::identity(),
            proj_inv_matrix: Matrix4::identity(),
            viewport: Vector2::new(1., 1.),
            focal: Vector2::new(1., 1.),
        }
    }
}

impl CameraUniform {
    pub(crate) fn set_view_mat(&mut self, view_matrix: Matrix4<f32>) {
        self.view_matrix = view_matrix;
        self.view_inv_matrix = view_matrix.invert().unwrap();
    }

    pub(crate) fn set_proj_mat(&mut self, proj_matrix: Matrix4<f32>) {
        self.proj_matrix = VIEWPORT_Y_FLIP * proj_matrix;
        self.proj_inv_matrix = proj_matrix.invert().unwrap();
    }

    pub fn set_camera(&mut self, camera: impl Camera) {
        self.set_proj_mat(camera.proj_matrix());
        self.set_view_mat(camera.view_matrix());
    }

    pub fn set_viewport(&mut self, viewport: Vector2<f32>) {
        self.viewport = viewport;
    }
    pub fn set_focal(&mut self, focal: Vector2<f32>) {
        self.focal = focal
    }
}

struct PreprocessPipeline(wgpu::ComputePipeline);

impl PreprocessPipeline {
    fn new(device: &wgpu::Device, sh_deg: u32, compressed: bool, is_2dgs: bool) -> Self {
        // 2DGS and 3DGS use the same bind group layout for preprocess
        // (camera, point cloud, sort, render settings)
        let pc_layout = if is_2dgs {
            PointCloud::bind_group_layout(device) // surfels use same layout as uncompressed
        } else if compressed {
            PointCloud::bind_group_layout_compressed(device)
        } else {
            PointCloud::bind_group_layout(device)
        };

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("preprocess pipeline layout"),
            bind_group_layouts: &[
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &pc_layout,
                &GPURSSorter::bind_group_layout_preprocess(device),
                &UniformBuffer::<SplattingArgsUniform>::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("preprocess shader"),
            source: wgpu::ShaderSource::Wgsl(Self::build_shader(sh_deg, compressed, is_2dgs).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("preprocess pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("preprocess"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self(pipeline)
    }

    fn build_shader(sh_deg: u32, compressed: bool, is_2dgs: bool) -> String {
        let shader_src: &str = if is_2dgs {
            include_str!("shaders/preprocess_2dgs.wgsl")
        } else if !compressed {
            include_str!("shaders/preprocess.wgsl")
        } else {
            include_str!("shaders/preprocess_compressed.wgsl")
        };
        let shader = format!(
            "
        const MAX_SH_DEG:u32 = {:}u;
        {:}",
            sh_deg, shader_src
        );
        return shader;
    }

    fn run<'a>(
        &mut self,
        encoder: &'a mut wgpu::CommandEncoder,
        pc: &PointCloud,
        camera: &UniformBuffer<CameraUniform>,
        render_settings: &UniformBuffer<SplattingArgsUniform>,
        sort_bg: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("preprocess compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&self.0);
        pass.set_bind_group(0, camera.bind_group(), &[]);
        pass.set_bind_group(1, pc.bind_group(), &[]);
        pass.set_bind_group(2, sort_bg, &[]);
        pass.set_bind_group(3, render_settings.bind_group(), &[]);

        let wgs_x = (pc.num_points() as f32 / 256.0).ceil() as u32;
        pass.dispatch_workgroups(wgs_x, 1, 1);
    }
}

/// Tiny 1-thread compute shader that copies sort_infos.keys_size to draw_indirect.instance_count.
/// Workaround for Metal/Apple Silicon where copy_buffer_to_buffer and clear_buffer don't
/// synchronize properly with compute shader writes to the same buffer.
struct CopyCountPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl CopyCountPipeline {
    fn new(device: &wgpu::Device) -> Self {
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("copy count bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("copy count pipeline layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("copy count shader"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
struct SortInfos {
    keys_size: atomic<u32>,
    padded_size: u32,
    even_pass: u32,
    odd_pass: u32,
}

struct DrawIndirect {
    vertex_count: u32,
    instance_count: atomic<u32>,
    base_vertex: u32,
    base_instance: u32,
}

@group(0) @binding(0)
var<storage, read_write> sort_infos: SortInfos;
@group(0) @binding(1)
var<storage, read_write> draw_indirect: DrawIndirect;

@compute @workgroup_size(1)
fn main() {
    let count = atomicLoad(&sort_infos.keys_size);
    atomicStore(&draw_indirect.instance_count, count);
}
"#
                .into(),
            ),
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("copy count pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    fn create_bind_group(
        &self,
        device: &wgpu::Device,
        sort_uni: &wgpu::Buffer,
        draw_indirect: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("copy count bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sort_uni.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: draw_indirect.as_entire_binding(),
                },
            ],
        })
    }

    fn run(&self, encoder: &mut wgpu::CommandEncoder, bind_group: &wgpu::BindGroup) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("copy count compute pass"),
            ..Default::default()
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }
}

pub struct Display {
    pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    format: wgpu::TextureFormat,
    view: wgpu::TextureView,
}

impl Display {
    pub fn new(
        device: &wgpu::Device,
        source_format: wgpu::TextureFormat,
        target_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("display pipeline layout"),
            bind_group_layouts: &[
                &Self::bind_group_layout(device),
                &UniformBuffer::<CameraUniform>::bind_group_layout(device),
                &UniformBuffer::<SplattingArgsUniform>::bind_group_layout(device),
            ],
            push_constant_ranges: &[],
        });
        let shader = device.create_shader_module(include_wgsl!("shaders/display.wgsl"));
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("display pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: target_format,
                    blend: Some(wgpu::BlendState::PREMULTIPLIED_ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            multiview: None,
            cache: None,
        });
        let (view, bind_group) = Self::create_render_target(device, source_format, width, height);
        Self {
            pipeline,
            view,
            format: source_format,
            bind_group,
        }
    }

    pub fn texture(&self) -> &wgpu::TextureView {
        &self.view
    }

    fn create_render_target(
        device: &wgpu::Device,
        format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> (wgpu::TextureView, wgpu::BindGroup) {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("display render image"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let texture_view = texture.create_view(&Default::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("render target bind group"),
            layout: &Display::bind_group_layout(device),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&texture_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });
        return (texture_view, bind_group);
    }

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("disply bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        })
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        let (view, bind_group) = Self::create_render_target(device, self.format, width, height);
        self.bind_group = bind_group;
        self.view = view;
    }

    pub fn render(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        background_color: wgpu::Color,
        camera: &UniformBuffer<CameraUniform>,
        render_settings: &UniformBuffer<SplattingArgsUniform>,
    ) {
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(background_color),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            ..Default::default()
        });
        render_pass.set_bind_group(0, &self.bind_group, &[]);
        render_pass.set_bind_group(1, camera.bind_group(), &[]);
        render_pass.set_bind_group(2, render_settings.bind_group(), &[]);
        render_pass.set_pipeline(&self.pipeline);

        render_pass.draw(0..4, 0..1);
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct SplattingArgs {
    pub camera: PerspectiveCamera,
    pub viewport: Vector2<u32>,
    pub gaussian_scaling: f32,
    pub max_sh_deg: u32,
    pub mip_splatting: Option<bool>,
    pub kernel_size: Option<f32>,
    pub clipping_box: Option<Aabb<f32>>,
    pub walltime: Duration,
    pub scene_center: Option<Point3<f32>>,
    pub scene_extend: Option<f32>,
    pub background_color: wgpu::Color,
}

pub const DEFAULT_KERNEL_SIZE: f32 = 0.3;
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SplattingArgsUniform {
    clipping_box_min: Vector4<f32>,
    clipping_box_max: Vector4<f32>,

    gaussian_scaling: f32,
    max_sh_deg: u32,
    mip_splatting: u32,
    kernel_size: f32,

    walltime: f32,
    scene_extend: f32,
    _pad: [u32; 2],

    scene_center: Vector4<f32>,
}

impl SplattingArgsUniform {
    /// replaces values with default values for point cloud
    pub fn from_args_and_pc(args: SplattingArgs, pc: &PointCloud) -> Self {
        Self {
            gaussian_scaling: args.gaussian_scaling,
            max_sh_deg: args.max_sh_deg,
            mip_splatting: args
                .mip_splatting
                .map(|v| v as u32)
                .unwrap_or(pc.mip_splatting().unwrap_or(false) as u32),
            kernel_size: args
                .kernel_size
                .unwrap_or(pc.dilation_kernel_size().unwrap_or(DEFAULT_KERNEL_SIZE)),
            clipping_box_min: args
                .clipping_box
                .map_or(pc.bbox().min, |b| b.min)
                .to_vec()
                .extend(0.),
            clipping_box_max: args
                .clipping_box
                .map_or(pc.bbox().max, |b| b.max)
                .to_vec()
                .extend(0.),
            walltime: args.walltime.as_secs_f32(),
            scene_center: pc.center().to_vec().extend(0.),
            scene_extend: args
                .scene_extend
                .unwrap_or(pc.bbox().radius())
                .max(pc.bbox().radius()),
            ..Default::default()
        }
    }
}

impl Default for SplattingArgsUniform {
    fn default() -> Self {
        Self {
            gaussian_scaling: 1.0,
            max_sh_deg: 3,
            mip_splatting: false as u32,
            kernel_size: DEFAULT_KERNEL_SIZE,
            clipping_box_max: Vector4::new(f32::INFINITY, f32::INFINITY, f32::INFINITY, 0.),
            clipping_box_min: Vector4::new(
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                f32::NEG_INFINITY,
                0.,
            ),
            walltime: 0.,
            scene_center: Vector4::new(0., 0., 0., 0.),
            scene_extend: 1.,
            _pad: [0; 2],
        }
    }
}
