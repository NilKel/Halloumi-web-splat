use bytemuck::Zeroable;
use cgmath::{
    BaseNum, ElementWise, EuclideanSpace, MetricSpace, Point3, Vector2, Vector3, Vector4,
};
use half::f16;
use num_traits::Float;
use std::fmt::Debug;
use std::mem;
use wgpu::util::DeviceExt;

use crate::io::GenericGaussianPointCloud;
use crate::uniform::UniformBuffer;

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GaussianCompressed {
    pub xyz: Point3<f32>,
    pub opacity: i8,
    pub scale_factor: i8,
    pub geometry_idx: u32,
    pub sh_idx: u32,
}
unsafe impl bytemuck::Zeroable for GaussianCompressed {}
unsafe impl bytemuck::Pod for GaussianCompressed {}

impl Default for GaussianCompressed {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Default for Gaussian {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Gaussian {
    pub xyz: Point3<f32>,
    pub opacity: f16,
    _pad: f16,
    pub cov: [f16; 6],
}

unsafe impl bytemuck::Zeroable for Gaussian {}
unsafe impl bytemuck::Pod for Gaussian {}

impl Gaussian {
    pub fn new(xyz: Point3<f32>, opacity: f16, cov: [f16; 6]) -> Self {
        Self {
            xyz: xyz,
            opacity: opacity,
            cov: cov,
            _pad: f16::ZERO,
        }
    }
}

/// 2DGS surfel: 2D disk in 3D space (quaternion + 2 scales)
/// Same size as Gaussian (28 bytes) for buffer compatibility
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Surfel {
    pub xyz: Point3<f32>,     // 12 bytes
    pub opacity: f16,         // 2 bytes
    pub shape: f16,           // 2 bytes (activated beta/general kernel shape, 0 for Gaussian)
    pub scale: [f16; 2],      // 4 bytes (sx, sy)
    pub rotation: [f16; 4],   // 8 bytes (w, x, y, z)
}
// Total: 28 bytes

unsafe impl bytemuck::Zeroable for Surfel {}
unsafe impl bytemuck::Pod for Surfel {}

impl Default for Surfel {
    fn default() -> Self {
        Self::zeroed()
    }
}

impl Surfel {
    pub fn new(xyz: Point3<f32>, opacity: f16, scale: [f16; 2], rotation: [f16; 4]) -> Self {
        Self {
            xyz,
            opacity,
            shape: f16::ZERO,
            scale,
            rotation,
        }
    }

    pub fn with_shape(mut self, shape: f16) -> Self {
        self.shape = shape;
        self
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Covariance3D(pub [f16; 6]);

impl Default for Covariance3D {
    fn default() -> Self {
        Covariance3D::zeroed()
    }
}

#[allow(dead_code)]
pub struct PointCloud {
    splat_2d_buffer: wgpu::Buffer,

    bind_group: wgpu::BindGroup,
    render_bind_group: wgpu::BindGroup,
    num_points: u32,
    sh_deg: u32,
    bbox: Aabb<f32>,
    compressed: bool,
    is_2dgs: bool,

    center: Point3<f32>,
    up: Option<Vector3<f32>>,

    mip_splatting: Option<bool>,
    kernel_size: Option<f32>,
    background_color: Option<wgpu::Color>,

    // 2DGS atlas texture support
    atlas_buffer: Option<wgpu::Buffer>,       // [H, W, C] FP16 packed as u32
    atlas_rects_buffer: Option<wgpu::Buffer>, // [N, 4] f32
    atlas_width: u32,
    atlas_height: u32,
    atlas_channels: u32,
    uv_extent: f32,
    kernel_type: u32,
    // Keep surfel buffer accessible for render shader (needs means3D for viewdir)
    surfel_buffer: Option<wgpu::Buffer>,
    surfel_render_bind_group: Option<wgpu::BindGroup>,
}

impl Debug for PointCloud {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PointCloud")
            .field("num_points", &self.num_points)
            .finish()
    }
}

impl PointCloud {
    pub fn new(
        device: &wgpu::Device,
        pc: GenericGaussianPointCloud,
    ) -> Result<Self, anyhow::Error> {
        let is_2dgs = pc.is_2dgs;

        let splat_size = if is_2dgs {
            mem::size_of::<Splat2DGS>()
        } else {
            mem::size_of::<Splat>()
        };

        let splat_2d_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d splats buffer"),
            size: (pc.num_points * splat_size) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let render_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("point cloud rendering bind group"),
            layout: &Self::bind_group_layout_render(device),
            entries: &[wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_2d_buffer.as_entire_binding(),
            }],
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("3d gaussians buffer"),
            contents: pc.gaussian_buffer(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let sh_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sh coefs buffer"),
            contents: pc.sh_coefs_buffer(),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        let mut bind_group_entries = vec![
            wgpu::BindGroupEntry {
                binding: 0,
                resource: vertex_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: sh_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: splat_2d_buffer.as_entire_binding(),
            },
        ];

        let bind_group = if pc.compressed() {
            let covars_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Covariances buffer"),
                contents: bytemuck::cast_slice(pc.covars.as_ref().unwrap().as_slice()),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });
            let quantization_uniform = UniformBuffer::new(
                device,
                pc.quantization.unwrap(),
                Some("quantization uniform buffer"),
            );
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: 3,
                resource: covars_buffer.as_entire_binding(),
            });
            bind_group_entries.push(wgpu::BindGroupEntry {
                binding: 4,
                resource: quantization_uniform.buffer().as_entire_binding(),
            });

            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("point cloud bind group (compressed)"),
                layout: &Self::bind_group_layout_compressed(device),
                entries: &bind_group_entries,
            })
        } else {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("point cloud bind group"),
                layout: &Self::bind_group_layout(device),
                entries: &bind_group_entries,
            })
        };

        // 2DGS: create atlas + rects buffers and surfel render bind group
        let mut atlas_buffer = None;
        let mut atlas_rects_buffer = None;
        let mut surfel_buffer_opt = None;
        let mut surfel_render_bind_group = None;

        if is_2dgs {
            // Surfel buffer for render pass (need means3D for viewdir)
            let surfel_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("surfel buffer (render)"),
                contents: pc.gaussian_buffer(),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

            surfel_render_bind_group = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("surfel render bind group"),
                layout: &Self::bind_group_layout_surfel_render(device),
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: surfel_buf.as_entire_binding(),
                }],
            }));
            surfel_buffer_opt = Some(surfel_buf);

            // Atlas texture buffer
            if let Some(ref atlas_data) = pc.atlas_texture {
                atlas_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("atlas texture buffer"),
                    contents: atlas_data,
                    usage: wgpu::BufferUsages::STORAGE,
                }));
            }

            // Atlas rects buffer
            if let Some(ref rects) = pc.atlas_rects {
                atlas_rects_buffer = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("atlas rects buffer"),
                    contents: bytemuck::cast_slice(rects),
                    usage: wgpu::BufferUsages::STORAGE,
                }));
            }
        }

        Ok(Self {
            splat_2d_buffer,

            bind_group,
            render_bind_group,
            num_points: pc.num_points as u32,
            sh_deg: pc.sh_deg,
            compressed: pc.compressed(),
            is_2dgs,
            bbox: pc.aabb.into(),
            center: pc.center,
            up: pc.up,
            mip_splatting: pc.mip_splatting,
            kernel_size: pc.kernel_size,
            background_color: pc.background_color.map(|c| wgpu::Color {
                r: c[0] as f64,
                g: c[1] as f64,
                b: c[2] as f64,
                a: 1.,
            }),
            atlas_buffer,
            atlas_rects_buffer,
            atlas_width: pc.atlas_width,
            atlas_height: pc.atlas_height,
            atlas_channels: pc.atlas_channels,
            uv_extent: pc.uv_extent,
            kernel_type: pc.kernel_type,
            surfel_buffer: surfel_buffer_opt,
            surfel_render_bind_group,
        })
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }

    pub fn num_points(&self) -> u32 {
        self.num_points
    }

    pub fn sh_deg(&self) -> u32 {
        self.sh_deg
    }

    pub fn bbox(&self) -> &Aabb<f32> {
        &self.bbox
    }

    pub(crate) fn bind_group(&self) -> &wgpu::BindGroup {
        &self.bind_group
    }
    pub(crate) fn render_bind_group(&self) -> &wgpu::BindGroup {
        &self.render_bind_group
    }

    pub fn bind_group_layout_compressed(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud bind group layout (compressed)"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::COMPUTE,
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

    pub fn bind_group_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud float bind group layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }

    pub fn bind_group_layout_render(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("point cloud rendering bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn bind_group_layout_surfel_render(device: &wgpu::Device) -> wgpu::BindGroupLayout {
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("surfel render bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }

    pub fn mip_splatting(&self) -> Option<bool> {
        self.mip_splatting
    }
    pub fn dilation_kernel_size(&self) -> Option<f32> {
        self.kernel_size
    }

    pub fn center(&self) -> Point3<f32> {
        self.center
    }

    pub fn up(&self) -> Option<Vector3<f32>> {
        self.up
    }

    pub fn is_2dgs(&self) -> bool {
        self.is_2dgs
    }

    pub fn kernel_type(&self) -> u32 {
        self.kernel_type
    }

    pub fn atlas_width(&self) -> u32 {
        self.atlas_width
    }

    pub fn atlas_height(&self) -> u32 {
        self.atlas_height
    }

    pub fn uv_extent(&self) -> f32 {
        self.uv_extent
    }

    pub(crate) fn atlas_buffer(&self) -> Option<&wgpu::Buffer> {
        self.atlas_buffer.as_ref()
    }

    pub(crate) fn atlas_rects_buffer(&self) -> Option<&wgpu::Buffer> {
        self.atlas_rects_buffer.as_ref()
    }

    pub(crate) fn surfel_render_bind_group(&self) -> Option<&wgpu::BindGroup> {
        self.surfel_render_bind_group.as_ref()
    }
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat {
    pub v: Vector4<f16>,
    pub pos: Vector2<f16>,
    pub color: Vector4<f16>,
}

/// 2DGS splat: screen-space data for ray-disk intersection rendering.
/// Transmat stored as f32 to avoid precision loss in ray-disk intersection.
/// Total: 64 bytes (16 u32s).
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Splat2DGS {
    pub tu: [f32; 3],       // Tu (row 0 of transmat)
    pub tv: [f32; 3],       // Tv (row 1 of transmat)
    pub tw: [f32; 3],       // Tw (row 2 of transmat)
    pub opacity: f32,        // opacity
    pub pos: u32,            // NDC center x, y (f16 pair)
    pub extent: u32,         // NDC extent x, y (f16 pair)
    pub color_rg: u32,       // R, G (f16 pair)
    pub color_b_shape: u32,  // B, shape (f16 pair)
    pub gauss_id: u32,       // original Gaussian index (for texture lookup)
    pub _pad: u32,           // alignment padding
}

#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Quantization {
    pub zero_point: i32,
    pub scale: f32,
    _pad: [u32; 2],
}

impl Quantization {
    #[cfg(feature = "npz")]
    pub fn new(zero_point: i32, scale: f32) -> Self {
        Quantization {
            zero_point,
            scale,
            ..Default::default()
        }
    }
}

impl Default for Quantization {
    fn default() -> Self {
        Self {
            zero_point: 0,
            scale: 1.,
            _pad: [0, 0],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable, Default)]
pub struct GaussianQuantization {
    pub color_dc: Quantization,
    pub color_rest: Quantization,
    pub opacity: Quantization,
    pub scaling_factor: Quantization,
}

#[repr(C)]
#[derive(Zeroable, Clone, Copy, Debug, PartialEq)]
pub struct Aabb<F: Float + BaseNum> {
    pub min: Point3<F>,
    pub max: Point3<F>,
}

impl<F: Float + BaseNum> Aabb<F> {
    pub fn new(min: Point3<F>, max: Point3<F>) -> Self {
        Self { min, max }
    }

    pub fn grow(&mut self, pos: &Point3<F>) {
        self.min.x = self.min.x.min(pos.x);
        self.min.y = self.min.y.min(pos.y);
        self.min.z = self.min.z.min(pos.z);

        self.max.x = self.max.x.max(pos.x);
        self.max.y = self.max.y.max(pos.y);
        self.max.z = self.max.z.max(pos.z);
    }

    pub fn corners(&self) -> [Point3<F>; 8] {
        [
            Vector3::new(F::zero(), F::zero(), F::zero()),
            Vector3::new(F::one(), F::zero(), F::zero()),
            Vector3::new(F::zero(), F::one(), F::zero()),
            Vector3::new(F::one(), F::one(), F::zero()),
            Vector3::new(F::zero(), F::zero(), F::one()),
            Vector3::new(F::one(), F::zero(), F::one()),
            Vector3::new(F::zero(), F::one(), F::one()),
            Vector3::new(F::one(), F::one(), F::one()),
        ]
        .map(|d| self.min + self.max.to_vec().mul_element_wise(d))
    }

    pub fn unit() -> Self {
        Self {
            min: Point3::new(-F::one(), -F::one(), -F::one()),
            max: Point3::new(F::one(), F::one(), F::one()),
        }
    }

    pub fn center(&self) -> Point3<F> {
        self.min.midpoint(self.max)
    }

    /// radius of a sphere that contains the aabb
    pub fn radius(&self) -> F {
        self.min.distance(self.max) / (F::one() + F::one())
    }

    pub fn size(&self) -> Vector3<F> {
        self.max - self.min
    }

    pub fn grow_union(&mut self, other: &Aabb<F>) {
        self.min.x = self.min.x.min(other.min.x);
        self.min.y = self.min.y.min(other.min.y);
        self.min.z = self.min.z.min(other.min.z);

        self.max.x = self.max.x.max(other.max.x);
        self.max.y = self.max.y.max(other.max.y);
        self.max.z = self.max.z.max(other.max.z);
    }
}

impl Into<Aabb<f32>> for Aabb<f16> {
    fn into(self) -> Aabb<f32> {
        Aabb {
            min: self.min.map(|v| v.into()),
            max: self.max.map(|v| v.into()),
        }
    }
}
