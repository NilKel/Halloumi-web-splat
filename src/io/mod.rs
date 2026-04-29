#[cfg(feature = "npz")]
use std::io::BufReader;
use std::io::{Read, Seek};
use std::path::Path;

use bytemuck::Zeroable;
use cgmath::{Array, EuclideanSpace, InnerSpace, Point3, Vector3};
use half::f16;

use crate::pointcloud::{Aabb, Covariance3D, Gaussian, GaussianCompressed, GaussianQuantization, Surfel};

#[cfg(feature = "npz")]
use self::npz::NpzReader;

use self::ply::PlyReader;

#[cfg(feature = "npz")]
pub mod npz;
pub mod ply;

pub trait PointCloudReader {
    fn read(&mut self) -> Result<GenericGaussianPointCloud, anyhow::Error>;

    fn magic_bytes() -> &'static [u8];
    fn file_ending() -> &'static str;
}

pub struct GenericGaussianPointCloud {
    gaussians: Vec<u8>,
    sh_coefs: Vec<u8>,
    compressed: bool,
    pub is_2dgs: bool,
    pub covars: Option<Vec<Covariance3D>>,
    pub quantization: Option<GaussianQuantization>,
    pub sh_deg: u32,
    pub num_points: usize,
    pub kernel_size: Option<f32>,
    pub mip_splatting: Option<bool>,
    pub background_color: Option<[f32; 3]>,

    pub up: Option<Vector3<f32>>,
    pub center: Point3<f32>,
    pub aabb: Aabb<f32>,

    // 2DGS atlas texture data
    pub atlas_texture: Option<Vec<u8>>,   // [H, W, C] FP16 data
    pub atlas_rects: Option<Vec<f32>>,    // [N, 4] as flat f32: (u0_px, v0_px, w_px, h_px)
    pub atlas_width: u32,
    pub atlas_height: u32,
    pub atlas_channels: u32,
    pub uv_extent: f32,
    pub kernel_type: u32,

    // Training-time activation + Compact Box knobs — must match the CUDA bake
    // (diff_surfel_bake_render forward.cu uses these via d_sh_bias / d_res_bias / d_compact_mult).
    pub sh_bias: f32,          // additive constant on SH eval before ReLU (default 0.5)
    pub res_bias: f32,          // additive constant on final (SH+residual+SB) before ReLU (default 0.0)
    pub compact_mult: f32,      // FastGS Compact Box multiplier on AdR r_lp (default 1.0)

    // Atlas texel format. 0 = FP16 RGB (legacy, 6 B/texel). 1 = UINT8 RGBA with
    // per-atlas linear dequant: residual = u8_norm * atlas_scale + atlas_offset
    // (matches CUDA's cudaReadModeNormalizedFloat path in forward.cu). 4 B/texel
    // and wgpu can do hardware bilinear via unpack4x8unorm.
    pub atlas_format: u32,
    pub atlas_scale: f32,       // dequant multiplier (UINT8 path; unused for FP16)
    pub atlas_offset: f32,      // dequant additive offset (UINT8 path; unused for FP16)

    // Spherical-Beta (SB) view-dependent lobes. [N, sb_number, 6] f32:
    // per lobe: (r, g, b, theta, phi, beta_raw). Flat length = N * sb_number * 6.
    pub sb_params: Option<Vec<f32>>,
    pub sb_number: u32,         // number of SB lobes per Gaussian (0 = SB disabled)
}

/// Atlas texel format constants (stored in NAT2 and TexParams).
pub const ATLAS_FORMAT_FP16_RGB: u32 = 0;
pub const ATLAS_FORMAT_UINT8_RGBA: u32 = 1;

impl GenericGaussianPointCloud {
    pub fn load<'a, R: Read + Seek>(f: R) -> Result<Self, anyhow::Error> {
        let mut signature: [u8; 4] = [0; 4];
        let mut f = f;
        f.read_exact(&mut signature)?;
        f.rewind()?;
        if signature.starts_with(PlyReader::<R>::magic_bytes()) {
            let mut ply_reader = PlyReader::new(f)?;
            return ply_reader.read();
        }
        #[cfg(feature = "npz")]
        if signature.starts_with(NpzReader::<R>::magic_bytes()) {
            let mut reader = BufReader::new(f);
            let mut npz_reader = NpzReader::new(&mut reader)?;
            return npz_reader.read();
        }
        return Err(anyhow::anyhow!("Unknown file format"));
    }

    fn new(
        gaussians: Vec<Gaussian>,
        sh_coefs: Vec<[[f16; 3]; 16]>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::zeroed();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let (center, mut up) = plane_from_points(
            gaussians
                .iter()
                .map(|g| g.xyz.cast().unwrap())
                .collect::<Vec<Point3<f32>>>()
                .as_slice(),
        );

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up: up,
            center,
            aabb: bbox,
            compressed: false,
            is_2dgs: false,
            atlas_texture: None,
            atlas_rects: None,
            atlas_width: 0,
            atlas_height: 0,
            atlas_channels: 0,
            uv_extent: 4.0,
            kernel_type: 0,
            sh_bias: 0.5,
            res_bias: 0.0,
            compact_mult: 1.0,
            atlas_format: ATLAS_FORMAT_FP16_RGB,
            atlas_scale: 1.0,
            atlas_offset: 0.0,
            sb_params: None,
            sb_number: 0,
        }
    }

    /// Create from 2DGS surfel data
    pub(crate) fn new_2dgs(
        surfels: Vec<Surfel>,
        sh_coefs: Vec<[[f16; 3]; 16]>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::zeroed();
        for v in &surfels {
            bbox.grow(&v.xyz);
        }

        let (center, mut up) = plane_from_points(
            surfels
                .iter()
                .map(|g| g.xyz.cast().unwrap())
                .collect::<Vec<Point3<f32>>>()
                .as_slice(),
        );

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&surfels).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars: None,
            quantization: None,
            up,
            center,
            aabb: bbox,
            compressed: false,
            is_2dgs: true,
            atlas_texture: None,
            atlas_rects: None,
            atlas_width: 0,
            atlas_height: 0,
            atlas_channels: 0,
            uv_extent: 4.0,
            kernel_type: 0,
            sh_bias: 0.5,
            res_bias: 0.0,
            compact_mult: 1.0,
            atlas_format: ATLAS_FORMAT_FP16_RGB,
            atlas_scale: 1.0,
            atlas_offset: 0.0,
            sb_params: None,
            sb_number: 0,
        }
    }

    #[cfg(feature = "npz")]
    fn new_compressed(
        gaussians: Vec<GaussianCompressed>,
        sh_coefs: Vec<u8>,
        sh_deg: u32,
        num_points: usize,
        kernel_size: Option<f32>,
        mip_splatting: Option<bool>,
        background_color: Option<[f32; 3]>,
        covars: Option<Vec<Covariance3D>>,
        quantization: Option<GaussianQuantization>,
    ) -> Self {
        let mut bbox: Aabb<f32> = Aabb::unit();
        for v in &gaussians {
            bbox.grow(&v.xyz);
        }

        let (center, mut up) = plane_from_points(
            gaussians
                .iter()
                .map(|g| g.xyz.cast().unwrap())
                .collect::<Vec<Point3<f32>>>()
                .as_slice(),
        );

        if bbox.radius() < 10. {
            up = None;
        }
        Self {
            gaussians: bytemuck::cast_slice(&gaussians).to_vec(),
            sh_coefs: bytemuck::cast_slice(&sh_coefs).to_vec(),
            sh_deg,
            num_points,
            kernel_size,
            mip_splatting,
            background_color,
            covars,
            quantization,
            up: up,
            center,
            aabb: bbox,
            compressed: true,
            is_2dgs: false,
            atlas_texture: None,
            atlas_rects: None,
            atlas_width: 0,
            atlas_height: 0,
            atlas_channels: 0,
            uv_extent: 4.0,
            kernel_type: 0,
            sh_bias: 0.5,
            res_bias: 0.0,
            compact_mult: 1.0,
            atlas_format: ATLAS_FORMAT_FP16_RGB,
            atlas_scale: 1.0,
            atlas_offset: 0.0,
            sb_params: None,
            sb_number: 0,
        }
    }

    pub fn gaussians(&self) -> anyhow::Result<&[Gaussian]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    pub fn gaussians_compressed(&self) -> anyhow::Result<&[GaussianCompressed]> {
        if self.compressed {
            Err(anyhow::anyhow!("Gaussians are compressed"))
        } else {
            Ok(bytemuck::cast_slice(&self.gaussians))
        }
    }

    pub fn sh_coefs_buffer(&self) -> &[u8] {
        &self.sh_coefs
    }

    pub fn gaussian_buffer(&self) -> &[u8] {
        &self.gaussians
    }

    pub fn compressed(&self) -> bool {
        self.compressed
    }
}

impl GenericGaussianPointCloud {
    /// Read the body of a NATL/NAT2 stream (after the 4-byte magic).
    ///
    /// NATL (v1, legacy) header, 28 B:
    ///   [W H C kernel_type] [N uv_extent _pad] (7 × 4B).
    ///   Atlas payload = H*W*C*2 B (FP16 RGB). No SB block.
    ///
    /// NAT2 (v2, current) header, 64 B post-magic, laid out as 16 × 4B words:
    ///   0: W, 1: H, 2: C, 3: kernel_type,
    ///   4: N, 5: uv_extent (f32), 6: sb_number, 7: atlas_format,
    ///   8: sh_bias (f32), 9: res_bias (f32), 10: compact_mult (f32), 11: _pad0,
    ///   12: atlas_scale (f32), 13: atlas_offset (f32), 14: _pad1, 15: _pad2.
    /// Atlas payload size depends on atlas_format:
    ///   FP16_RGB   (0): H*W*C*2 B
    ///   UINT8_RGBA (1): H*W*4   B
    /// SB block follows atlas when sb_number > 0: N*sb_number*6*4 B of f32.
    fn load_atlas_body<R: Read>(&mut self, reader: &mut R, is_v2: bool) -> anyhow::Result<()> {
        let (w, h, c, kernel_type, n, uv_extent);
        let (sb_number, atlas_format, sh_bias, res_bias, compact_mult, atlas_scale, atlas_offset);

        if is_v2 {
            let mut header = [0u8; 64];
            reader.read_exact(&mut header)?;
            let word = |i: usize| [header[i], header[i+1], header[i+2], header[i+3]];
            w            = u32::from_le_bytes(word(0));
            h            = u32::from_le_bytes(word(4));
            c            = u32::from_le_bytes(word(8));
            kernel_type  = u32::from_le_bytes(word(12));
            n            = u32::from_le_bytes(word(16)) as usize;
            uv_extent    = f32::from_le_bytes(word(20));
            sb_number    = u32::from_le_bytes(word(24));
            atlas_format = u32::from_le_bytes(word(28));
            sh_bias      = f32::from_le_bytes(word(32));
            res_bias     = f32::from_le_bytes(word(36));
            compact_mult = f32::from_le_bytes(word(40));
            // word(44) = _pad0
            atlas_scale  = f32::from_le_bytes(word(48));
            atlas_offset = f32::from_le_bytes(word(52));
            // word(56), word(60) = _pad1, _pad2
        } else {
            let mut header = [0u8; 28];
            reader.read_exact(&mut header)?;
            let word = |i: usize| [header[i], header[i+1], header[i+2], header[i+3]];
            w            = u32::from_le_bytes(word(0));
            h            = u32::from_le_bytes(word(4));
            c            = u32::from_le_bytes(word(8));
            kernel_type  = u32::from_le_bytes(word(12));
            n            = u32::from_le_bytes(word(16)) as usize;
            uv_extent    = f32::from_le_bytes(word(20));
            // Legacy defaults — matches the old hardcoded Halloumi activation path.
            sb_number    = 0;
            atlas_format = ATLAS_FORMAT_FP16_RGB;
            sh_bias      = 0.5;
            res_bias     = 0.0;
            compact_mult = 1.0;
            atlas_scale  = 1.0;
            atlas_offset = 0.0;
        }

        if n != self.num_points {
            return Err(anyhow::anyhow!(
                "Atlas has {} rects but PLY has {} points",
                n, self.num_points
            ));
        }

        let rects_size = n * 4 * 4;
        let mut rects_bytes = vec![0u8; rects_size];
        reader.read_exact(&mut rects_bytes)?;
        let rects: Vec<f32> = bytemuck::cast_slice(&rects_bytes).to_vec();

        let atlas_size = match atlas_format {
            ATLAS_FORMAT_FP16_RGB   => h as usize * w as usize * c as usize * 2,
            ATLAS_FORMAT_UINT8_RGBA => h as usize * w as usize * 4,
            _ => return Err(anyhow::anyhow!(
                "Unknown atlas_format {}, expected 0 (FP16_RGB) or 1 (UINT8_RGBA)",
                atlas_format
            )),
        };
        let mut atlas_data = vec![0u8; atlas_size];
        reader.read_exact(&mut atlas_data)?;

        let mut sb_params: Option<Vec<f32>> = None;
        let mut sb_bytes = 0usize;
        if sb_number > 0 {
            sb_bytes = n * sb_number as usize * 6 * 4;
            let mut sb_raw = vec![0u8; sb_bytes];
            reader.read_exact(&mut sb_raw)?;
            sb_params = Some(bytemuck::cast_slice(&sb_raw).to_vec());
        }

        log::info!(
            "loaded atlas {}x{}x{} (fmt={}), {} rects, kernel_type={}, uv_extent={}, \
             sb_number={}, sh_bias={}, res_bias={}, compact_mult={}, \
             atlas_scale={}, atlas_offset={} ({:.1} MB)",
            w, h, c, if atlas_format == ATLAS_FORMAT_UINT8_RGBA { "uint8_rgba" } else { "fp16_rgb" },
            n, kernel_type, uv_extent, sb_number,
            sh_bias, res_bias, compact_mult, atlas_scale, atlas_offset,
            (rects_size + atlas_size + sb_bytes) as f64 / 1e6
        );

        self.atlas_texture = Some(atlas_data);
        self.atlas_rects = Some(rects);
        self.atlas_width = w;
        self.atlas_height = h;
        self.atlas_channels = c;
        self.uv_extent = uv_extent;
        self.kernel_type = kernel_type;
        self.sh_bias = sh_bias;
        self.res_bias = res_bias;
        self.compact_mult = compact_mult;
        self.atlas_format = atlas_format;
        self.atlas_scale = atlas_scale;
        self.atlas_offset = atlas_offset;
        self.sb_params = sb_params;
        self.sb_number = sb_number;
        Ok(())
    }

    /// Load atlas textures from a binary file. Accepts NATL (v1) and NAT2 (v2).
    pub fn load_atlas_from_file(&mut self, path: &Path) -> anyhow::Result<()> {
        let mut file = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        let is_v2 = match &magic {
            b"NAT2" => true,
            b"NATL" => false,
            _ => return Err(anyhow::anyhow!("Invalid atlas file magic (expected NATL or NAT2)")),
        };
        self.load_atlas_body(&mut file, is_v2)
    }

    /// Load atlas textures from raw bytes (for WASM). Accepts NATL (v1) and NAT2 (v2).
    pub fn load_atlas_from_bytes(&mut self, data: &[u8]) -> anyhow::Result<()> {
        let mut cursor = std::io::Cursor::new(data);
        let mut magic = [0u8; 4];
        cursor.read_exact(&mut magic)?;
        let is_v2 = match &magic {
            b"NAT2" => true,
            b"NATL" => false,
            _ => return Err(anyhow::anyhow!("Invalid atlas file magic (expected NATL or NAT2)")),
        };
        self.load_atlas_body(&mut cursor, is_v2)
    }
}

// Fit a plane to a collection of points.
// Fast, and accurate to within a few degrees.
// Returns None if the points do not span a plane.
// see http://www.ilikebigbits.com/2017_09_25_plane_from_points_2.html
fn plane_from_points(points: &[Point3<f32>]) -> (Point3<f32>, Option<Vector3<f32>>) {
    let n = points.len();

    let mut sum = Point3 {
        x: 0.0f32,
        y: 0.0f32,
        z: 0.0f32,
    };
    for p in points {
        sum = &sum + p.to_vec();
    }
    let centroid = &sum * (1.0 / (n as f32));
    if n < 3 {
        return (centroid, None);
    }

    // Calculate full 3x3 covariance matrix, excluding symmetries:
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut xz = 0.0;
    let mut yy = 0.0;
    let mut yz = 0.0;
    let mut zz = 0.0;

    for p in points {
        let r = p - centroid;
        xx += r.x * r.x;
        xy += r.x * r.y;
        xz += r.x * r.z;
        yy += r.y * r.y;
        yz += r.y * r.z;
        zz += r.z * r.z;
    }

    xx /= n as f32;
    xy /= n as f32;
    xz /= n as f32;
    yy /= n as f32;
    yz /= n as f32;
    zz /= n as f32;

    let mut weighted_dir = Vector3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    {
        let det_x = yy * zz - yz * yz;
        let axis_dir = Vector3 {
            x: det_x,
            y: xz * yz - xy * zz,
            z: xy * yz - xz * yy,
        };
        let mut weight = det_x * det_x;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    {
        let det_y = xx * zz - xz * xz;
        let axis_dir = Vector3 {
            x: xz * yz - xy * zz,
            y: det_y,
            z: xy * xz - yz * xx,
        };
        let mut weight = det_y * det_y;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    {
        let det_z = xx * yy - xy * xy;
        let axis_dir = Vector3 {
            x: xy * yz - xz * yy,
            y: xy * xz - yz * xx,
            z: det_z,
        };
        let mut weight = det_z * det_z;
        if weighted_dir.dot(axis_dir) < 0.0 {
            weight = -weight;
        }
        weighted_dir += &axis_dir * weight;
    }

    let mut normal = weighted_dir.normalize();

    if normal.dot(Vector3::unit_y()) < 0. {
        normal = -normal;
    }
    if normal.is_finite() {
        (centroid, Some(normal))
    } else {
        (centroid, None)
    }
}
