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
}

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
    /// Load atlas textures from a NATL binary file.
    /// Format: [magic "NATL" 4B] [W: u32] [H: u32] [C: u32] [kernel_type: u32]
    ///         [N: u32] [uv_extent: f32] [pad: u32]
    ///         [rects: N*4 f32] [atlas: H*W*C FP16]
    pub fn load_atlas_from_file(&mut self, path: &Path) -> anyhow::Result<()> {
        use std::io::Read as _;
        let mut file = std::fs::File::open(path)?;
        let mut magic = [0u8; 4];
        file.read_exact(&mut magic)?;
        if &magic != b"NATL" {
            return Err(anyhow::anyhow!("Invalid atlas file magic (expected NATL)"));
        }
        let mut header = [0u8; 28]; // 7 × u32
        file.read_exact(&mut header)?;
        let w = u32::from_le_bytes([header[0], header[1], header[2], header[3]]);
        let h = u32::from_le_bytes([header[4], header[5], header[6], header[7]]);
        let c = u32::from_le_bytes([header[8], header[9], header[10], header[11]]);
        let kernel_type = u32::from_le_bytes([header[12], header[13], header[14], header[15]]);
        let n = u32::from_le_bytes([header[16], header[17], header[18], header[19]]) as usize;
        let uv_extent = f32::from_le_bytes([header[20], header[21], header[22], header[23]]);
        // header[24..28] is padding

        if n != self.num_points {
            return Err(anyhow::anyhow!(
                "Atlas has {} rects but PLY has {} points",
                n, self.num_points
            ));
        }

        // Read rects: N * 4 * 4 bytes (f32)
        let rects_size = n * 4 * 4;
        let mut rects_bytes = vec![0u8; rects_size];
        file.read_exact(&mut rects_bytes)?;
        let rects: Vec<f32> = bytemuck::cast_slice(&rects_bytes).to_vec();

        // Read atlas: H * W * C * 2 bytes (FP16)
        let atlas_size = h as usize * w as usize * c as usize * 2;
        let mut atlas_data = vec![0u8; atlas_size];
        file.read_exact(&mut atlas_data)?;

        log::info!(
            "loaded atlas {}x{}x{}, {} rects, kernel_type={}, uv_extent={} ({:.1} MB)",
            w, h, c, n, kernel_type, uv_extent,
            (rects_size + atlas_size) as f64 / 1e6
        );

        self.atlas_texture = Some(atlas_data);
        self.atlas_rects = Some(rects);
        self.atlas_width = w;
        self.atlas_height = h;
        self.atlas_channels = c;
        self.uv_extent = uv_extent;
        self.kernel_type = kernel_type;
        Ok(())
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
