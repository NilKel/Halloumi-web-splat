use std::{fs::File, path::PathBuf, time::Duration};

use cgmath::Vector2;
use half::f16;
use web_splats::{
    io::GenericGaussianPointCloud, GaussianRenderer, PerspectiveCamera, PointCloud, Scene,
    SplattingArgs, Split, WGPUContext,
};
use web_splats::Camera;

const SPLAT_STRIDE: usize = 80;

async fn read_buffer_bytes(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
    size: u64,
) -> Vec<u8> {
    let dst = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("debug readback buffer"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("debug readback encoder"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &dst, 0, size);
    let submission = queue.submit([encoder.finish()]);

    let slice = dst.slice(..);
    let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
    slice.map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();
    rx.receive().await.unwrap().unwrap();
    let view = slice.get_mapped_range();
    let bytes = view.to_vec();
    drop(view);
    dst.unmap();
    bytes
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn u32_at(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn unpack2x16(word: u32) -> [f32; 2] {
    let lo = f16::from_bits((word & 0xffff) as u16).to_f32();
    let hi = f16::from_bits((word >> 16) as u16).to_f32();
    [lo, hi]
}

fn print_splat(bytes: &[u8], offset: usize) {
    let tu = [
        f32_at(bytes, offset + 0),
        f32_at(bytes, offset + 4),
        f32_at(bytes, offset + 8),
    ];
    let tv = [
        f32_at(bytes, offset + 12),
        f32_at(bytes, offset + 16),
        f32_at(bytes, offset + 20),
    ];
    let tw = [
        f32_at(bytes, offset + 24),
        f32_at(bytes, offset + 28),
        f32_at(bytes, offset + 32),
    ];
    let opacity = f32_at(bytes, offset + 36);
    let center = unpack2x16(u32_at(bytes, offset + 40));
    let extent = unpack2x16(u32_at(bytes, offset + 44));
    let color_rg = unpack2x16(u32_at(bytes, offset + 48));
    let color_b_shape = unpack2x16(u32_at(bytes, offset + 52));
    let gauss_id = u32_at(bytes, offset + 56);
    let depth_plane = [
        f32_at(bytes, offset + 60),
        f32_at(bytes, offset + 64),
        f32_at(bytes, offset + 68),
    ];
    println!("splat.offset_bytes={offset}");
    println!("  Tu={tu:?}");
    println!("  Tv={tv:?}");
    println!("  Tw={tw:?}");
    println!("  opacity={opacity:.9}");
    println!("  center_pix={center:?}");
    println!("  extent_pix={extent:?}");
    println!(
        "  color_shape=[{:.9}, {:.9}, {:.9}, {:.9}]",
        color_rg[0], color_rg[1], color_b_shape[0], color_b_shape[1]
    );
    println!("  gauss_id={gauss_id}");
    println!("  depth_plane={depth_plane:?}");
}

fn print_mat4(name: &str, m: cgmath::Matrix4<f32>) {
    println!("{name}:");
    for r in 0..4 {
        println!(
            "  [{:.9}, {:.9}, {:.9}, {:.9}]",
            m[r][0], m[r][1], m[r][2], m[r][3]
        );
    }
}

#[pollster::main]
async fn main() {
    let args = std::env::args().collect::<Vec<_>>();
    if args.len() != 3 {
        eprintln!("usage: debug_2dgs_stage <baked-or-debug.ply> <cameras.json>");
        std::process::exit(2);
    }
    let ply_path = PathBuf::from(&args[1]);
    let scene_path = PathBuf::from(&args[2]);

    let scene = Scene::from_json(File::open(&scene_path).unwrap()).unwrap();
    let wgpu_context = WGPUContext::new_instance().await;
    let device = &wgpu_context.device;
    let queue = &wgpu_context.queue;

    let pc_raw = GenericGaussianPointCloud::load(File::open(&ply_path).unwrap()).unwrap();
    let pc = PointCloud::new(device, queue, pc_raw).unwrap();
    let mut renderer =
        GaussianRenderer::new_2dgs(device, queue, wgpu::TextureFormat::Rgba16Float, pc.sh_deg(), &pc)
            .await;

    let chosen = scene
        .cameras(Some(Split::Test))
        .first()
        .cloned()
        .or_else(|| scene.cameras(None).first().cloned())
        .expect("scene has at least one camera");

    let mut resolution: Vector2<u32> = Vector2::new(chosen.width, chosen.height);
    if resolution.x > 1600 {
        let s = resolution.x as f32 / 1600.0;
        resolution.x = 1600;
        resolution.y = (resolution.y as f32 / s) as u32;
    }

    let mut camera: PerspectiveCamera = chosen.into();
    if !pc.is_2dgs() {
        camera.fit_near_far(pc.bbox());
    }
    print_mat4("camera.view_raw_indexed", camera.view_matrix());
    print_mat4("camera.proj_raw_indexed", camera.proj_matrix());

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("debug preprocess encoder"),
    });
    renderer.prepare(
        &mut encoder,
        device,
        queue,
        &pc,
        SplattingArgs {
            camera,
            viewport: resolution,
            gaussian_scaling: 1.0,
            max_sh_deg: pc.sh_deg(),
            mip_splatting: None,
            kernel_size: None,
            clipping_box: None,
            walltime: Duration::from_secs(100),
            scene_center: None,
            scene_extend: None,
            background_color: wgpu::Color::BLACK,
        },
        &mut None,
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .unwrap();

    let visible = renderer.num_visible_points(device, queue).await;
    println!("viewport={:?}", resolution);
    println!("pc.num_points={}", pc.num_points());
    println!("visible.keys_size={visible}");

    let sort_info = read_buffer_bytes(
        device,
        queue,
        renderer.debug_sort_info_buffer().unwrap(),
        16,
    )
    .await;
    println!(
        "sort_info=[{}, {}, {}, {}]",
        u32_at(&sort_info, 0),
        u32_at(&sort_info, 4),
        u32_at(&sort_info, 8),
        u32_at(&sort_info, 12)
    );

    let indices = read_buffer_bytes(
        device,
        queue,
        renderer.debug_sort_indices_buffer().unwrap(),
        (pc.num_points() as u64) * 4,
    )
    .await;
    let first_index = if visible > 0 { u32_at(&indices, 0) } else { u32::MAX };
    println!("sorted_indices.first={first_index}");

    let splats = read_buffer_bytes(
        device,
        queue,
        pc.debug_splat_2d_buffer(),
        (pc.num_points() as u64) * SPLAT_STRIDE as u64,
    )
    .await;
    if visible > 0 {
        let sample_count = visible.min(8);
        println!("sorted_samples.front:");
        for i in 0..sample_count {
            let idx = u32_at(&indices, i as usize * 4);
            let offset = idx as usize * SPLAT_STRIDE;
            let depth = f32_at(&splats, offset + 68);
            let center = unpack2x16(u32_at(&splats, offset + 40));
            let extent = unpack2x16(u32_at(&splats, offset + 44));
            println!("  rank={i} slot={idx} depth={depth:.6} center={center:?} extent={extent:?}");
        }
        println!("sorted_samples.back:");
        let start = visible.saturating_sub(sample_count);
        for i in start..visible {
            let idx = u32_at(&indices, i as usize * 4);
            let offset = idx as usize * SPLAT_STRIDE;
            let depth = f32_at(&splats, offset + 68);
            let center = unpack2x16(u32_at(&splats, offset + 40));
            let extent = unpack2x16(u32_at(&splats, offset + 44));
            println!("  rank={i} slot={idx} depth={depth:.6} center={center:?} extent={extent:?}");
        }
    }
    if visible > 0 {
        print_splat(&splats, first_index as usize * SPLAT_STRIDE);
    }
    for i in 0..pc.num_points().min(4) {
        println!("raw_splat_slot[{i}]");
        print_splat(&splats, i as usize * SPLAT_STRIDE);
    }
}
