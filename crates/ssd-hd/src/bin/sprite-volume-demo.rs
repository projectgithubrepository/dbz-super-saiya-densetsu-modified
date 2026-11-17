//! Small Windows-friendly visual proof of concept for the sprite-volume pipeline.
//! Run: cargo run --release -p ssd-hd --bin sprite-volume-demo -- [output.png]

use ssd_hd::sprite_volume::{project_perspective, PerspectiveCamera, RgbaFrame, SpriteVolume};
use std::path::PathBuf;

const SPRITE_W: usize = 32;
const SPRITE_H: usize = 40;
const SCALE: usize = 6;

fn paint_rect(
    pixels: &mut [[u8; 4]],
    depth: &mut [u8],
    x0: usize,
    y0: usize,
    x1: usize,
    y1: usize,
    color: [u8; 4],
    z: u8,
) {
    for y in y0..y1 {
        for x in x0..x1 {
            let at = y * SPRITE_W + x;
            pixels[at] = color;
            depth[at] = z;
        }
    }
}

fn sample_fighter() -> (RgbaFrame, Vec<u8>) {
    let mut pixels = vec![[0, 0, 0, 0]; SPRITE_W * SPRITE_H];
    let mut depth = vec![0; SPRITE_W * SPRITE_H];

    let hair = [245, 190, 48, 255];
    let skin = [238, 174, 126, 255];
    let gi = [220, 72, 35, 255];
    let blue = [35, 75, 190, 255];
    let dark = [30, 28, 42, 255];
    let white = [245, 238, 220, 255];

    // Spiky hair, head and face.
    paint_rect(&mut pixels, &mut depth, 12, 2, 19, 7, hair, 35);
    paint_rect(&mut pixels, &mut depth, 8, 5, 14, 10, hair, 45);
    paint_rect(&mut pixels, &mut depth, 18, 4, 24, 10, hair, 45);
    paint_rect(&mut pixels, &mut depth, 11, 8, 22, 19, skin, 135);
    paint_rect(&mut pixels, &mut depth, 9, 10, 12, 16, hair, 75);
    paint_rect(&mut pixels, &mut depth, 21, 10, 24, 16, hair, 75);
    paint_rect(&mut pixels, &mut depth, 13, 13, 15, 14, dark, 225);
    paint_rect(&mut pixels, &mut depth, 19, 13, 21, 14, dark, 225);
    paint_rect(&mut pixels, &mut depth, 16, 16, 18, 17, [180, 90, 80, 255], 230);

    // Neck, gi, undershirt, arms and wristbands.
    paint_rect(&mut pixels, &mut depth, 14, 18, 19, 23, skin, 165);
    paint_rect(&mut pixels, &mut depth, 9, 21, 23, 32, gi, 145);
    paint_rect(&mut pixels, &mut depth, 14, 21, 18, 31, blue, 205);
    paint_rect(&mut pixels, &mut depth, 4, 22, 9, 29, skin, 190);
    paint_rect(&mut pixels, &mut depth, 23, 22, 28, 29, skin, 190);
    paint_rect(&mut pixels, &mut depth, 4, 28, 9, 31, blue, 210);
    paint_rect(&mut pixels, &mut depth, 23, 28, 28, 31, blue, 210);
    paint_rect(&mut pixels, &mut depth, 11, 31, 21, 37, blue, 115);
    paint_rect(&mut pixels, &mut depth, 8, 36, 14, 40, dark, 165);
    paint_rect(&mut pixels, &mut depth, 19, 36, 25, 40, dark, 165);
    paint_rect(&mut pixels, &mut depth, 9, 36, 13, 37, white, 180);
    paint_rect(&mut pixels, &mut depth, 20, 36, 24, 37, white, 180);

    let frame = RgbaFrame::new(SPRITE_W, SPRITE_H, pixels).expect("valid built-in sample");
    (frame, depth)
}

fn set_pixel(canvas: &mut [[u8; 4]], width: usize, x: usize, y: usize, color: [u8; 4]) {
    if let Some(pixel) = canvas.get_mut(y * width + x) {
        if x < width {
            *pixel = color;
        }
    }
}

fn fill_rect(
    canvas: &mut [[u8; 4]],
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
    color: [u8; 4],
) {
    for py in y..(y + h).min(height) {
        for px in x..(x + w).min(width) {
            set_pixel(canvas, width, px, py, color);
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sprite-volume-demo.png"));

    let (source, _depth_map) = sample_fighter();
    let volume = SpriteVolume::from_silhouette(&source, 7)?;
    let projected = project_perspective(&volume, PerspectiveCamera {
        yaw: 0.78, pitch: 0.32, distance: 220.0, focal_length: 150.0,
        width: 96, height: 96, near_plane: 0.1,
    })?;

    // Two panels: original pixel art on the left, stacked volume on the right.
    let width = 760_usize;
    let height = 390_usize;
    let mut canvas = vec![[18, 22, 34, 255]; width * height];

    // Panel backgrounds.
    fill_rect(&mut canvas, width, height, 28, 28, 250, 330, [29, 36, 54, 255]);
    fill_rect(&mut canvas, width, height, 310, 28, 420, 330, [29, 36, 54, 255]);

    // Flat source sprite.
    for y in 0..source.height {
        for x in 0..source.width {
            let color = source.pixels[y * source.width + x];
            if color[3] != 0 {
                fill_rect(
                    &mut canvas, width, height,
                    75 + x * SCALE, 75 + y * SCALE,
                    SCALE, SCALE, color,
                );
            }
        }
    }

    // Perspective render of the silhouette-derived volume, using the same projector as the game.
    let scale = 6_usize;
    let origin_x = 390_usize;
    let origin_y = 45_usize;
    for y in 0..projected.height {
        for x in 0..projected.width {
            let color = projected.pixels[y * projected.width + x];
            if color[3] != 0 {
                fill_rect(
                    &mut canvas, width, height,
                    origin_x + x * scale, origin_y + y * scale,
                    scale, scale, color,
                );
            }
        }
    }

    let rgba: Vec<u8> = canvas.iter().flat_map(|pixel| pixel.iter().copied()).collect();
    ssd_hd::png::write_rgba(&output, width as u32, height as u32, &rgba)?;
    println!("Wrote {}", output.display());
    println!("Sprite: {}x{} | depth slices: {} | voxels: {}",
        volume.width, volume.height, volume.slices.len(), volume.voxel_count());
    println!("This is a generated test fighter, rendered with the live silhouette-stack projector.");
    Ok(())
}
