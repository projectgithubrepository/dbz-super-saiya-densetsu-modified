//! Small Windows-friendly visual proof of concept for the sprite-volume pipeline.
//! Run: cargo run --release -p ssd-hd --bin sprite-volume-demo -- [output.png]

use ssd_hd::sprite_volume::{RgbaFrame, SpriteVolume};
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

    let (source, depth_map) = sample_fighter();
    let volume = SpriteVolume::from_depth_map(&source, &depth_map, 12, 4)?;

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

    // Isometric-ish stack: every slice is shifted, so the thickness is visible as real layers.
    let scale = 6_usize;
    let shift_x = 3_usize;
    let shift_y = 2_usize;
    let origin_x = 390_usize;
    let origin_y = 90_usize;
    for slice in &volume.slices {
        let shade = 150_u16 + (slice.z as u16 * 105 / (volume.slices.len().saturating_sub(1).max(1) as u16));
        for y in 0..slice.height {
            for x in 0..slice.width {
                let mut color = slice.pixels[y * slice.width + x];
                if color[3] == 0 {
                    continue;
                }
                color[0] = (u16::from(color[0]) * shade / 255) as u8;
                color[1] = (u16::from(color[1]) * shade / 255) as u8;
                color[2] = (u16::from(color[2]) * shade / 255) as u8;
                let px = origin_x + x * scale + slice.z * shift_x;
                let py = origin_y + y * scale + (volume.slices.len() - 1 - slice.z) * shift_y;
                fill_rect(&mut canvas, width, height, px, py, scale, scale, color);
            }
        }
    }

    let rgba: Vec<u8> = canvas.iter().flat_map(|pixel| pixel.iter().copied()).collect();
    ssd_hd::png::write_rgba(&output, width as u32, height as u32, &rgba)?;
    println!("Wrote {}", output.display());
    println!("Sprite: {}x{} | depth slices: {} | voxels: {}",
        volume.width, volume.height, volume.slices.len(), volume.voxel_count());
    println!("This is a generated test fighter, not extracted game artwork.");
    Ok(())
}
