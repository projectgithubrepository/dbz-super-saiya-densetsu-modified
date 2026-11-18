//! The Enhanced picture: the frame the PPU drew, drawn again at a whole multiple of its resolution
//! from the same render log and nothing else (`--hd-scale`).
//!
//! The 1x renderer runs first, and while it draws it records what the picture needs of each pixel
//! ([`Capture`]): whether a Mode 7 plane is the top layer there, the colour math of the pixel, and
//! the state of its line the plane was drawn with. The picture is then the look ([`Look`]) over the
//! 1x frame, except on the plane pixels, which are sampled again at the picture's resolution: the
//! matrix is applied to each finer position, so the plane's diagonals are drawn on that grid (the
//! HD Mode 7). Every pass works row by row on the rayon pool and reads only what was recorded, so
//! the output does not depend on the threads.

use crate::art;
use crate::mode7::Plane;
use crate::screen::{Plan, convert, direct_color};
use crate::{Frame, Renderer, Unsupported, WIDTH};
use rayon::prelude::*;
use ssd_oracle::{Color, FrameLog, LineRegs};

/// The look of the Enhanced picture where it is not a Mode 7 plane.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Look {
    /// Pixel art made bigger: Scale2x (EPX) on the power-of-two part of the scale, then whole
    /// pixels, so every block is uniform and the diagonals are smoothed.
    #[default]
    Crisp,
    /// A smooth interpolation: a bicubic (Catmull-Rom) filter over the 1x frame.
    Smooth,
    /// No filter: every pixel of the 1x frame is a block of whole pixels (nearest neighbour).
    Pixel,
}

/// How an Enhanced picture is made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdOptions {
    /// Pixels of the picture per pixel of the PPU's frame, 1 to [`crate::MAX_SCALE`].
    pub scale: u32,
    pub look: Look,
}

/// A rectangle of an [`HdFrame`], in its pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// An Enhanced picture: `width * height` RGBA pixels, row by row, alpha 255.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HdFrame {
    pub rgba: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub scale: u32,
    /// Where the PPU's own picture sits in the frame: the whole frame here. Overlays keep their
    /// coordinates in this rectangle when the frame is wider than the picture (widescreen).
    pub logical: Rect,
}

impl Drop for HdFrame {
    /// The picture's bytes go back to the pool the next picture is made in (see [`crate::pool`]).
    fn drop(&mut self) {
        crate::pool::BYTES.give(std::mem::take(&mut self.rgba));
    }
}

/// The Mode 7 state of one line: the entry of the log it belongs to, the plane's parameters, the
/// CGRAM it reads and the VRAM copy it reads (`None`: the log's own VRAM).
#[derive(Clone, Copy)]
struct State {
    entry: usize,
    plane: Plane,
    cgram: [u16; 256],
    vram: Option<usize>,
}

/// What the picture needs of one pixel of the 1x frame.
#[derive(Clone, Copy)]
struct Record {
    /// 1 for BG1, 2 for BG2 (EXTBG), when a Mode 7 plane is the top layer; 0 otherwise.
    plane: u8,
    /// The index of the pixel's line in [`Capture::states`]; meaningful when `plane` is not 0.
    state: u32,
    plan: Plan,
}

impl Record {
    const NONE: Self = Self { plane: 0, state: 0, plan: Plan { force_black: false, math: None } };
}

/// What the 1x renderer records while it draws a frame for the Enhanced picture. It keeps its
/// buffers from one frame to the next.
#[derive(Default)]
pub(crate) struct Capture {
    /// One record per pixel of the frame, row by row.
    records: Vec<Record>,
    states: Vec<State>,
    /// Copies of VRAM, for the frames that write it; the first `copies` are in use.
    vram: Vec<Vec<u16>>,
    copies: usize,
    /// The copy that holds the VRAM the lines being drawn read, while it is current.
    current: Option<usize>,
    /// The frame writes no VRAM, so every line reads the log's VRAM.
    shared_vram: bool,
}

impl Capture {
    fn begin(&mut self, log: &FrameLog) {
        self.records.clear();
        self.records.resize(usize::from(log.height) * WIDTH, Record::NONE);
        self.states.clear();
        self.copies = 0;
        self.current = None;
        self.shared_vram = log.vram_writes.is_empty();
    }

    /// The VRAM changed: the next Mode 7 line copies it again.
    pub(crate) fn vram_changed(&mut self) {
        self.current = None;
    }

    /// Records the state of a Mode 7 line and returns its index.
    fn push_state(&mut self, entry: usize, plane: Plane, cgram: &[u16; 256], vram: &[u16]) -> u32 {
        let vram = if self.shared_vram {
            None
        } else if let Some(copy) = self.current {
            Some(copy)
        } else {
            let copy = self.copies;
            self.copies += 1;
            if copy == self.vram.len() {
                self.vram.push(vram.to_vec());
            } else {
                self.vram[copy].copy_from_slice(vram);
            }
            self.current = Some(copy);
            Some(copy)
        };
        self.states.push(State { entry, plane, cgram: *cgram, vram });
        (self.states.len() - 1) as u32
    }
}

impl Renderer {
    /// Records, for the pixels `from..` of line `line` drawn by entry `index` of the log, what the
    /// Enhanced picture needs (see [`Capture`]). Called after the line is drawn, blank or not: a
    /// blank line has no plane.
    pub(crate) fn capture_line(
        &mut self,
        index: usize,
        line: usize,
        from: usize,
        regs: &LineRegs,
        vram: &[u16],
    ) {
        let mut capture = std::mem::take(&mut self.capture);
        let blank = regs.forced_blanking;
        let state = (regs.bg_mode == 7 && !blank)
            .then(|| capture.push_state(index, Plane::of(regs), &self.cgram, vram));
        for x in from..WIDTH {
            let plane = if blank { 0 } else { self.plane_layer(regs, x) };
            capture.records[(line - 1) * WIDTH + x] = if plane == 0 {
                Record::NONE
            } else {
                Record { plane, state: state.unwrap_or(0), plan: self.plan(regs, x) }
            };
        }
        self.capture = capture;
    }

    /// Renders the frame `log` describes as the Enhanced picture (see the module documentation).
    /// The picture of scale 1, with either look, is the frame of [`Renderer::render`].
    ///
    /// # Errors
    ///
    /// The frame uses something the renderer does not model ([`Unsupported`]).
    ///
    /// # Panics
    ///
    /// `options.scale` is not in `1..=MAX_SCALE`.
    pub fn render_hd(
        &mut self,
        log: &FrameLog,
        options: HdOptions,
    ) -> Result<HdFrame, Unsupported> {
        assert!((1..=crate::MAX_SCALE).contains(&options.scale), "the scale is 1 to 8");
        // The application already identifies flight from the game's WRAM state.
        // Do not gate the 3D test on a second, brittle PPU-register signature: flight
        // transitions can change those registers between scanlines and suppress the effect.
        let flight_heading = self.flight_active.then_some(self.flight_heading);
        // One complete somersault every three seconds at the SNES frame rate (60 Hz).
        // This phase affects only the volumetric sprite projection, never game state or Mode 7.
        let flight_spin = flight_heading.map(|_| {
            const SPIN_PERIOD_FRAMES: u64 = 180;
            let phase = (self.flight_spin_frame % SPIN_PERIOD_FRAMES) as f32
                / SPIN_PERIOD_FRAMES as f32
                * std::f32::consts::TAU;
            self.flight_spin_frame = self.flight_spin_frame.wrapping_add(1);
            phase
        });
        if flight_heading.is_none() {
            self.flight_spin_frame = 0;
        }
        let adjusted_log = flight_heading.map(|heading| flight_perspective_log(log, heading));
        let log = adjusted_log.as_ref().unwrap_or(log);
        if self.art.as_ref().is_some_and(|art| art.has_art()) {
            return self.render_hd_art(log, options, flight_heading, flight_spin);
        }
        let mut capture = std::mem::take(&mut self.capture);
        capture.begin(log);
        self.capture = capture;
        self.capturing = true;
        self.begin_text(options);
        let frame = self.render_log(log, None, false).map(|mut frame| {
            if self.sprite_stacking {
                apply_sprite_stacks(log, &mut frame, 0, flight_heading, flight_spin);
            }
            frame
        });
        self.capturing = false;
        self.text_on = false;
        let capture = std::mem::take(&mut self.capture);
        let beneath = self.title_beneath(log, options);
        let picture = frame.map(|frame| {
            let mut hd = {
                let shown = self.without_text(&frame, &beneath);
                picture(shown.as_ref().unwrap_or(&frame), &capture, Some(log), options)
            };
            self.draw_text(&mut hd, &frame, options, &beneath);
            hd
        });
        self.capture = capture;
        picture
    }

    /// Starts noting the text cells of the frame about to be drawn, when the pictures draw text
    /// from a font and the scale is 2 or more.
    fn begin_text(&mut self, options: HdOptions) {
        self.text.clear();
        self.text_on = self.font.is_some() && options.scale >= 2;
    }

    /// The pictures under the title's text, one per BG layer that holds clear title glyphs: the 1x
    /// frame drawn again with only that layer off (see [`crate::font::beneath_logs`]). A fresh
    /// renderer draws them, so the frame's own state is left as it is.
    fn title_beneath(&self, log: &FrameLog, options: HdOptions) -> [Option<Frame>; 4] {
        if self.font.is_none() || options.scale < 2 {
            return [None, None, None, None];
        }
        crate::font::beneath_logs(log, self.text.cells())
            .map(|blank| blank.and_then(|blank| Renderer::new().render(&blank).ok()))
    }

    /// The 1x `frame` as the filter that enlarges it should see it: with the letters of the text
    /// the font redraws painted over by what they stand on (see [`crate::font::without_letters`]),
    /// so that no sub-pixel of an original letter is left beside the HD glyph. The frame itself
    /// when no text is redrawn.
    fn without_text(&self, frame: &Frame, beneath: &[Option<Frame>; 4]) -> Option<Frame> {
        if self.font.is_none() || self.text.cells().is_empty() {
            return None;
        }
        let beneath_pixels: [Option<&[Color]>; 4] =
            std::array::from_fn(|bg| beneath[bg].as_ref().map(|frame| &frame.pixels[..]));
        let pixels = crate::font::without_letters(
            &frame.pixels,
            frame.width,
            self.text.cells(),
            beneath_pixels,
        );
        Some(Frame { pixels, sources: Vec::new(), ..*frame })
    }

    /// Draws the text cells noted while the 1x `frame` was drawn into its picture `hd`, over the
    /// pictures under the title's text (`beneath`, per BG layer).
    fn draw_text(
        &self,
        hd: &mut HdFrame,
        frame: &Frame,
        options: HdOptions,
        beneath: &[Option<Frame>; 4],
    ) {
        if let (Some(font), true) = (&self.font, options.scale >= 2) {
            let beneath_pixels: [Option<&[Color]>; 4] =
                std::array::from_fn(|bg| beneath[bg].as_ref().map(|frame| &frame.pixels[..]));
            font.draw_with(
                &mut hd.rgba,
                hd.width as usize,
                options.scale as usize,
                self.text.cells(),
                &frame.pixels,
                frame.width,
                beneath_pixels,
            );
        }
    }

    /// [`Renderer::render_hd`] with the pack's art (see the `art` module): the frame is drawn
    /// once to see what shows where, and again without the sprites an HD pose replaces.
    fn render_hd_art(
        &mut self,
        log: &FrameLog,
        options: HdOptions,
        flight_heading: Option<f32>,
        flight_spin: Option<f32>,
    ) -> Result<HdFrame, Unsupported> {
        let scale = options.scale as usize;
        let height = usize::from(log.height);
        let mut art = self.art.take().expect("art is on");
        art.begin(WIDTH, height, scale);
        art.begin_bg(log);
        art.active = true;
        let mut binder = art.binder.take().expect("the art's binder");
        self.art = Some(art);
        let mut capture = std::mem::take(&mut self.capture);
        capture.begin(log);
        self.capture = capture;
        self.capturing = true;
        let before = self.obj_pixels;
        self.begin_text(options);
        let first = self.render_log(log, Some(&mut binder), false);
        self.capturing = false;
        self.text_on = false;
        let mut art = self.art.take().expect("art is on");
        art.binder = Some(binder);
        art.active = false;
        let first = match first {
            Ok(frame) => frame,
            Err(reason) => {
                self.art = Some(art);
                return Err(reason);
            }
        };
        let stack_mask = if self.sprite_stacking { sprite_stack_mask(log, flight_heading.is_some()) } else { 0 };
        let mut found = art.find_poses(log);
        // The volumetric renderer takes ownership of the original OAM pixels for these poses.
        found.retain(|(pose, _)| pose.sprites & stack_mask == 0);
        let mut under = None;
        if !found.is_empty() {
            let after = self.obj_pixels;
            self.obj_pixels = before;
            self.hidden_objects = found.iter().fold(0, |all, (pose, _)| all | pose.sprites);
            let rows = art::pose_rows(&found, height);
            self.only_rows = rows;
            under = self.render_log(log, None, false).ok();
            self.only_rows = None;
            self.hidden_objects = 0;
            if let (Some(under), Some((first_row, past))) = (under.as_mut(), rows) {
                // The lines the poses do not reach are the first picture's
                for row in (0..first_row).chain(past..height) {
                    let span = row * WIDTH..(row + 1) * WIDTH;
                    under.pixels[span.clone()].copy_from_slice(&first.pixels[span]);
                }
            }
            self.obj_pixels = after;
        }
        let capture = std::mem::take(&mut self.capture);
        let beneath = self.title_beneath(log, options);
        let drawn = under.as_ref().unwrap_or(&first);
        let mut stacked = Frame {
            width: drawn.width,
            height: drawn.height,
            pixels: drawn.pixels.clone(),
            sources: Vec::new(),
        };
        let remaining_art_mask = found.iter().fold(0_u128, |all, (pose, _)| all | pose.sprites);
        if self.sprite_stacking {
            apply_sprite_stacks(log, &mut stacked, remaining_art_mask, flight_heading, flight_spin);
        }
        let shown = self.without_text(&stacked, &beneath);
        let mut hd = picture(shown.as_ref().unwrap_or(&stacked), &capture, Some(log), options);
        self.capture = capture;
        art.compose(&mut hd, &first, &found, 0, (0, 0, WIDTH as i32, height as i32));
        self.art = Some(art);
        self.draw_text(&mut hd, &first, options, &beneath);
        Ok(hd)
    }
}

/// The picture of the 1x `frame` (drawn from `log`, recorded in `capture`) at the options.
pub(crate) fn picture(
    frame: &Frame,
    capture: &Capture,
    log: Option<&FrameLog>,
    options: HdOptions,
) -> HdFrame {
    let scale = options.scale as usize;
    let lines = frame.height;
    let wide = frame.width;
    // With no plane to sample again, a crisp picture of a power of two (the Scale2x passes) and a pixel picture (the
    // colours of each line, repeated) write the picture's bytes directly
    let direct = match options.look {
        Look::Crisp if scale.is_power_of_two() && scale > 1 => true,
        Look::Pixel => true,
        _ => false,
    };
    if log.is_none() && direct {
        let rgba = if options.look == Look::Pixel {
            pixel_rgba(&frame.pixels, wide, lines, scale)
        } else {
            crisp_rgba(&frame.pixels, wide, lines, scale)
        };
        let (width, height) = ((wide * scale) as u32, (lines * scale) as u32);
        return HdFrame {
            rgba,
            width,
            height,
            scale: options.scale,
            logical: Rect { x: 0, y: 0, width, height },
        };
    }
    let source: Vec<u32> = frame.pixels.iter().map(|c| pack(c.r, c.g, c.b)).collect();
    let filtered = match options.look {
        Look::Crisp => crisp(&source, wide, lines, scale),
        Look::Smooth => smooth(&source, wide, lines, scale),
        Look::Pixel => whole_pixels(source, wide, lines, scale),
    };
    let width = wide * scale;
    let height = lines * scale;
    let mut rgba = crate::pool::BYTES.take(width * height * 4);
    rgba.par_chunks_mut(width * 4).enumerate().for_each(|(y, row)| {
        let line = y / scale;
        for (x, pixel) in row.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            let planed = log.and_then(|log| {
                let record = *capture.records.get(line * wide + x / scale)?;
                plane_colour(capture, log, record, (x / scale, x % scale), (line, y % scale), scale)
            });
            let [r, g, b] = planed.unwrap_or_else(|| unpack(filtered[y * width + x]));
            *pixel = [r, g, b, 0xFF];
        }
    });
    let (width, height) = (width as u32, height as u32);
    HdFrame {
        rgba,
        width,
        height,
        scale: options.scale,
        logical: Rect { x: 0, y: 0, width, height },
    }
}

/// The colour of a Mode 7 plane pixel of the picture: `x` is the pixel of the line and the
/// sub-position of the picture inside it, `y` the line (0-based) and the sub-row. `None` where the
/// pixel is not a plane pixel, the plane is transparent there, or the line is mosaic (the look
/// draws those). At `scale` 1 it is the colour the 1x renderer drew.
fn plane_colour(
    capture: &Capture,
    log: &FrameLog,
    record: Record,
    x: (usize, usize),
    y: (usize, usize),
    scale: usize,
) -> Option<[u8; 3]> {
    if record.plane == 0 {
        return None;
    }
    let state = &capture.states[record.state as usize];
    let regs = &log.entries[state.entry].regs;
    if regs.mosaic_bg[0] {
        return None;
    }
    let vram = match state.vram {
        Some(copy) => &capture.vram[copy][..],
        None => &log.vram[..],
    };
    // The PPU's line numbers start at 1: the first picture row is scanline 1
    let scale = scale as i64;
    let byte =
        state.plane.sample(vram, scale, (x.0 as i32, x.1 as i64), ((y.0 + 1) as u16, y.1 as i64));
    let color = if record.plane == 2 { byte & 0x7F } else { byte };
    if color == 0 {
        return None;
    }
    let colour =
        if regs.direct_color { direct_color(0, color) } else { state.cgram[usize::from(color)] };
    let shown = convert(record.plan.apply(colour), regs.brightness);
    Some([shown.r, shown.g, shown.b])
}

fn pack(r: u8, g: u8, b: u8) -> u32 {
    u32::from(r) << 16 | u32::from(g) << 8 | u32::from(b)
}

fn unpack(packed: u32) -> [u8; 3] {
    [(packed >> 16) as u8, (packed >> 8) as u8, packed as u8]
}

/// The crisp look: Scale2x on the power-of-two part of `scale`, then whole pixels for its odd part.
fn crisp(source: &[u32], width: usize, height: usize, scale: usize) -> Vec<u32> {
    let doublings = scale.trailing_zeros();
    let mut image = source.to_vec();
    let (mut w, mut h) = (width, height);
    for _ in 0..doublings {
        image = scale2x(&image, w, h);
        w *= 2;
        h *= 2;
    }
    whole_pixels(image, w, h, scale >> doublings)
}

/// The `width` x `height` `image` made `factor` times bigger with whole pixels: each pixel a `factor` x `factor` block
/// (the pixel look, and the odd part of the crisp one).
fn whole_pixels(image: Vec<u32>, width: usize, height: usize, factor: usize) -> Vec<u32> {
    if factor == 1 {
        return image;
    }
    let out_w = width * factor;
    let mut out = vec![0_u32; out_w * height * factor];
    out.par_chunks_mut(out_w).enumerate().for_each(|(y, row)| {
        let source_row = &image[(y / factor) * width..][..width];
        for (x, pixel) in row.iter_mut().enumerate() {
            *pixel = source_row[x / factor];
        }
    });
    out
}

/// The four pixels Scale2x makes of `centre` with its neighbours above, right, left and below
/// (upper left, upper right, lower left, lower right): a corner takes the colour of a neighbour
/// when the two neighbours on that corner agree and the pixel differs from both.
#[inline(always)]
#[allow(clippy::needless_bitwise_bool)] // (the `&` keeps the corner rules free of branches: lanes are selected)
fn epx(centre: u32, up: u32, right: u32, left: u32, down: u32) -> [u32; 4] {
    // (`&` and not `&&`: no branch, so the compiler can select lanes)
    let e0 = if (left == up) & (left != down) & (up != right) { up } else { centre };
    let e1 = if (up == right) & (up != left) & (right != down) { right } else { centre };
    let e2 = if (down == left) & (down != right) & (left != up) { left } else { centre };
    let e3 = if (right == down) & (right != up) & (down != left) { down } else { centre };
    [e0, e1, e2, e3]
}

/// The source rows one task of the Scale2x passes takes, at least and at most.
const BAND_MIN: usize = 4;
const BAND_MAX: usize = 32;

/// How many source rows one task takes: about one band for each worker of the pool.
fn band_rows(height: usize) -> usize {
    height.div_ceil(rayon::current_num_threads().max(1)).clamp(BAND_MIN, BAND_MAX)
}

/// Scale2x of the row `cur` of an image, with the rows `up` and `down` around it: the two rows
/// (`upper`, `lower`) of twice the width that it becomes. The ends of a row repeat their pixel.
/// Written so that the inside of the row, which is nearly all of it, is a loop over slices of equal
/// length with no index to check: the compiler turns it into vector code.
#[inline(always)]
fn epx_row(up: &[u32], cur: &[u32], down: &[u32], upper: &mut [u32], lower: &mut [u32]) {
    let width = cur.len();
    assert!(up.len() == width && down.len() == width);
    assert!(upper.len() == 2 * width && lower.len() == 2 * width);
    let edge = |x: usize, upper: &mut [u32], lower: &mut [u32]| {
        let left = cur[x.saturating_sub(1)];
        let right = cur[(x + 1).min(width - 1)];
        let [e0, e1, e2, e3] = epx(cur[x], up[x], right, left, down[x]);
        upper[2 * x] = e0;
        upper[2 * x + 1] = e1;
        lower[2 * x] = e2;
        lower[2 * x + 1] = e3;
    };
    if width < 3 {
        for x in 0..width {
            edge(x, upper, lower);
        }
        return;
    }
    edge(0, upper, lower);
    edge(width - 1, upper, lower);
    let inside = width - 2;
    let (left, centre, right) = (&cur[..inside], &cur[1..=inside], &cur[2..]);
    let (above, below) = (&up[1..=inside], &down[1..=inside]);
    let upper = upper[2..2 * width - 2].as_chunks_mut::<2>().0;
    let lower = lower[2..2 * width - 2].as_chunks_mut::<2>().0;
    for ((((((&l, &c), &r), &u), &d), upper), lower) in
        left.iter().zip(centre).zip(right).zip(above).zip(below).zip(upper).zip(lower)
    {
        let [e0, e1, e2, e3] = epx(c, u, r, l, d);
        *upper = [e0, e1];
        *lower = [e2, e3];
    }
}

/// The rows around source row `y` (above, the row, below), the edges repeating their row.
#[inline(always)]
fn rows3(source: &[u32], width: usize, height: usize, y: usize) -> (&[u32], &[u32], &[u32]) {
    (
        &source[y.saturating_sub(1) * width..][..width],
        &source[y * width..][..width],
        &source[(y + 1).min(height - 1) * width..][..width],
    )
}

/// Scale2x (EPX) of a `width` x `height` image: each pixel becomes a 2 x 2 block whose corners
/// take the colour of a neighbour when the two neighbours on that corner agree and the pixel
/// differs from both.
fn scale2x(source: &[u32], width: usize, height: usize) -> Vec<u32> {
    let mut out = vec![0_u32; 4 * width * height];
    scale2x_into(source, width, height, &mut out);
    out
}

/// [`scale2x`] written into `out`, `4 * width * height` words.
fn scale2x_into(source: &[u32], width: usize, height: usize, out: &mut [u32]) {
    let band = band_rows(height);
    out.par_chunks_mut(4 * width * band).enumerate().for_each(|(n, rows)| {
        for (k, rows) in rows.chunks_exact_mut(4 * width).enumerate() {
            let (upper, lower) = rows.split_at_mut(2 * width);
            let (up, cur, down) = rows3(source, width, height, n * band + k);
            epx_row(up, cur, down, upper, lower);
        }
    });
}

/// Two Scale2x passes at once, a 4x picture of the `width` x `height` image `source`, written into
/// `out` (`16 * width * height` words). Each task takes a band of source rows and makes into a
/// scratch it keeps the rows of the first pass that its second pass reads (one above the band's
/// own and one below, so that the second pass sees exactly the rows of the whole first picture),
/// then writes the second pass's rows straight to `out`: the first picture, a megabyte or more, is
/// never written out.
fn scale4_into(source: &[u32], width: usize, height: usize, out: &mut [u32]) {
    let band = band_rows(height);
    let (mid_width, mid_height) = (2 * width, 2 * height);
    out.par_chunks_mut(16 * width * band).enumerate().for_each_init(Vec::new, |mid, (n, rows)| {
        let (first, past) = (n * band, (n * band + band).min(height));
        // The rows of the first picture the band's second pass reads: from the one above its first
        // row to the one below its last, inside the picture
        let needed_from = (2 * first).saturating_sub(1);
        let needed_to = (2 * past).min(mid_height - 1);
        // They are made by these source rows, each making a pair of rows
        let (source_from, source_to) = (needed_from / 2, needed_to / 2);
        let base = 2 * source_from;
        mid.clear();
        mid.resize(2 * (source_to - source_from + 1) * mid_width, 0);
        for (k, pair) in mid.chunks_exact_mut(2 * mid_width).enumerate() {
            let (upper, lower) = pair.split_at_mut(mid_width);
            let (up, cur, down) = rows3(source, width, height, source_from + k);
            epx_row(up, cur, down, upper, lower);
        }
        let mid_row = |row: usize| &mid[(row - base) * mid_width..][..mid_width];
        for (k, pair) in rows.chunks_exact_mut(4 * mid_width).enumerate() {
            let row = 2 * first + k;
            let (upper, lower) = pair.split_at_mut(2 * mid_width);
            epx_row(
                mid_row(row.saturating_sub(1)),
                mid_row(row),
                mid_row((row + 1).min(mid_height - 1)),
                upper,
                lower,
            );
        }
    });
}

/// The word of the RGBA bytes `[r, g, b, 255]` of a colour, as they are in memory.
#[inline(always)]
fn rgba_word(colour: Color) -> u32 {
    u32::from_ne_bytes([colour.r, colour.g, colour.b, 0xFF])
}

/// The crisp look of a power-of-two `scale` above 1, as RGBA bytes (alpha 255): Scale2x on the
/// colours as words of their bytes (the filter only compares colours, so any one-to-one packing
/// gives the same picture), the last passes written straight into the picture's bytes. The bytes
/// are a buffer of [`crate::pool::BYTES`].
fn crisp_rgba(pixels: &[Color], width: usize, height: usize, scale: usize) -> Vec<u8> {
    let doublings = scale.trailing_zeros();
    let mut image = crate::pool::WORDS.take(width * height);
    for (word, &colour) in image.iter_mut().zip(pixels) {
        *word = rgba_word(colour);
    }
    let (mut w, mut h) = (width, height);
    // All the passes but the last two (or one) make a picture of their own
    for _ in 0..doublings.saturating_sub(2) {
        let mut next = crate::pool::WORDS.take(4 * w * h);
        scale2x_into(&image, w, h, &mut next);
        crate::pool::WORDS.give(std::mem::replace(&mut image, next));
        w *= 2;
        h *= 2;
    }
    let mut rgba = crate::pool::BYTES.take(4 * width * scale * height * scale);
    let last = |image: &[u32], words: &mut [u32]| {
        if doublings == 1 {
            scale2x_into(image, w, h, words);
        } else {
            scale4_into(image, w, h, words);
        }
    };
    match bytemuck::try_cast_slice_mut::<u8, u32>(&mut rgba) {
        Ok(words) => last(&image, words),
        // (a buffer whose address is not a multiple of four: the allocator never gives one)
        Err(_) => {
            let mut words = vec![0_u32; rgba.len() / 4];
            last(&image, &mut words);
            for (bytes, word) in rgba.as_chunks_mut::<4>().0.iter_mut().zip(words) {
                *bytes = word.to_ne_bytes();
            }
        }
    }
    crate::pool::WORDS.give(image);
    rgba
}

/// The pixel look of `scale` as RGBA bytes (alpha 255): each pixel of the `width` x `height` frame a block of
/// `scale` x `scale`. The bytes are a buffer of [`crate::pool::BYTES`].
fn pixel_rgba(pixels: &[Color], width: usize, height: usize, scale: usize) -> Vec<u8> {
    let row_bytes = 4 * width * scale;
    let mut rgba = crate::pool::BYTES.take(row_bytes * height * scale);
    rgba.par_chunks_mut(row_bytes * scale).enumerate().for_each(|(line, block)| {
        let (first, rest) = block.split_at_mut(row_bytes);
        let colours = &pixels[line * width..][..width];
        for (cells, colour) in first.chunks_exact_mut(4 * scale).zip(colours) {
            for cell in cells.as_chunks_mut::<4>().0 {
                *cell = [colour.r, colour.g, colour.b, 0xFF];
            }
        }
        for row in rest.chunks_exact_mut(row_bytes) {
            row.copy_from_slice(first);
        }
    });
    rgba
}

/// The Catmull-Rom weights of the four taps around a picture pixel, in whole units, and the index
/// of the first tap. The picture pixel `index` of `scale` centres its source position at
/// `(2 * index + 1 - scale) / (2 * scale)` source pixels; the weights sum to `2 * d^3` with `d` the
/// denominator `2 * scale`.
fn taps(index: usize, scale: i64) -> (isize, [i64; 4]) {
    let d = 2 * scale;
    let t = 2 * index as i64 + 1 - scale;
    let first = t.div_euclid(d);
    let m = t.rem_euclid(d);
    let (m2, m3) = (m * m, m * m * m);
    let weights = [
        -m3 + 2 * m2 * d - m * d * d,
        3 * m3 - 5 * m2 * d + 2 * d * d * d,
        -3 * m3 + 4 * m2 * d + m * d * d,
        m3 - m2 * d,
    ];
    ((first - 1) as isize, weights)
}

/// The smooth look: a separable Catmull-Rom filter, the rows first and then the columns, in
/// integer arithmetic with the weights of [`taps`] and a rounded division at the end.
fn smooth(source: &[u32], width: usize, height: usize, scale: usize) -> Vec<u32> {
    let scale = scale as i64;
    let out_w = width * scale as usize;
    let out_h = height * scale as usize;
    let whole = 2 * (2 * scale).pow(3);
    let denominator = whole * whole;
    let columns: Vec<(isize, [i64; 4])> = (0..out_w).map(|x| taps(x, scale)).collect();
    let rows: Vec<(isize, [i64; 4])> = (0..out_h).map(|y| taps(y, scale)).collect();
    let clamp = |index: isize, size: usize| index.clamp(0, size as isize - 1) as usize;

    // The rows, filtered across: units of `whole`
    let mut across = vec![[0_i64; 3]; height * out_w];
    across.par_chunks_mut(out_w).enumerate().for_each(|(y, row)| {
        let source_row = &source[y * width..][..width];
        for (x, sample) in row.iter_mut().enumerate() {
            let (first, weights) = columns[x];
            let mut sum = [0_i64; 3];
            for (k, &weight) in weights.iter().enumerate() {
                let packed = source_row[clamp(first + k as isize, width)];
                let [r, g, b] = unpack(packed);
                for (total, channel) in sum.iter_mut().zip([r, g, b]) {
                    *total += weight * i64::from(channel);
                }
            }
            *sample = sum;
        }
    });

    let mut out = vec![0_u32; out_w * out_h];
    out.par_chunks_mut(out_w).enumerate().for_each(|(y, row)| {
        let (first, weights) = rows[y];
        for (x, pixel) in row.iter_mut().enumerate() {
            let mut sum = [0_i64; 3];
            for (k, &weight) in weights.iter().enumerate() {
                let sample = across[clamp(first + k as isize, height) * out_w + x];
                for (total, value) in sum.iter_mut().zip(sample) {
                    *total += weight * value;
                }
            }
            let [r, g, b] = sum.map(|value| {
                ((value + denominator / 2).div_euclid(denominator)).clamp(0, 255) as u8
            });
            *pixel = pack(r, g, b);
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MAX_SCALE;
    use ssd_oracle::{LineEntry, SpriteLine};

    /// A frame log of 224 lines with the same registers on each and the given VRAM, empty OAM.
    fn mode_7_log(regs: LineRegs, vram: Vec<u16>, cgram: &[u16; 256]) -> FrameLog {
        let entries = (1..=224)
            .map(|scanline| LineEntry {
                scanline,
                from_pixel: 0,
                regs,
                sprites: SpriteLine::default(),
            })
            .collect();
        FrameLog {
            complete: true,
            height: 224,
            h_hi_res: false,
            v_hi_res: false,
            entries,
            cgram_start: cgram.to_vec(),
            cgram_writes: Vec::new(),
            vram,
            vram_writes: Vec::new(),
            oam_low: vec![0; 512],
            oam_high: vec![0; 32],
            oam_writes: Vec::new(),
        }
    }

    /// Mode 7, BG1 only, the identity matrix at `matrix` (8.8 fixed point), brightness full.
    fn plane_regs(matrix: u16) -> LineRegs {
        LineRegs {
            brightness: 15,
            bg_mode: 7,
            main_bg: [true, false, false, false],
            m7_a: matrix,
            m7_d: matrix,
            ..LineRegs::default()
        }
    }

    /// A plane whose row 0 of tile 0 has colour `(column % 8) + 1` and whose tile map is zero.
    fn stripes() -> Vec<u16> {
        let mut vram = vec![0_u16; 32768];
        for (column, word) in vram.iter_mut().take(8).enumerate() {
            *word = ((column as u16) + 1) << 8;
        }
        vram
    }

    fn cgram_identity() -> [u16; 256] {
        let mut cgram = [0_u16; 256];
        for (index, entry) in cgram.iter_mut().enumerate() {
            *entry = index as u16;
        }
        cgram
    }

    #[test]
    fn flight_sprite_camera_preserves_near_native_sprite_scale() {
        let focal = 128.0;
        let distance = sprite_camera_distance(focal, true);
        let projected_scale = focal / distance;
        assert!(projected_scale > 0.90 && projected_scale < 1.0);
        assert_eq!(sprite_camera_distance(focal, false), focal * 2.8);
    }

    #[test]
    fn flight_plane_has_scanline_perspective_instead_of_uniform_affine_scale() {
        let log = mode_7_log(plane_regs(256), stripes(), &cgram_identity());
        let transformed = flight_perspective_log(&log, -std::f32::consts::FRAC_PI_2);
        let near_horizon = &transformed.entries[0].regs;
        let near_camera = &transformed.entries[223].regs;
        assert!(i16::from_ne_bytes(near_horizon.m7_a.to_ne_bytes()).unsigned_abs()
            < i16::from_ne_bytes(near_camera.m7_a.to_ne_bytes()).unsigned_abs());
        assert!(i16::from_ne_bytes(near_horizon.m7_d.to_ne_bytes()).unsigned_abs()
            < i16::from_ne_bytes(near_camera.m7_d.to_ne_bytes()).unsigned_abs());
    }

    #[test]
    fn flight_heading_rotates_the_mode7_sampling_basis() {
        let log = mode_7_log(plane_regs(256), stripes(), &cgram_identity());
        let straight = flight_perspective_log(&log, -std::f32::consts::FRAC_PI_2);
        let turned = flight_perspective_log(&log, 0.0);
        let row = 180;
        assert_ne!(straight.entries[row].regs.m7_a, turned.entries[row].regs.m7_a);
        assert_ne!(straight.entries[row].regs.m7_b, turned.entries[row].regs.m7_b);
    }

    #[test]
    fn enhanced_flight_state_changes_the_rendered_mode7_plane() {
        let log = mode_7_log(plane_regs(256), stripes(), &cgram_identity());
        let options = HdOptions { scale: 1, look: Look::Pixel };
        let ordinary = Renderer::new().render_hd(&log, options).unwrap();
        let mut flight = Renderer::new();
        flight.set_flight_state(true, -std::f32::consts::FRAC_PI_2);
        let perspective = flight.render_hd(&log, options).unwrap();
        assert_ne!(ordinary.rgba, perspective.rgba);
    }

    fn rgb(pixel: &[u8]) -> [u8; 3] {
        [pixel[0], pixel[1], pixel[2]]
    }

    #[test]
    fn scale_one_is_the_ppu_frame_for_every_look_and_for_a_plane() {
        let log = mode_7_log(plane_regs(0x0080), stripes(), &cgram_identity());
        let mut renderer = Renderer::new();
        let frame = renderer.render(&log).unwrap();
        for look in [Look::Crisp, Look::Smooth, Look::Pixel] {
            let hd = renderer.render_hd(&log, HdOptions { scale: 1, look }).unwrap();
            assert_eq!((hd.width, hd.height, hd.scale), (256, 224, 1));
            for (pixel, expected) in hd.rgba.as_chunks::<4>().0.iter().zip(&frame.pixels) {
                assert_eq!(rgb(pixel), [expected.r, expected.g, expected.b], "{look:?}");
            }
        }
    }

    #[test]
    fn a_plane_is_sampled_at_the_picture_resolution() {
        // The matrix halves the plane: each screen pixel covers half a plane column, so the
        // plane's column is the picture's x divided by 8 at scale 4
        let log = mode_7_log(plane_regs(0x0080), stripes(), &cgram_identity());
        let hd =
            Renderer::new().render_hd(&log, HdOptions { scale: 4, look: Look::Crisp }).unwrap();
        assert_eq!((hd.width, hd.height), (1024, 896));
        for x in 0..1024_usize {
            let column = (x / 8) % 8;
            let expected = convert(column as u16 + 1, 15);
            let pixel = &hd.rgba[x * 4..x * 4 + 3];
            assert_eq!(rgb(pixel), [expected.r, expected.g, expected.b], "x {x}");
        }
    }

    #[test]
    fn crisp_is_scale2x_on_a_checker_diagonal() {
        // The top-left block takes the top-left pixel's colour, its corner the colour of the
        // pixel to its right, its lower-left corner stays; the top-right block takes the left
        let source = [1, 0, 0, 1];
        let out = scale2x(&source, 2, 2);
        let top_left = [out[0], out[1], out[4], out[5]];
        let top_right = [out[2], out[3], out[6], out[7]];
        assert_eq!(top_left, [1, 1, 1, 0]);
        assert_eq!(top_right, [0, 0, 1, 0]);
    }

    #[test]
    fn odd_scales_are_whole_pixels_and_the_power_of_two_part_is_smoothed() {
        let source = [1, 2, 3, 4];
        for scale in [3_usize, 5, 7] {
            let out = crisp(&source, 2, 2, scale);
            let width = 2 * scale;
            for y in (0..2 * scale).step_by(scale) {
                for x in (0..width).step_by(scale) {
                    let block = out[y * width + x];
                    for dy in 0..scale {
                        for dx in 0..scale {
                            assert_eq!(out[(y + dy) * width + x + dx], block, "scale {scale}");
                        }
                    }
                }
            }
        }
        // Scale 6: Scale2x, then whole pixels of 3
        let six = crisp(&source, 2, 2, 6);
        assert_eq!(six.len(), 12 * 12);
        assert_eq!(six[0], scale2x(&source, 2, 2)[0]);
        assert!(six[..3].iter().all(|&p| p == six[0]));
    }

    /// A small deterministic generator for the equality tests.
    struct Random(u64);

    impl Random {
        fn next(&mut self) -> u32 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            (self.0 >> 16) as u32
        }
    }

    /// The corner rules of the first version of the code, with the branches.
    fn epx_original(centre: u32, up: u32, right: u32, left: u32, down: u32) -> [u32; 4] {
        let (mut e0, mut e1, mut e2, mut e3) = (centre, centre, centre, centre);
        if left == up && left != down && up != right {
            e0 = up;
        }
        if up == right && up != left && right != down {
            e1 = right;
        }
        if down == left && down != right && left != up {
            e2 = left;
        }
        if right == down && right != up && down != left {
            e3 = down;
        }
        [e0, e1, e2, e3]
    }

    /// The Scale2x of the first version of the code: one serial pass, every neighbour looked up
    /// with its index clamped.
    fn scale2x_reference(source: &[u32], width: usize, height: usize) -> Vec<u32> {
        let mut out = vec![0_u32; 4 * width * height];
        for y in 0..height {
            for x in 0..width {
                let at = |x: usize, y: usize| source[y * width + x];
                let [e0, e1, e2, e3] = epx_original(
                    at(x, y),
                    at(x, y.saturating_sub(1)),
                    at((x + 1).min(width - 1), y),
                    at(x.saturating_sub(1), y),
                    at(x, (y + 1).min(height - 1)),
                );
                out[2 * y * 2 * width + 2 * x] = e0;
                out[2 * y * 2 * width + 2 * x + 1] = e1;
                out[(2 * y + 1) * 2 * width + 2 * x] = e2;
                out[(2 * y + 1) * 2 * width + 2 * x + 1] = e3;
            }
        }
        out
    }

    /// The first version of the crisp picture of a power-of-two scale: the colours packed as
    /// `0xRRGGBB`, Scale2x for every doubling but the last, which unpacks into RGBA bytes.
    fn crisp_rgba_reference(
        pixels: &[Color],
        width: usize,
        height: usize,
        scale: usize,
    ) -> Vec<u8> {
        let mut image: Vec<u32> = pixels.iter().map(|c| pack(c.r, c.g, c.b)).collect();
        let (mut across, mut down) = (width, height);
        for _ in 1..scale.trailing_zeros() {
            image = scale2x_reference(&image, across, down);
            across *= 2;
            down *= 2;
        }
        let last = scale2x_reference(&image, across, down);
        last.iter()
            .flat_map(|&packed| {
                let [r, g, b] = unpack(packed);
                [r, g, b, 0xFF]
            })
            .collect()
    }

    /// An image of `colours` distinct colours, in runs and noise, so every Scale2x condition comes up.
    fn random_pixels(random: &mut Random, count: usize, colours: u32) -> Vec<Color> {
        let palette: Vec<Color> = (0..colours)
            .map(|_| Color {
                r: random.next() as u8,
                g: random.next() as u8,
                b: random.next() as u8,
                a: random.next() as u8,
            })
            .collect();
        (0..count).map(|_| palette[(random.next() % colours) as usize]).collect()
    }

    #[test]
    fn scale2x_equals_the_serial_version_on_random_images() {
        let mut random = Random(0x9E37_79B9_7F4A_7C15);
        for (width, height) in
            [(1, 1), (1, 5), (2, 2), (2, 7), (3, 3), (4, 1), (5, 9), (17, 6), (64, 33), (372, 40)]
        {
            for colours in [2, 3, 5, 200] {
                let source: Vec<u32> =
                    (0..width * height).map(|_| random.next() % colours).collect();
                assert_eq!(
                    scale2x(&source, width, height),
                    scale2x_reference(&source, width, height),
                    "{width} x {height}, {colours} colours"
                );
            }
        }
    }

    #[test]
    fn the_crisp_picture_equals_the_first_version_at_every_power_of_two() {
        let mut random = Random(0xD1B5_4A32_D192_ED03);
        for (width, height) in
            [(1, 1), (2, 3), (3, 2), (5, 4), (16, 9), (37, 21), (96, 70), (372, 31)]
        {
            for colours in [2, 3, 6, 400] {
                let pixels = random_pixels(&mut random, width * height, colours);
                for scale in [2, 4, 8] {
                    if width * height * scale * scale > 400_000 {
                        continue;
                    }
                    let fast = crisp_rgba(&pixels, width, height, scale);
                    let reference = crisp_rgba_reference(&pixels, width, height, scale);
                    assert!(
                        fast == reference,
                        "{width} x {height}, {colours} colours, scale {scale}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_crisp_picture_is_the_same_in_a_pool_of_any_size() {
        // The buffers come back dirty: a picture made after others is the same as the first
        let mut random = Random(77);
        let pixels = random_pixels(&mut random, 372 * 24, 4);
        let first = crisp_rgba(&pixels, 372, 24, 4);
        let mut dirty = crisp_rgba(&random_pixels(&mut random, 372 * 24, 90), 372, 24, 4);
        dirty.fill(0xA5);
        crate::pool::BYTES.give(dirty);
        assert!(crisp_rgba(&pixels, 372, 24, 4) == first, "a dirty buffer changed the picture");
    }

    #[test]
    fn the_pixel_look_is_each_pixel_as_a_block_with_or_without_a_plane() {
        let mut random = Random(5);
        let (width, height) = (37, 9);
        let pixels = random_pixels(&mut random, width * height, 40);
        let frame = Frame { pixels: pixels.clone(), sources: Vec::new(), width, height };
        for scale in 1..=MAX_SCALE as usize {
            let options = HdOptions { scale: scale as u32, look: Look::Pixel };
            // The bytes written directly (no plane) and the picture of a log with no plane pixel are the same blocks
            let direct = picture(&frame, &Capture::default(), None, options);
            let capture =
                Capture { records: vec![Record::NONE; width * height], ..Capture::default() };
            let log = mode_7_log(LineRegs::default(), vec![0; 32768], &cgram_identity());
            let general = picture(&frame, &capture, Some(&log), options);
            assert!(direct.rgba == general.rgba, "scale {scale}");
            let row = 4 * width * scale;
            for (y, line) in direct.rgba.chunks_exact(row).enumerate() {
                for (x, cell) in line.as_chunks::<4>().0.iter().enumerate() {
                    let colour = pixels[(y / scale) * width + x / scale];
                    assert_eq!(
                        *cell,
                        [colour.r, colour.g, colour.b, 0xFF],
                        "scale {scale} at {x}, {y}"
                    );
                }
            }
        }
    }

    #[test]
    fn smooth_keeps_flat_colours_and_the_source_at_scale_one() {
        let flat = vec![pack(10, 200, 30); 6];
        assert!(smooth(&flat, 3, 2, 4).iter().all(|&p| p == pack(10, 200, 30)));
        let source = [pack(1, 2, 3), pack(4, 5, 6), pack(7, 8, 9), pack(10, 11, 12)];
        assert_eq!(smooth(&source, 2, 2, 1), source);
    }

    #[test]
    fn every_scale_renders_and_the_logical_rect_is_the_whole_picture() {
        let log = mode_7_log(plane_regs(0x0100), stripes(), &cgram_identity());
        let mut renderer = Renderer::new();
        for scale in 1..=MAX_SCALE {
            for look in [Look::Crisp, Look::Smooth, Look::Pixel] {
                let hd = renderer.render_hd(&log, HdOptions { scale, look }).unwrap();
                let (w, h) = (256 * scale, 224 * scale);
                assert_eq!((hd.width, hd.height), (w, h));
                assert_eq!(hd.rgba.len(), (w * h * 4) as usize);
                assert_eq!(hd.logical, Rect { x: 0, y: 0, width: w, height: h });
                assert!(hd.rgba.as_chunks::<4>().0.iter().all(|p| p[3] == 0xFF));
            }
        }
    }

    #[test]
    fn a_picture_of_a_frame_that_is_not_modelled_is_refused() {
        let mut log = mode_7_log(plane_regs(0x0100), stripes(), &cgram_identity());
        log.complete = false;
        let refused = Renderer::new().render_hd(&log, HdOptions { scale: 2, look: Look::Crisp });
        assert_eq!(refused, Err(Unsupported::Incomplete));
    }
}

/// Returns the OAM entries belonging to fighter and field entities eligible for sprite stacking.
/// Effects, ground shadows, portraits and UI objects are deliberately excluded.
fn stackable_pose(pose: &crate::art::poses::Pose, flight: bool) -> bool {
    !pose.shadow
        && !pose.cut
        && pose.sprites != 0
        && !pose.arr.is_empty()
        && (flight || pose.fighter || pose.field)
}

/// OAM entries that the volume pass will replace. In flight, the game may classify the craft as
/// an effect rather than a fighter/field group, so include every complete non-shadow entity.
fn sprite_stack_mask(log: &FrameLog, flight: bool) -> u128 {
    use crate::art::poses::{self, View};
    let view = View::window();
    let sprites = poses::sprites_in(log, &view);
    let poses = poses::poses_in(log, &sprites, &view);
    let mut mask = 0_u128;
    for pose in &poses {
        if !stackable_pose(pose, flight) {
            continue;
        }
        let is_union = poses.iter().any(|other| {
            other.sprites != pose.sprites
                && other.sprites & pose.sprites == other.sprites
                && stackable_pose(other, flight)
        });
        if !is_union { mask |= pose.sprites; }
    }
    mask
}


/// Rebuilds the Mode 7 line matrices as a perspective ground plane for the live flight mode.
/// The SNES affine plane is sampled per scanline; increasing its map-space step toward the
/// horizon produces the depth compression that a flat affine Mode 7 image cannot provide.
fn flight_perspective_log(log: &FrameLog, heading: f32) -> FrameLog {
    let mut adjusted = log.clone();
    let height = usize::from(log.height).max(1) as f32;
    // Mode 7 flight frames are 224/239 lines tall. Keep the horizon above the caption
    // band, and make the texture sampling step grow toward the camera (not shrink).
    let horizon = (height * 0.30).clamp(20.0, (height - 24.0).max(20.0));
    let bottom = (height - 1.0).max(horizon + 1.0);
    let (sin_yaw, cos_yaw) = (heading + std::f32::consts::FRAC_PI_2).sin_cos();

    for entry in &mut adjusted.entries {
        let regs = &mut entry.regs;
        if regs.forced_blanking || regs.bg_mode != 7 || regs.main_bg != [true, false, false, false] {
            continue;
        }
        let row = f32::from(entry.scanline);
        // Perspective projection: close to the horizon one screen row covers very little
        // world space; toward the bottom it covers progressively more. The previous formula
        // inverted this relationship and compressed the foreground while magnifying the horizon.
        let depth = ((row - horizon) / (bottom - horizon)).clamp(0.0, 1.0);
        let perspective = 0.18 + 2.35 * depth.powf(1.35);
        let a = f32::from(regs.m7_a as i16);
        let b = f32::from(regs.m7_b as i16);
        let c = f32::from(regs.m7_c as i16);
        let d = f32::from(regs.m7_d as i16);

        // Rotate the sampling basis around Mode 7's existing center, then apply scanline depth.
        let values = [
            (a * cos_yaw - b * sin_yaw) * perspective,
            (a * sin_yaw + b * cos_yaw) * perspective,
            (c * cos_yaw - d * sin_yaw) * perspective,
            (c * sin_yaw + d * cos_yaw) * perspective,
        ];
        let signed = |value: f32| -> u16 {
            value.round().clamp(i16::MIN as f32, i16::MAX as f32) as i16 as u16
        };
        [regs.m7_a, regs.m7_b, regs.m7_c, regs.m7_d] = values.map(signed);
    }
    adjusted
}

/// Keeps a flight sprite close to its source size while allowing the depth axis to
/// produce visible parallax. The old 2.8x distance shrank every flight craft to roughly 36%.
fn sprite_camera_distance(focal: f32, flight: bool) -> f32 {
    focal * if flight { 1.05 } else { 2.8 }
}

/// Projects one independent pose per connected OAM entity and replaces only those pixels.
/// The clean PPU background is redrawn with selected OAM entries hidden, so perspective shifts
/// do not leave the original flat sprite behind.
fn apply_sprite_stacks(
    log: &FrameLog,
    frame: &mut Frame,
    additional_hidden: u128,
    flight_heading: Option<f32>,
    flight_spin: Option<f32>,
) -> u128 {
    use crate::art::poses::{self, View};
    use crate::sprite_volume::{project_perspective, project_perspective_with_roll, PerspectiveCamera, RgbaFrame, SpriteVolume};

    if frame.width != WIDTH || frame.height == 0 { return 0; }
    let view = View::window();
    let sprites = poses::sprites_in(log, &view);
    let poses = poses::poses_in(log, &sprites, &view);
    let mut draws = Vec::new();
    let mut mask = 0_u128;
    let mut eligible = 0usize;
    let mut unions = 0usize;
    let mut volume_failures = 0usize;
    let mut effect_poses = 0usize;
    const PAD: usize = 18;
    const MAX_HALF_DEPTH: usize = 7;

    for pose in &poses {
        if !stackable_pose(pose, flight_heading.is_some()) {
            continue;
        }
        let is_union = poses.iter().any(|other| {
            other.sprites != pose.sprites
                && other.sprites & pose.sprites == other.sprites
                && stackable_pose(other, flight_heading.is_some())
        });
        if is_union { unions += 1; continue; }
        eligible += 1;

        let width = pose.w + PAD * 2;
        let height = pose.h + PAD * 2;
        let mut rgba = vec![[0, 0, 0, 0]; width * height];
        if pose.rgba.len() == pose.arr.len() * 4 {
            // Effect poses carry their actual scanline-resolved RGBA colours. Re-reading CGRAM
            // here would flatten palette effects and can give the craft the wrong colours.
            for (at, pixel) in pose.rgba.chunks_exact(4).enumerate() {
                if pixel[3] == 0 { continue; }
                let x = at % pose.w + PAD;
                let y = at / pose.w + PAD;
                rgba[y * width + x] = [pixel[0], pixel[1], pixel[2], pixel[3]];
            }
            effect_poses += 1;
        } else {
            for (at, &index) in pose.arr.iter().enumerate() {
                if index == 0 { continue; }
                let palette_index = 128 + usize::from(pose.palette) * 16 + usize::from(index);
                let Some(&colour) = log.cgram_start.get(palette_index) else { continue };
                let [r, g, b] = poses::rgb555(colour);
                let x = at % pose.w + PAD;
                let y = at / pose.w + PAD;
                rgba[y * width + x] = [r, g, b, 255];
            }
        }
        let Ok(source) = RgbaFrame::new(width, height, rgba) else { continue };

        let volume_result = if flight_heading.is_some() {
            // A flight craft is read as a true thick stack, not a 13-slice paper-thin extrusion.
            // Scale the depth budget with the source silhouette so the side faces survive rotation.
            let half_depth = (width.max(height) / 5).clamp(10, 24);
            SpriteVolume::from_flight_silhouette_with_depth(&source, half_depth)
        } else {
            SpriteVolume::from_silhouette(&source, MAX_HALF_DEPTH)
        };
        let Ok(volume) = volume_result else { volume_failures += 1; continue };
        let focal = width.max(height) as f32 * 2.2;
        let camera = PerspectiveCamera {
            yaw: flight_heading.map_or(0.78, |heading| heading + std::f32::consts::FRAC_PI_2 + 0.78),
            // Rotate the volume around its horizontal axis: this is the visible 360° aerial flip.
            // The base viewing pitch remains, while the animation completes one turn in 180 frames.
            pitch: if flight_heading.is_some() {
                0.48
            } else {
                0.18
            },
            distance: if flight_heading.is_some() { focal * 0.92 } else { sprite_camera_distance(focal, false) },
            focal_length: focal, width, height, near_plane: 0.1,
        };
        let projected_result = if let Some(spin) = flight_spin {
            project_perspective_with_roll(&volume, camera, spin)
        } else {
            project_perspective(&volume, camera)
        };
        let Ok(projected) = projected_result else { volume_failures += 1; continue };
        mask |= pose.sprites;
        draws.push((pose.first, pose.org.0 - PAD as i32, pose.org.1 - PAD as i32, projected));
    }

    // Report the first real frame's conversion counts once. This distinguishes a renderer
    // path/pose-classification failure from a projection that runs but looks too subtle.
    static REPORTED: std::sync::atomic::AtomicBool =
        std::sync::atomic::AtomicBool::new(false);
    if !REPORTED.swap(true, std::sync::atomic::Ordering::Relaxed) {
        log::warn!(
            "sprite stacking: frame={}x{}, flight={}, OAM sprites={}, poses={}, eligible={}, effects={}, unions={}, converted={}, volume/projection failures={}, hidden-mask={:#034x}",
            frame.width, frame.height, flight_heading.is_some(), sprites.len(), poses.len(), eligible,
            effect_poses, unions, draws.len(), volume_failures, mask
        );
    }
    if mask == 0 { return 0; }
    let mut background_renderer = Renderer::new();
    background_renderer.hidden_objects = mask | additional_hidden;
    let Ok(background) = background_renderer.render(log) else { return 0 };
    frame.pixels.copy_from_slice(&background.pixels);

    draws.sort_by_key(|(first, _, _, _)| std::cmp::Reverse(*first));
    for (_, left, top, projected) in draws {
        for y in 0..projected.height {
            let screen_y = top + y as i32;
            if screen_y < 0 || screen_y >= frame.height as i32 { continue; }
            for x in 0..projected.width {
                let screen_x = left + x as i32;
                if screen_x < 0 || screen_x >= frame.width as i32 { continue; }
                let pixel = projected.pixels[y * projected.width + x];
                if pixel[3] == 0 { continue; }
                frame.pixels[screen_y as usize * frame.width + screen_x as usize] =
                    Color::rgb(pixel[0], pixel[1], pixel[2]);
            }
        }
    }
    mask
}
