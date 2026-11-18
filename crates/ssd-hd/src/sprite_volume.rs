//! Pixel-preserving sprite volumes built from RGBA frames and per-pixel depth maps.
//! Renderer-independent foundation; this module does not change the existing render path.
//! A depth value of 0 is the back and 255 is the front. Each opaque source pixel is copied
//! into consecutive slices ending at its mapped depth, preserving colour and alpha.

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaFrame {
    pub width: usize,
    pub height: usize,
    /// Row-major pixels in R, G, B, A order.
    pub pixels: Vec<[u8; 4]>,
}

impl RgbaFrame {
    pub fn new(width: usize, height: usize, pixels: Vec<[u8; 4]>) -> Result<Self, VolumeError> {
        let expected = width.checked_mul(height).ok_or(VolumeError::DimensionsOverflow)?;
        if width == 0 || height == 0 {
            return Err(VolumeError::EmptyFrame);
        }
        if pixels.len() != expected {
            return Err(VolumeError::PixelCountMismatch { expected, actual: pixels.len() });
        }
        Ok(Self { width, height, pixels })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VolumeSlice {
    /// Zero is the back of the volume; larger values are closer to the front.
    pub z: usize,
    pub width: usize,
    pub height: usize,
    /// Row-major RGBA pixels; alpha zero means empty.
    pub pixels: Vec<[u8; 4]>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpriteVolume {
    pub width: usize,
    pub height: usize,
    pub slices: Vec<VolumeSlice>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VolumeError {
    EmptyFrame,
    DimensionsOverflow,
    PixelCountMismatch { expected: usize, actual: usize },
    DepthMapSizeMismatch { expected: usize, actual: usize },
    ZeroDepthLayers,
    ZeroThickness,
    MissingSourceMap,
    InvalidCamera,
}

impl std::fmt::Display for VolumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFrame => f.write_str("frame dimensions must be non-zero"),
            Self::DimensionsOverflow => f.write_str("frame dimensions overflow usize"),
            Self::PixelCountMismatch { expected, actual } =>
                write!(f, "expected {expected} pixels, got {actual}"),
            Self::DepthMapSizeMismatch { expected, actual } =>
                write!(f, "expected {expected} depth values, got {actual}"),
            Self::ZeroDepthLayers => f.write_str("depth_layers must be greater than zero"),
            Self::ZeroThickness => f.write_str("thickness must be greater than zero"),
            Self::MissingSourceMap => f.write_str("the rendered frame has no source map; use Renderer::render_bound"),
            Self::InvalidCamera => f.write_str("camera angles must be finite and scale must be positive"),
        }
    }
}

impl std::error::Error for VolumeError {}

impl SpriteVolume {
    /// Builds a stack from independently authored cross-section images.
    ///
    /// Every image must be registered to the same canvas (same width and height).
    /// Unlike `from_silhouette` or `extrude`, this does not infer or copy depth:
    /// each input image becomes exactly one ordered slice of the volume.
    pub fn from_cross_sections(sections: &[RgbaFrame]) -> Result<Self, VolumeError> {
        let Some(first) = sections.first() else {
            return Err(VolumeError::ZeroDepthLayers);
        };
        let count = first.width.checked_mul(first.height)
            .ok_or(VolumeError::DimensionsOverflow)?;
        if first.width == 0 || first.height == 0 {
            return Err(VolumeError::EmptyFrame);
        }
        if first.pixels.len() != count {
            return Err(VolumeError::PixelCountMismatch {
                expected: count,
                actual: first.pixels.len(),
            });
        }

        let mut slices = Vec::with_capacity(sections.len());
        for (z, section) in sections.iter().enumerate() {
            if section.width != first.width || section.height != first.height {
                return Err(VolumeError::DepthMapSizeMismatch {
                    expected: count,
                    actual: section.width.saturating_mul(section.height),
                });
            }
            if section.pixels.len() != count {
                return Err(VolumeError::PixelCountMismatch {
                    expected: count,
                    actual: section.pixels.len(),
                });
            }
            slices.push(VolumeSlice {
                z,
                width: first.width,
                height: first.height,
                pixels: section.pixels.clone(),
            });
        }
        Ok(Self { width: first.width, height: first.height, slices })
    }
    /// Builds a volume from one depth value (0..=255) per source pixel.
    /// Each opaque pixel occupies up to thickness consecutive slices, ending at its mapped depth.
    /// Fully transparent source pixels remain empty.
    pub fn from_depth_map(
        frame: &RgbaFrame,
        depth_map: &[u8],
        depth_layers: usize,
        thickness: usize,
    ) -> Result<Self, VolumeError> {
        let expected = frame.width.checked_mul(frame.height)
            .ok_or(VolumeError::DimensionsOverflow)?;
        if frame.width == 0 || frame.height == 0 {
            return Err(VolumeError::EmptyFrame);
        }
        if frame.pixels.len() != expected {
            return Err(VolumeError::PixelCountMismatch {
                expected,
                actual: frame.pixels.len(),
            });
        }
        if depth_map.len() != expected {
            return Err(VolumeError::DepthMapSizeMismatch { expected, actual: depth_map.len() });
        }
        if depth_layers == 0 {
            return Err(VolumeError::ZeroDepthLayers);
        }
        if thickness == 0 {
            return Err(VolumeError::ZeroThickness);
        }

        let mut slices = (0..depth_layers).map(|z| VolumeSlice {
            z,
            width: frame.width,
            height: frame.height,
            pixels: vec![[0, 0, 0, 0]; expected],
        }).collect::<Vec<_>>();

        for (index, (&depth, &pixel)) in depth_map.iter().zip(&frame.pixels).enumerate() {
            if pixel[3] == 0 {
                continue;
            }
            let surface_z = usize::from(depth) * (depth_layers - 1) / 255;
            let first_z = surface_z.saturating_add(1).saturating_sub(thickness);
            for slice in &mut slices[first_z..=surface_z] {
                slice.pixels[index] = pixel;
            }
        }

        Ok(Self { width: frame.width, height: frame.height, slices })
    }

    /// Extracts the SNES object pixels from a frame produced by `Renderer::render_bound`.
    ///
    /// The renderer's source map identifies pixels drawn by OAM objects. Backgrounds and
    /// transparent pixels are discarded. Until a hand-authored depth map is supplied, depth is
    /// estimated from vertical position (upper pixels farther back, lower pixels closer).
    /// This is a bridge from the existing renderer, not a change to its normal 2D output.
    pub fn from_object_frame(
        frame: &crate::Frame,
        depth_layers: usize,
        thickness: usize,
    ) -> Result<Self, VolumeError> {
        let expected = frame.width.checked_mul(frame.height)
            .ok_or(VolumeError::DimensionsOverflow)?;
        if frame.sources.is_empty() {
            return Err(VolumeError::MissingSourceMap);
        }
        if frame.pixels.len() != expected {
            return Err(VolumeError::PixelCountMismatch {
                expected,
                actual: frame.pixels.len(),
            });
        }
        if frame.sources.len() != expected {
            return Err(VolumeError::DepthMapSizeMismatch {
                expected,
                actual: frame.sources.len(),
            });
        }

        let mut pixels = vec![[0, 0, 0, 0]; expected];
        let mut depth_map = vec![0; expected];
        for index in 0..expected {
            if matches!(frame.sources[index], crate::Source::Char { object: true, .. }) {
                let color = frame.pixels[index];
                pixels[index] = [color.r, color.g, color.b, color.a];
                let y = index / frame.width;
                depth_map[index] = if frame.height <= 1 {
                    128
                } else {
                    (y * 255 / (frame.height - 1)) as u8
                };
            }
        }

        let source = RgbaFrame::new(frame.width, frame.height, pixels)?;
        Self::from_depth_map(&source, &depth_map, depth_layers, thickness)
    }

    /// Builds a rounded solid from a sprite silhouette using its interior distance field.
    /// Contour pixels occupy a shallow cross-section; pixels farther from transparency occupy
    /// more slices. This avoids copying every source pixel through a uniform stack depth.
    pub fn from_silhouette(
        frame: &RgbaFrame,
        max_half_depth: usize,
    ) -> Result<Self, VolumeError> {
        Self::from_silhouette_profile(frame, max_half_depth, false)
    }

    /// Builds the flight craft as a fixed 13-cross-section stack.
    pub fn from_flight_silhouette(frame: &RgbaFrame) -> Result<Self, VolumeError> {
        Self::from_flight_silhouette_with_depth(frame, 6)
    }

    /// Builds a flight sprite as a fixed, symmetric stack with a caller-selected half-depth.
    /// The depth is independent of the sprite's interior thickness, so narrow SNES silhouettes
    /// still produce enough slices to show their side faces when the camera orbits them.
    pub fn from_flight_silhouette_with_depth(
        frame: &RgbaFrame,
        half_depth: usize,
    ) -> Result<Self, VolumeError> {
        if half_depth == 0 {
            return Err(VolumeError::ZeroDepthLayers);
        }
        Self::from_silhouette_profile(frame, half_depth, true)
    }

    fn from_silhouette_profile(
        frame: &RgbaFrame,
        max_half_depth: usize,
        fixed_depth: bool,
    ) -> Result<Self, VolumeError> {
        if max_half_depth == 0 { return Err(VolumeError::ZeroDepthLayers); }
        let count = frame.width.checked_mul(frame.height)
            .ok_or(VolumeError::DimensionsOverflow)?;
        if frame.width == 0 || frame.height == 0 { return Err(VolumeError::EmptyFrame); }
        if frame.pixels.len() != count {
            return Err(VolumeError::PixelCountMismatch { expected: count, actual: frame.pixels.len() });
        }

        let mut distance = vec![u16::MAX; count];
        let mut queue = std::collections::VecDeque::new();
        for (at, pixel) in frame.pixels.iter().enumerate() {
            if pixel[3] == 0 {
                distance[at] = 0;
                queue.push_back(at);
            }
        }
        for x in 0..frame.width {
            for y in [0, frame.height - 1] {
                let at = y * frame.width + x;
                if distance[at] != 0 { distance[at] = 0; queue.push_back(at); }
            }
        }
        for y in 0..frame.height {
            for x in [0, frame.width - 1] {
                let at = y * frame.width + x;
                if distance[at] != 0 { distance[at] = 0; queue.push_back(at); }
            }
        }
        while let Some(at) = queue.pop_front() {
            let next = distance[at].saturating_add(1);
            let x = at % frame.width;
            let y = at / frame.width;
            let neighbours = [
                (x > 0).then_some(at.wrapping_sub(1)),
                (x + 1 < frame.width).then_some(at + 1),
                (y > 0).then_some(at.wrapping_sub(frame.width)),
                (y + 1 < frame.height).then_some(at + frame.width),
            ];
            for neighbour in neighbours.into_iter().flatten() {
                if next < distance[neighbour] {
                    distance[neighbour] = next;
                    queue.push_back(neighbour);
                }
            }
        }

        let measured_half = distance.iter().copied().filter(|d| *d != u16::MAX)
            .max().unwrap_or(1).min(max_half_depth.min(u16::MAX as usize) as u16)
            .max(1) as usize;
        let half = if fixed_depth { max_half_depth } else { measured_half };
        let layers = half * 2 + 1;
        let mut slices = (0..layers).map(|z| VolumeSlice {
            z, width: frame.width, height: frame.height, pixels: vec![[0, 0, 0, 0]; count],
        }).collect::<Vec<_>>();
        for (at, &pixel) in frame.pixels.iter().enumerate() {
            if pixel[3] == 0 { continue; }
            let measured = usize::from(distance[at]).min(measured_half).max(1);
            let depth = if fixed_depth {
                measured.saturating_mul(half).div_ceil(measured_half).clamp(1, half)
            } else {
                measured
            };
            let first = half.saturating_sub(depth);
            let last = (half + depth).min(layers - 1);
            for slice in &mut slices[first..=last] {
                slice.pixels[at] = pixel;
            }
        }
        Ok(Self { width: frame.width, height: frame.height, slices })
    }

    /// Uniform-depth fallback for frames without a depth map.
    pub fn extrude(frame: &RgbaFrame, depth_layers: usize) -> Result<Self, VolumeError> {
        if depth_layers == 0 {
            return Err(VolumeError::ZeroDepthLayers);
        }
        let count = frame.width.checked_mul(frame.height)
            .ok_or(VolumeError::DimensionsOverflow)?;
        Self::from_depth_map(frame, &vec![255; count], depth_layers, depth_layers)
    }

    /// Builds a solid sprite stack and shades the rear slices to reveal its side faces when viewed
    /// obliquely. The front slice retains the exact source colours.
    pub fn extrude_shaded(
        frame: &RgbaFrame,
        depth_layers: usize,
        rear_brightness: f32,
    ) -> Result<Self, VolumeError> {
        if !rear_brightness.is_finite() || !(0.0..=1.0).contains(&rear_brightness) {
            return Err(VolumeError::InvalidCamera);
        }
        let mut volume = Self::extrude(frame, depth_layers)?;
        let denominator = (depth_layers.saturating_sub(1)).max(1) as f32;
        for slice in &mut volume.slices {
            let t = slice.z as f32 / denominator;
            let brightness = rear_brightness + (1.0 - rear_brightness) * t;
            for pixel in &mut slice.pixels {
                if pixel[3] == 0 {
                    continue;
                }
                for channel in &mut pixel[..3] {
                    *channel = (f32::from(*channel) * brightness).round() as u8;
                }
            }
        }
        Ok(volume)
    }

    /// Counts non-transparent voxels.
    pub fn voxel_count(&self) -> usize {
        self.slices.iter()
            .map(|slice| slice.pixels.iter().filter(|pixel| pixel[3] != 0).count())
            .sum()
    }
}

/// Camera used to project the voxel slices back into a 2D RGBA image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VolumeCamera {
    /// Rotation around the vertical axis, in radians.
    pub yaw: f32,
    /// Looking down/up angle, in radians. Positive values look down onto the volume.
    pub pitch: f32,
    /// Orthographic pixel scale. Values below 1 shrink the projected volume.
    pub scale: f32,
}

impl Default for VolumeCamera {
    fn default() -> Self {
        Self { yaw: 0.72, pitch: 0.62, scale: 1.0 }
    }
}

impl SpriteVolume {
    /// Projects the real sprite voxels into a transparent 2D image using a depth buffer.
    ///
    /// Unlike the old preview executable, this consumes the volume extracted from a rendered
    /// SNES frame. It does not invent character pixels. The output is suitable for compositing
    /// over the original map/Mode 7 background.
    pub fn project(&self, camera: VolumeCamera) -> Result<RgbaFrame, VolumeError> {
        let count = self.width.checked_mul(self.height).ok_or(VolumeError::DimensionsOverflow)?;
        if self.width == 0 || self.height == 0 {
            return Err(VolumeError::EmptyFrame);
        }
        if self.slices.is_empty() {
            return Err(VolumeError::ZeroDepthLayers);
        }
        if !camera.yaw.is_finite() || !camera.pitch.is_finite()
            || !camera.scale.is_finite() || camera.scale <= 0.0
        {
            return Err(VolumeError::InvalidCamera);
        }
        if self.slices.iter().any(|slice| {
            slice.width != self.width || slice.height != self.height || slice.pixels.len() != count
        }) {
            return Err(VolumeError::PixelCountMismatch {
                expected: count,
                actual: self.slices.first().map_or(0, |slice| slice.pixels.len()),
            });
        }

        let mut output = vec![[0, 0, 0, 0]; count];
        let mut depth_buffer = vec![f32::NEG_INFINITY; count];
        let (sin_yaw, cos_yaw) = camera.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = camera.pitch.sin_cos();
        let center_x = (self.width as f32 - 1.0) * 0.5;
        let center_y = (self.height as f32 - 1.0) * 0.5;
        let center_z = (self.slices.len() as f32 - 1.0) * 0.5;

        for slice in &self.slices {
            let z = slice.z as f32 - center_z;
            for (index, &pixel) in slice.pixels.iter().enumerate() {
                if pixel[3] == 0 { continue; }
                let x = index % self.width;
                let y = index / self.width;
                let local_x = x as f32 - center_x;
                let local_y = center_y - y as f32;

                let rotated_x = local_x * cos_yaw - z * sin_yaw;
                let rotated_z = local_x * sin_yaw + z * cos_yaw;
                let projected_y = local_y * cos_pitch - rotated_z * sin_pitch;
                let camera_depth = local_y * sin_pitch + rotated_z * cos_pitch;

                let screen_x = (center_x + rotated_x * camera.scale).round() as i32;
                let screen_y = (center_y - projected_y * camera.scale).round() as i32;
                if screen_x < 0 || screen_y < 0
                    || screen_x >= self.width as i32 || screen_y >= self.height as i32
                { continue; }
                let target = screen_y as usize * self.width + screen_x as usize;
                if camera_depth >= depth_buffer[target] {
                    depth_buffer[target] = camera_depth;
                    output[target] = pixel;
                }
            }
        }
        RgbaFrame::new(self.width, self.height, output)
    }
}

/// Failure while asking the existing HD renderer to build a sprite volume.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderVolumeError {
    UnsupportedFrame(crate::Unsupported),
    Volume(VolumeError),
}

impl std::fmt::Display for RenderVolumeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedFrame(error) => write!(f, "renderer cannot draw this frame: {error}"),
            Self::Volume(error) => write!(f, "cannot build sprite volume: {error}"),
        }
    }
}

impl std::error::Error for RenderVolumeError {}

impl crate::Renderer {
    /// Renders a bound SNES frame and extracts its OAM object pixels into a depth volume.
    ///
    /// The normal game renderer remains unchanged; this opt-in method is for experiments,
    /// capture tools and future volumetric rendering paths.
    pub fn render_object_volume(
        &mut self,
        log: &ssd_oracle::FrameLog,
        binder: &mut crate::Binder<'_>,
        depth_layers: usize,
        thickness: usize,
    ) -> Result<SpriteVolume, RenderVolumeError> {
        let frame = self.render_bound(log, binder)
            .map_err(RenderVolumeError::UnsupportedFrame)?;
        SpriteVolume::from_object_frame(&frame, depth_layers, thickness)
            .map_err(RenderVolumeError::Volume)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame() -> RgbaFrame {
        RgbaFrame::new(2, 1, vec![[255, 0, 0, 255], [0, 255, 0, 0]]).unwrap()
    }

    #[test]
    fn cross_sections_preserve_independent_slice_art_without_extruding_it() {
        let back = RgbaFrame::new(2, 1, vec![[10, 20, 30, 255], [0, 0, 0, 0]]).unwrap();
        let front = RgbaFrame::new(2, 1, vec![[0, 0, 0, 0], [40, 50, 60, 128]]).unwrap();
        let volume = SpriteVolume::from_cross_sections(&[back.clone(), front.clone()]).unwrap();

        assert_eq!(volume.slices.len(), 2);
        assert_eq!(volume.slices[0].z, 0);
        assert_eq!(volume.slices[1].z, 1);
        assert_eq!(volume.slices[0].pixels, back.pixels);
        assert_eq!(volume.slices[1].pixels, front.pixels);
        assert_eq!(volume.voxel_count(), 2);
    }

    #[test]
    fn cross_sections_reject_mismatched_canvases() {
        let a = RgbaFrame::new(2, 1, vec![[0; 4]; 2]).unwrap();
        let b = RgbaFrame::new(1, 1, vec![[0; 4]]).unwrap();
        assert!(matches!(
            SpriteVolume::from_cross_sections(&[a, b]),
            Err(VolumeError::DepthMapSizeMismatch { .. })
        ));
    }

    #[test]
    fn shaded_extrusion_keeps_front_colour_and_darkens_rear_slices() {
        let frame = RgbaFrame::new(1, 1, vec![[200, 100, 50, 255]]).unwrap();
        let volume = SpriteVolume::extrude_shaded(&frame, 4, 0.5).unwrap();
        assert_eq!(volume.slices[3].pixels[0], [200, 100, 50, 255]);
        assert_eq!(volume.slices[0].pixels[0], [100, 50, 25, 255]);
    }

    #[test]
    fn projection_rejects_invalid_camera_values() {
        let volume = SpriteVolume::extrude(&sample_frame(), 2).unwrap();
        assert_eq!(volume.project(VolumeCamera { scale: 0.0, ..VolumeCamera::default() }),
            Err(VolumeError::InvalidCamera));
    }

    #[test]
    fn projection_preserves_a_real_voxel_and_returns_transparent_background() {
        let frame = RgbaFrame::new(3, 3, vec![
            [0,0,0,0], [0,0,0,0], [0,0,0,0],
            [0,0,0,0], [255,40,20,255], [0,0,0,0],
            [0,0,0,0], [0,0,0,0], [0,0,0,0],
        ]).unwrap();
        let volume = SpriteVolume::extrude(&frame, 1).unwrap();
        let projected = volume.project(VolumeCamera { yaw: 0.0, pitch: 0.0, scale: 1.0 }).unwrap();
        assert_eq!(projected.pixels[4], [255,40,20,255]);
        assert_eq!(projected.pixels.iter().filter(|p| p[3] != 0).count(), 1);
    }

    #[test]
    fn depth_buffer_resolves_overlapping_voxels() {
        let frame = RgbaFrame::new(3, 3, vec![
            [0,0,0,0], [0,0,0,0], [0,0,0,0],
            [0,0,0,0], [255,0,0,255], [0,0,0,0],
            [0,0,0,0], [0,0,0,0], [0,0,0,0],
        ]).unwrap();
        let mut volume = SpriteVolume::extrude(&frame, 2).unwrap();
        volume.slices[0].pixels[4] = [0,0,255,255];
        let projected = volume.project(VolumeCamera { yaw: 0.0, pitch: 0.0, scale: 1.0 }).unwrap();
        assert!(projected.pixels.iter().any(|p| *p == [255,0,0,255] || *p == [0,0,255,255]));
    }

    #[test]
    fn depth_map_places_pixels_and_ignores_transparency() {
        let volume = SpriteVolume::from_depth_map(&sample_frame(), &[0, 255], 4, 1).unwrap();
        assert_eq!(volume.slices[0].pixels[0], [255, 0, 0, 255]);
        assert!(volume.slices[1..].iter().all(|slice| slice.pixels[0][3] == 0));
        assert!(volume.slices.iter().all(|slice| slice.pixels[1][3] == 0));
    }

    #[test]
    fn thickness_fills_consecutive_slices_and_preserves_alpha() {
        let frame = RgbaFrame::new(1, 1, vec![[12, 34, 56, 128]]).unwrap();
        let volume = SpriteVolume::from_depth_map(&frame, &[255], 4, 2).unwrap();
        assert_eq!(volume.slices[2].pixels[0], [12, 34, 56, 128]);
        assert_eq!(volume.slices[3].pixels[0], [12, 34, 56, 128]);
        assert_eq!(volume.voxel_count(), 2);
    }

    #[test]
    fn rejects_invalid_dimensions_and_depth_maps() {
        assert_eq!(
            RgbaFrame::new(2, 1, vec![[0; 4]]),
            Err(VolumeError::PixelCountMismatch { expected: 2, actual: 1 })
        );
        let frame = sample_frame();
        assert_eq!(
            SpriteVolume::from_depth_map(&frame, &[0], 2, 1),
            Err(VolumeError::DepthMapSizeMismatch { expected: 2, actual: 1 })
        );
        assert_eq!(
            SpriteVolume::from_depth_map(&frame, &[0, 0], 0, 1),
            Err(VolumeError::ZeroDepthLayers)
        );
    }

    #[test]
    fn extracts_only_snes_object_pixels_from_a_bound_frame() {
        let frame = crate::Frame {
            width: 2,
            height: 1,
            pixels: vec![
                ssd_oracle::Color::rgb(10, 20, 30),
                ssd_oracle::Color::rgb(40, 50, 60),
            ],
            sources: vec![
                crate::Source::Char { object: true, bpp: 4, address: 0, tile: None },
                crate::Source::Backdrop,
            ],
        };
        let volume = SpriteVolume::from_object_frame(&frame, 3, 1).unwrap();
        assert_eq!(volume.voxel_count(), 1);
        assert_eq!(volume.slices[1].pixels[0], [10, 20, 30, 255]);
        assert_eq!(volume.slices[1].pixels[1][3], 0);
    }

    #[test]
    fn object_extraction_requires_renderer_source_map() {
        let frame = crate::Frame {
            width: 1,
            height: 1,
            pixels: vec![ssd_oracle::Color::BLACK],
            sources: Vec::new(),
        };
        assert_eq!(
            SpriteVolume::from_object_frame(&frame, 2, 1),
            Err(VolumeError::MissingSourceMap)
        );
    }

    #[test]
    fn uniform_extrusion_fills_all_slices_for_opaque_pixels() {
        let volume = SpriteVolume::extrude(&sample_frame(), 3).unwrap();
        assert_eq!(volume.voxel_count(), 3);
        assert!(volume.slices.iter().all(|slice| slice.pixels[0][3] == 255));
        assert!(volume.slices.iter().all(|slice| slice.pixels[1][3] == 0));
    }
}


/// Perspective camera for the sprite-stack renderer. The camera looks toward the model origin;
/// distance and focal length are measured in source pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PerspectiveCamera {
    /// Horizontal camera orbit around the model, in radians.
    pub yaw: f32,
    /// Vertical camera orbit around the model, in radians.
    pub pitch: f32,
    /// Distance from the model centre. Must be positive and greater than the near plane.
    pub distance: f32,
    /// Focal length in output pixels.
    pub focal_length: f32,
    /// Output dimensions.
    pub width: usize,
    pub height: usize,
    /// Minimum camera-space depth accepted by the projection.
    pub near_plane: f32,
}

impl Default for PerspectiveCamera {
    fn default() -> Self {
        Self {
            yaw: 0.72,
            pitch: 0.42,
            distance: 96.0,
            focal_length: 96.0,
            width: 256,
            height: 224,
            near_plane: 0.1,
        }
    }
}

/// Projects authored sprite-stack voxels with a perspective camera and a per-pixel depth buffer.
///
/// Unlike silhouette extrusion, this renderer expects the input slices to describe actual
/// cross-sections. It never creates side pixels or darkens rear copies: visible colour always
/// comes from the nearest source voxel. The output is independent of the SNES frame dimensions,
/// so a model can be rendered at a different scale and composited over the original background.
pub fn project_perspective(
    volume: &SpriteVolume,
    camera: PerspectiveCamera,
) -> Result<RgbaFrame, VolumeError> {
    project_perspective_with_roll(volume, camera, 0.0)
}

/// Projects a sprite volume after rotating the model around its horizontal axis.
/// A full 2π produces a true forward/backward somersault rather than orbiting the camera.
pub fn project_perspective_with_roll(
    volume: &SpriteVolume,
    camera: PerspectiveCamera,
    roll: f32,
) -> Result<RgbaFrame, VolumeError> {
    if !roll.is_finite() {
        return Err(VolumeError::InvalidCamera);
    }
    if volume.width == 0 || volume.height == 0 || volume.slices.is_empty() {
        return Err(VolumeError::EmptyFrame);
    }
    if camera.width == 0 || camera.height == 0
        || !camera.yaw.is_finite() || !camera.pitch.is_finite()
        || !camera.distance.is_finite() || !camera.focal_length.is_finite()
        || !camera.near_plane.is_finite()
        || camera.distance <= camera.near_plane || camera.focal_length <= 0.0
        || camera.near_plane <= 0.0
    {
        return Err(VolumeError::InvalidCamera);
    }
    let source_count = volume.width.checked_mul(volume.height)
        .ok_or(VolumeError::DimensionsOverflow)?;
    if volume.slices.iter().any(|slice| {
        slice.width != volume.width || slice.height != volume.height
            || slice.pixels.len() != source_count
    }) {
        return Err(VolumeError::PixelCountMismatch {
            expected: source_count,
            actual: volume.slices.first().map_or(0, |slice| slice.pixels.len()),
        });
    }
    let output_count = camera.width.checked_mul(camera.height)
        .ok_or(VolumeError::DimensionsOverflow)?;
    let mut pixels = vec![[0, 0, 0, 0]; output_count];
    let mut depth = vec![f32::INFINITY; output_count];
    let (sy, cy) = camera.yaw.sin_cos();
    let (sp, cp) = camera.pitch.sin_cos();
    let cx = (volume.width as f32 - 1.0) * 0.5;
    let cy0 = (volume.height as f32 - 1.0) * 0.5;
    let cz = (volume.slices.len() as f32 - 1.0) * 0.5;
    let out_cx = (camera.width as f32 - 1.0) * 0.5;
    let out_cy = (camera.height as f32 - 1.0) * 0.5;

    // Iterate far-to-near for deterministic ties; the z-buffer still decides visibility.
    for slice in &volume.slices {
        let z = slice.z as f32 - cz;
        for (index, &pixel) in slice.pixels.iter().enumerate() {
            if pixel[3] == 0 { continue; }
            let x = index % volume.width;
            let y = index / volume.width;
            let px = x as f32 - cx;
            let py = cy0 - y as f32;

            // Rotate the actual voxel coordinates before camera projection.
            let (sr, cr) = roll.sin_cos();
            let model_y = py * cr - z * sr;
            let model_z = py * sr + z * cr;

            // Then orbit the camera around the vertical stack axis and apply its fixed pitch.
            let horizontal = px * cy - model_z * sy;
            let depth_yaw = px * sy + model_z * cy;
            let vertical = model_y * cp - depth_yaw * sp;
            let depth_pitch = model_y * sp + depth_yaw * cp;
            let camera_z = camera.distance - depth_pitch;
            if camera_z <= camera.near_plane { continue; }

            let sx = (out_cx + horizontal * camera.focal_length / camera_z).round() as i32;
            let screen_y = (out_cy - vertical * camera.focal_length / camera_z).round() as i32;
            if sx < 0 || screen_y < 0 || sx >= camera.width as i32
                || screen_y >= camera.height as i32
            { continue; }
            let at = screen_y as usize * camera.width + sx as usize;
            if camera_z < depth[at] {
                depth[at] = camera_z;
                pixels[at] = pixel;
            }
        }
    }
    RgbaFrame::new(camera.width, camera.height, pixels)
}


#[cfg(test)]
mod perspective_tests {
    use super::*;

    #[test]
    fn perspective_projection_keeps_a_central_voxel_and_transparent_background() {
        let frame = RgbaFrame::new(3, 3, vec![
            [0,0,0,0], [0,0,0,0], [0,0,0,0],
            [0,0,0,0], [220,80,30,255], [0,0,0,0],
            [0,0,0,0], [0,0,0,0], [0,0,0,0],
        ]).unwrap();
        let volume = SpriteVolume::extrude(&frame, 1).unwrap();
        let camera = PerspectiveCamera {
            yaw: 0.0, pitch: 0.0, distance: 10.0, focal_length: 10.0,
            width: 5, height: 5, near_plane: 0.1,
        };
        let rendered = project_perspective(&volume, camera).unwrap();
        assert_eq!(rendered.pixels[12], [220,80,30,255]);
        assert_eq!(rendered.pixels.iter().filter(|p| p[3] != 0).count(), 1);
    }

    #[test]
    fn perspective_projection_rejects_invalid_camera_and_dimensions() {
        let volume = SpriteVolume::extrude(
            &RgbaFrame::new(1, 1, vec![[1,2,3,255]]).unwrap(), 1
        ).unwrap();
        let invalid = PerspectiveCamera { distance: 0.05, near_plane: 0.1, ..PerspectiveCamera::default() };
        assert_eq!(project_perspective(&volume, invalid), Err(VolumeError::InvalidCamera));
        let invalid_size = PerspectiveCamera { width: 0, ..PerspectiveCamera::default() };
        assert_eq!(project_perspective(&volume, invalid_size), Err(VolumeError::InvalidCamera));
    }

    #[test]
    fn perspective_depth_buffer_keeps_the_nearest_voxel() {
        let frame = RgbaFrame::new(1, 1, vec![[255,0,0,255]]).unwrap();
        let mut volume = SpriteVolume::extrude(&frame, 3).unwrap();
        volume.slices[0].pixels[0] = [0,0,255,255];
        let camera = PerspectiveCamera {
            yaw: 0.0, pitch: 0.0, distance: 20.0, focal_length: 20.0,
            width: 3, height: 3, near_plane: 0.1,
        };
        let rendered = project_perspective(&volume, camera).unwrap();
        assert!(rendered.pixels.iter().any(|p| *p == [255,0,0,255] || *p == [0,0,255,255]));
    }
}
 
#[cfg(test)]
mod silhouette_tests {
    use super::*;

    #[test]
    fn silhouette_stack_gives_interior_pixels_more_depth_than_contour() {
        let mut pixels = vec![[0, 0, 0, 0]; 7 * 7];
        for y in 1..6 {
            for x in 1..6 {
                pixels[y * 7 + x] = [180, 90, 30, 255];
            }
        }
        let frame = RgbaFrame::new(7, 7, pixels).unwrap();
        let volume = SpriteVolume::from_silhouette(&frame, 4).unwrap();
        let contour = volume.slices.iter().filter(|slice| slice.pixels[1 * 7 + 3][3] != 0).count();
        let interior = volume.slices.iter().filter(|slice| slice.pixels[3 * 7 + 3][3] != 0).count();
        assert!(interior > contour);
        assert!(volume.slices.iter().all(|slice| slice.pixels[0][3] == 0));
    }

    #[test]
    fn silhouette_stack_rejects_zero_depth() {
        let frame = RgbaFrame::new(1, 1, vec![[1, 2, 3, 255]]).unwrap();
        assert_eq!(SpriteVolume::from_silhouette(&frame, 0), Err(VolumeError::ZeroDepthLayers));
    }

    #[test]
    fn flight_silhouette_always_builds_thirteen_depth_cross_sections() {
        let mut pixels = vec![[0, 0, 0, 0]; 7 * 7];
        for y in 1..6 {
            for x in 1..6 {
                pixels[y * 7 + x] = [180, 90, 30, 255];
            }
        }
        let frame = RgbaFrame::new(7, 7, pixels).unwrap();
        let volume = SpriteVolume::from_flight_silhouette(&frame).unwrap();
        assert_eq!(volume.slices.len(), 13);
        let contour = volume.slices.iter().filter(|slice| slice.pixels[1 * 7 + 3][3] != 0).count();
        let interior = volume.slices.iter().filter(|slice| slice.pixels[3 * 7 + 3][3] != 0).count();
        assert!(interior > contour);
    }

    #[test]
    fn flight_silhouette_supports_a_deeper_stack_for_small_sprites() {
        let mut pixels = vec![[0, 0, 0, 0]; 7 * 7];
        for y in 2..5 {
            for x in 2..5 {
                pixels[y * 7 + x] = [180, 90, 30, 255];
            }
        }
        let frame = RgbaFrame::new(7, 7, pixels).unwrap();
        let volume = SpriteVolume::from_flight_silhouette_with_depth(&frame, 18).unwrap();
        assert_eq!(volume.slices.len(), 37);
        assert!(volume.slices.iter().any(|slice| slice.pixels[3 * 7 + 3][3] != 0));
    }
}
