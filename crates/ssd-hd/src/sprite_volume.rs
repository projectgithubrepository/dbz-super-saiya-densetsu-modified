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
        }
    }
}

impl std::error::Error for VolumeError {}

impl SpriteVolume {
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

    /// Uniform-depth fallback for frames without a depth map.
    pub fn extrude(frame: &RgbaFrame, depth_layers: usize) -> Result<Self, VolumeError> {
        if depth_layers == 0 {
            return Err(VolumeError::ZeroDepthLayers);
        }
        let count = frame.width.checked_mul(frame.height)
            .ok_or(VolumeError::DimensionsOverflow)?;
        Self::from_depth_map(frame, &vec![255; count], depth_layers, depth_layers)
    }

    /// Counts non-transparent voxels.
    pub fn voxel_count(&self) -> usize {
        self.slices.iter()
            .map(|slice| slice.pixels.iter().filter(|pixel| pixel[3] != 0).count())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_frame() -> RgbaFrame {
        RgbaFrame::new(2, 1, vec![[255, 0, 0, 255], [0, 255, 0, 0]]).unwrap()
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
    fn uniform_extrusion_fills_all_slices_for_opaque_pixels() {
        let volume = SpriteVolume::extrude(&sample_frame(), 3).unwrap();
        assert_eq!(volume.voxel_count(), 3);
        assert!(volume.slices.iter().all(|slice| slice.pixels[0][3] == 255));
        assert!(volume.slices.iter().all(|slice| slice.pixels[1][3] == 0));
    }
}
