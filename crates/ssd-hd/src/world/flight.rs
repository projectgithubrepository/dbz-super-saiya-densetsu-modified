//! Deterministic 3D flight controller layered over the original 2D map plane.
//! X/Z map to legacy map X/Y; world Y is visual altitude. This module is not yet
//! connected to the cartridge's movement, collision, event, or flight state machine.

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Vec3 { pub x: f32, pub y: f32, pub z: f32 }

impl Vec3 {
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self { Self { x, y, z } }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct FlightInput {
    pub turn: f32,
    pub pitch: f32,
    pub accelerate: f32,
    pub brake: f32,
    pub vertical: f32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightConfig {
    pub acceleration: f32,
    pub max_speed: f32,
    pub drag: f32,
    pub brake_strength: f32,
    pub turn_rate: f32,
    pub pitch_rate: f32,
    pub vertical_speed: f32,
    pub max_altitude: f32,
    pub map_units_per_world_unit: f32,
}

impl Default for FlightConfig {
    fn default() -> Self {
        Self {
            acceleration: 18.0, max_speed: 72.0, drag: 0.8,
            brake_strength: 36.0, turn_rate: 1.8, pitch_rate: 1.2,
            vertical_speed: 24.0, max_altitude: 160.0,
            map_units_per_world_unit: 1.0,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightState3D {
    pub position: Vec3,
    pub heading: f32,
    pub pitch: f32,
    pub speed: f32,
}

impl Default for FlightState3D {
    fn default() -> Self {
        Self { position: Vec3::default(), heading: 0.0, pitch: 0.0, speed: 0.0 }
    }
}

impl FlightState3D {
    #[must_use]
    pub fn from_legacy(map_x: u16, map_y: u16) -> Self {
        Self { position: Vec3::new(f32::from(map_x), 0.0, f32::from(map_y)), ..Self::default() }
    }

    /// Advances the model; delta time is capped at 100 ms to avoid pause-induced jumps.
    pub fn update(&mut self, input: FlightInput, dt_seconds: f32, config: FlightConfig) {
        let dt = if dt_seconds.is_finite() { dt_seconds.clamp(0.0, 0.1) } else { 0.0 };
        if dt == 0.0 { return; }

        self.heading = (self.heading
            + input.turn.clamp(-1.0, 1.0) * config.turn_rate.max(0.0) * dt)
            .rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch
            + input.pitch.clamp(-1.0, 1.0) * config.pitch_rate.max(0.0) * dt)
            .clamp(-1.2, 1.2);

        let thrust = input.accelerate.clamp(0.0, 1.0) * config.acceleration.max(0.0);
        let braking = input.brake.clamp(0.0, 1.0) * config.brake_strength.max(0.0);
        self.speed = (self.speed + (thrust - braking - config.drag.max(0.0) * self.speed) * dt)
            .clamp(0.0, config.max_speed.max(0.0));

        let horizontal_speed = self.speed * self.pitch.cos();
        self.position.x += self.heading.cos() * horizontal_speed * dt;
        self.position.z += self.heading.sin() * horizontal_speed * dt;
        self.position.y = (self.position.y
            + (self.speed * self.pitch.sin()
                + input.vertical.clamp(-1.0, 1.0) * config.vertical_speed.max(0.0)) * dt)
            .clamp(0.0, config.max_altitude.max(0.0));
    }

    /// Projects X/Z back to legacy map coordinates, with explicit scale and map bounds.
    #[must_use]
    pub fn to_legacy_map(&self, config: FlightConfig, map_size: (i32, i32)) -> (u16, u16) {
        let scale = if config.map_units_per_world_unit.is_finite()
            && config.map_units_per_world_unit > 0.0 {
            config.map_units_per_world_unit
        } else { 1.0 };
        let x = (self.position.x * scale).round().clamp(0.0, map_size.0.max(0) as f32);
        let y = (self.position.z * scale).round().clamp(0.0, map_size.1.max(0) as f32);
        (x as u16, y as u16)
    }

    #[must_use]
    pub fn chase_camera(&self, distance: f32, height: f32, look_ahead: f32) -> FlightCamera {
        let forward = Vec3::new(self.heading.cos(), self.pitch.sin(), self.heading.sin());
        FlightCamera {
            position: Vec3::new(
                self.position.x - forward.x * distance,
                self.position.y + height - forward.y * distance,
                self.position.z - forward.z * distance,
            ),
            target: Vec3::new(
                self.position.x + forward.x * look_ahead,
                self.position.y + forward.y * look_ahead,
                self.position.z + forward.z * look_ahead,
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlightCamera { pub position: Vec3, pub target: Vec3 }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn acceleration_moves_forward_deterministically() {
        let mut s = FlightState3D::default();
        let i = FlightInput { accelerate: 1.0, ..FlightInput::default() };
        for _ in 0..10 { s.update(i, 0.1, FlightConfig::default()); }
        assert!(s.speed > 0.0 && s.position.x > 0.0);
        assert_eq!(s.position.z, 0.0);
    }

    #[test]
    fn turning_changes_horizontal_direction() {
        let mut s = FlightState3D { speed: 20.0, ..FlightState3D::default() };
        s.update(FlightInput { turn: 1.0, ..FlightInput::default() }, 0.1,
            FlightConfig { drag: 0.0, ..FlightConfig::default() });
        assert!(s.heading > 0.0 && s.position.z > 0.0);
    }

    #[test]
    fn pitch_and_vertical_control_change_altitude() {
        let mut s = FlightState3D { speed: 30.0, ..FlightState3D::default() };
        s.update(FlightInput { pitch: 1.0, vertical: 1.0, ..FlightInput::default() },
            0.1, FlightConfig::default());
        assert!(s.position.y > 0.0 && s.position.x > 0.0);
    }

    #[test]
    fn braking_reduces_speed_and_ground_clamps_altitude() {
        let mut s = FlightState3D { speed: 30.0, ..FlightState3D::default() };
        s.update(FlightInput { brake: 1.0, vertical: -1.0, ..FlightInput::default() },
            0.1, FlightConfig::default());
        assert!(s.speed < 30.0);
        assert_eq!(s.position.y, 0.0);
    }

    #[test]
    fn legacy_projection_uses_x_and_z_and_clamps_to_map() {
        let mut s = FlightState3D::from_legacy(20, 30);
        s.position = Vec3::new(900.0, 40.0, -4.0);
        assert_eq!(s.to_legacy_map(FlightConfig::default(), (512, 256)), (512, 0));
    }

    #[test]
    fn invalid_delta_time_does_not_move_state() {
        let mut s = FlightState3D::default();
        s.update(FlightInput { accelerate: 1.0, ..FlightInput::default() },
            f32::NAN, FlightConfig::default());
        assert_eq!(s, FlightState3D::default());
    }

    #[test]
    fn chase_camera_is_behind_heading() {
        let s = FlightState3D { position: Vec3::new(10.0, 5.0, 20.0), ..FlightState3D::default() };
        let c = s.chase_camera(8.0, 3.0, 12.0);
        assert_eq!(c.position, Vec3::new(2.0, 8.0, 20.0));
        assert_eq!(c.target, Vec3::new(22.0, 5.0, 20.0));
    }
}
