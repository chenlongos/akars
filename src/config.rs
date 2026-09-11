//! 配置文件加载：robot.toml (机器人/摄像头/电机/TPU/Web) + arm.json (机械臂运动点位)。

use crate::arm::ArmMotionConfig;
use crate::motor::MotorConfig;
use crate::tpu::InferenceConfig;
use serde::Deserialize;
use std::net::SocketAddr;
use std::path::Path;

/// 从 TOML 文件加载的机器人配置顶层结构。
/// 每个 section 对应用户 robot.example.toml 中的同名块。
#[derive(Debug, Deserialize)]
pub struct RobotConfigFile {
    #[serde(default)]
    pub camera: CameraSection,
    #[serde(default)]
    pub motor: MotorSection,
    #[serde(default)]
    pub arm: ArmSection,
    #[serde(default, rename = "tpu")]
    pub tpu_section: TpuSection,
    #[serde(default)]
    pub web: WebSection,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct CameraSection {
    #[serde(default = "default_camera_device")]
    pub device: String,
    #[serde(default = "default_camera_width")]
    pub frame_width: u32,
    #[serde(default = "default_camera_height")]
    pub frame_height: u32,
}

impl Default for CameraSection {
    fn default() -> Self {
        Self {
            device: default_camera_device(),
            frame_width: default_camera_width(),
            frame_height: default_camera_height(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct MotorSection {
    #[serde(default = "default_motor_port")]
    pub port: String,
    #[serde(default = "default_baudrate")]
    pub baudrate: i32,
    #[serde(default)]
    pub speed_scale: Option<i32>,
    #[serde(default)]
    pub ppr: Option<u16>,
    #[serde(default)]
    pub pwm_freq: Option<u16>,
    #[serde(default)]
    pub min_speed: Option<i32>,
}

impl Default for MotorSection {
    fn default() -> Self {
        Self {
            port: default_motor_port(),
            baudrate: default_baudrate(),
            speed_scale: None,
            ppr: None,
            pwm_freq: None,
            min_speed: None,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct ArmSection {
    #[serde(default = "default_arm_port")]
    pub port: String,
    #[serde(default = "default_baudrate")]
    pub baudrate: i32,
}

impl Default for ArmSection {
    fn default() -> Self {
        Self {
            port: default_arm_port(),
            baudrate: default_baudrate(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct TpuSection {
    #[serde(default = "default_one")]
    pub classes_num: i32,
    #[serde(default = "default_half")]
    pub confidence_threshold: f32,
    #[serde(default = "default_half")]
    pub iou_threshold: f32,
}

impl Default for TpuSection {
    fn default() -> Self {
        Self {
            classes_num: default_one(),
            confidence_threshold: default_half(),
            iou_threshold: default_half(),
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub struct WebSection {
    #[serde(default = "default_listen")]
    pub listen: SocketAddr,
    #[serde(default)]
    pub mock: bool,
}

impl Default for WebSection {
    fn default() -> Self {
        Self {
            listen: default_listen(),
            mock: false,
        }
    }
}

// ── 默认值辅助函数 ───────────────────────────────────────────────────────────

fn default_camera_device() -> String {
    "/dev/video0".to_string()
}
fn default_camera_width() -> u32 {
    640
}
fn default_camera_height() -> u32 {
    480
}
fn default_motor_port() -> String {
    "/dev/ttyS1".to_string()
}
fn default_arm_port() -> String {
    "/dev/ttyS2".to_string()
}
fn default_baudrate() -> i32 {
    115200
}
fn default_one() -> i32 {
    1
}
fn default_half() -> f32 {
    0.5
}
fn default_listen() -> SocketAddr {
    SocketAddr::from(([0, 0, 0, 0], 8080))
}

// ── 从 section 转换到模块内使用的 config struct ───────────────────────────────

impl RobotConfigFile {
    pub fn to_motor_config(&self) -> MotorConfig {
        let mut cfg = MotorConfig::default();
        cfg.device = self.motor.port.clone();
        cfg.baudrate = self.motor.baudrate;
        if let Some(v) = self.motor.speed_scale {
            cfg.speed_scale = v;
        }
        if let Some(v) = self.motor.ppr {
            cfg.ppr = v;
        }
        if let Some(v) = self.motor.pwm_freq {
            cfg.pwm_freq = v;
        }
        if let Some(v) = self.motor.min_speed {
            cfg.min_speed = v;
        }
        cfg
    }

    pub fn to_inference_config(&self) -> InferenceConfig {
        InferenceConfig {
            classes_num: self.tpu_section.classes_num,
            confidence_threshold: self.tpu_section.confidence_threshold,
            iou_threshold: self.tpu_section.iou_threshold,
        }
    }
}

// ── 加载函数 ─────────────────────────────────────────────────────────────────

/// 从 TOML 文件加载 robot 配置。文件不存在时返回默认值。
pub fn load_robot_config<P: AsRef<Path>>(path: P) -> Result<RobotConfigFile, String> {
    let path = path.as_ref();
    match std::fs::read_to_string(path) {
        Ok(text) => {
            // 给 TOML section 加 rename_all = "UPPERCASE" 会更方便地匹配字段。
            // 我们逐 section 处理，因为 web section 字段名是小写 listen。
            let cfg: RobotConfigFile = toml::from_str(&text).map_err(|e| {
                format!("failed to parse robot config {}: {e}", path.display())
            })?;
            Ok(cfg)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("[config] robot config {} not found, using defaults", path.display());
            Ok(RobotConfigFile {
                camera: CameraSection::default(),
                motor: MotorSection::default(),
                arm: ArmSection::default(),
                tpu_section: TpuSection::default(),
                web: WebSection::default(),
            })
        }
        Err(e) => Err(format!(
            "failed to read robot config {}: {e}",
            path.display()
        )),
    }
}

/// 从 JSON 文件加载 arm 运动配置。文件不存在时返回 Default。
pub fn load_arm_motion_config<P: AsRef<Path>>(path: P) -> Result<ArmMotionConfig, String> {
    let path = path.as_ref();
    match std::fs::read_to_string(path) {
        Ok(text) => {
            let cfg: ArmMotionConfig = serde_json::from_str(&text).map_err(|e| {
                format!("failed to parse arm motion config {}: {e}", path.display())
            })?;
            Ok(cfg)
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            eprintln!(
                "[config] arm motion config {} not found, using defaults",
                path.display()
            );
            Ok(ArmMotionConfig::default())
        }
        Err(e) => Err(format!(
            "failed to read arm motion config {}: {e}",
            path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_robot_toml_minimal() {
        let toml_str = r#"
[camera]
DEVICE = "/dev/video0"
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

[motor]
PORT = "/dev/ttyS1"
BAUDRATE = 115200
SPEED_SCALE = 85
PPR = 4680
PWM_FREQ = 20000
MIN_SPEED = 15

[arm]
PORT = "/dev/ttyS2"
BAUDRATE = 115200

[tpu]
CLASSES_NUM = 1
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

[web]
LISTEN = "0.0.0.0:8080"
MOCK = false
"#;
        let cfg: RobotConfigFile = toml::from_str(toml_str).unwrap();
        assert_eq!(cfg.camera.device, "/dev/video0");
        assert_eq!(cfg.motor.port, "/dev/ttyS1");
        assert_eq!(cfg.motor.speed_scale, Some(85));
        assert_eq!(cfg.arm.port, "/dev/ttyS2");
        assert_eq!(cfg.tpu_section.classes_num, 1);
        assert_eq!(cfg.tpu_section.confidence_threshold, 0.5);
        assert_eq!(cfg.tpu_section.iou_threshold, 0.5);
        assert_eq!(cfg.web.listen.to_string(), "0.0.0.0:8080");
    }

    #[test]
    fn loads_arm_json() {
        let json_str = r#"{
  "grab_position": { "servo0": 245, "servo1": 180 },
  "lift_position": { "servo0": 200, "servo1": 180 },
  "gripper_open": 150,
  "gripper_close": 90
}"#;
        let cfg: ArmMotionConfig = serde_json::from_str(json_str).unwrap();
        assert_eq!(cfg.grab_position.servo0, 245.0);
        assert_eq!(cfg.lift_position.servo0, 200.0);
        assert_eq!(cfg.gripper_open, 150.0);
        assert_eq!(cfg.gripper_close, 90.0);
    }

    #[test]
    fn motor_section_maps_to_config() {
        let cfg: RobotConfigFile = toml::from_str(
            r#"
[motor]
PORT = "/dev/ttyS1"
SPEED_SCALE = 85
PPR = 4680
PWM_FREQ = 20000
MIN_SPEED = 15
"#,
        )
        .unwrap();
        let mc = cfg.to_motor_config();
        assert_eq!(mc.device, "/dev/ttyS1");
        assert_eq!(mc.speed_scale, 85);
        assert_eq!(mc.ppr, 4680);
        assert_eq!(mc.pwm_freq, 20000);
        assert_eq!(mc.min_speed, 15);
    }
}
