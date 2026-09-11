use crate::serial::SerialPort;
use serde::{Deserialize, Serialize};
use std::io;
use std::thread;
use std::time::Duration;

const ANGLE_MAX: f32 = 270.0;
const PULSE_MIN: i32 = 500;
const PULSE_MAX: i32 = 2500;

/// 单个双舵机位置（servo0 + servo1）。
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ArmPosition {
    pub servo0: f32,
    pub servo1: f32,
}

/// 机械臂运动点位配置（从 arm.json 加载）。
#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
pub struct ArmMotionConfig {
    /// 抓取位置：机械臂伸向物体。
    pub grab_position: ArmPosition,
    /// 抬起位置：抓住物体后抬起。同时也作为「预备/回位」位置使用。
    pub lift_position: ArmPosition,
    /// 夹爪打开角度（servo2）。
    pub gripper_open: f32,
    /// 夹爪关闭角度（servo2）。
    pub gripper_close: f32,
}

impl Default for ArmMotionConfig {
    fn default() -> Self {
        Self {
            grab_position: ArmPosition {
                servo0: 245.0,
                servo1: 185.0,
            },
            lift_position: ArmPosition {
                servo0: 180.0,
                servo1: 100.0,
            },
            gripper_open: 140.0,
            gripper_close: 90.0,
        }
    }
}

pub struct Arm {
    port: SerialPort,
    motion: ArmMotionConfig,
}

impl Arm {
    pub fn open(device: &str, baudrate: i32, motion: ArmMotionConfig) -> io::Result<Self> {
        crate::pinmux::configure_for_device(device);
        Ok(Self {
            port: SerialPort::open(device, baudrate, false)?,
            motion,
        })
    }

    pub fn set_angle(&mut self, servo_id: i32, angle: f32, time_ms: i32) {
        let clamped = angle.clamp(0.0, ANGLE_MAX);
        let pulse = angle_to_pulse(clamped);
        self.send_command(&format!("#{servo_id:03}P{pulse:04}T{time_ms}!"));
    }

    pub fn release_torque(&mut self, servo_id: i32) {
        self.send_command(&format!("#{servo_id:03}PULK"));
    }

    pub fn restore_torque(&mut self, servo_id: i32) {
        self.send_command(&format!("#{servo_id:03}PULR"));
    }

    pub fn motion_config(&self) -> &ArmMotionConfig {
        &self.motion
    }

    pub fn set_motion_config(&mut self, motion: ArmMotionConfig) {
        self.motion = motion;
    }

    /// 完整抓取动作：伸臂 → 夹取 → 抬起。
    pub fn grab(&mut self) {
        let m = self.motion;
        self.set_angle(0, m.grab_position.servo0, 1000);
        self.set_angle(1, m.grab_position.servo1, 1000);
        self.set_angle(2, m.gripper_open, 1000);
        sleep_ms(1500);

        self.set_angle(2, m.gripper_close, 1000);
        sleep_ms(1000);

        self.set_angle(0, m.lift_position.servo0, 1000);
        self.set_angle(1, m.lift_position.servo1, 1000);
        sleep_ms(1200);
    }

    /// 回到预备/回位位置（用 lift_position 作为 ready）。
    pub fn release_pos(&mut self) {
        let m = self.motion;
        self.set_angle(0, m.lift_position.servo0, 1000);
        self.set_angle(1, m.lift_position.servo1, 1000);
        self.set_angle(2, m.gripper_open, 1000);
    }

    /// 只张开夹爪，不移动主体。
    pub fn release(&mut self) {
        self.set_angle(2, self.motion.gripper_open, 1000);
    }

    /// 回到预备/回位位置（同 release_pos）。
    pub fn grab_pos(&mut self) {
        self.release_pos();
    }

    /// 展示/举起：抬臂 + 夹取姿态。
    pub fn show(&mut self) {
        let m = self.motion;
        self.set_angle(0, m.lift_position.servo0, 1000);
        self.set_angle(1, m.lift_position.servo1, 1000);
        self.set_angle(2, m.gripper_close, 1000);
    }

    fn send_command(&mut self, command: &str) {
        if let Err(err) = self.port.write_all_drain(command.as_bytes()) {
            eprintln!("[arm] write failed: {err}");
        }
    }
}

fn angle_to_pulse(angle: f32) -> i32 {
    ((500.0 + (angle / ANGLE_MAX) * 2000.0).round() as i32).clamp(PULSE_MIN, PULSE_MAX)
}

fn sleep_ms(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

#[cfg(test)]
mod tests {
    use super::angle_to_pulse;

    #[test]
    fn converts_angles_to_pulses() {
        assert_eq!(angle_to_pulse(0.0), 500);
        assert_eq!(angle_to_pulse(270.0), 2500);
        assert_eq!(angle_to_pulse(135.0), 1500);
    }
}
