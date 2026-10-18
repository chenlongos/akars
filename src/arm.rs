use crate::serial::SerialPort;
use std::io;
use std::thread;
use std::time::Duration;

// 舵机角度实测标定。整条抓取序列就四组姿势,对照着改:
//
//   姿势              舵机0   舵机1   舵机2(爪)
//   待抓/归位          READY   READY   OPEN       grab_pos()
//   伸到最低           GRAB    GRAB    OPEN       grab() 第 1 步
//   夹住(仍在最低位)   GRAB    GRAB    CLOSE      grab() 第 2 步
//   抬起/持球找桶      LIFT    LIFT    CLOSE      grab() 第 3 步,一直保持到桶口才 release()
//
// 注意"抬"几乎全是舵机1 干的(READY 185 → LIFT),舵机0 只是从 GRAB 回到 LIFT 位。
const ID2_ANGLE_OPEN: f32 = 140.0;
const ID2_ANGLE_CLOSE: f32 = 90.0;
const ANGLE_MAX: f32 = 270.0;
const PULSE_MIN: i32 = 500;
const PULSE_MAX: i32 = 2500;

const SERVO0_READY: f32 = 150.0;
const SERVO1_READY: f32 = 185.0;
const SERVO0_GRAB: f32 = 245.0;
const SERVO1_GRAB: f32 = 185.0;
// LIFT 当前与 READY 同值 —— 是有意的,不是复制粘贴漏改:持球姿势就是归位姿势。
const SERVO0_LIFT: f32 = 150.0;
const SERVO1_LIFT: f32 = 185.0;

pub struct Arm {
    port: Option<SerialPort>,
}

impl Arm {
    /// 无硬件桩:port=None,所有指令静默丢弃。仅供测试使用。
    #[cfg(test)]
    pub fn mock() -> Self {
        Self { port: None }
    }

    pub fn open(device: &str, baudrate: i32) -> io::Result<Self> {
        crate::pinmux::configure_for_device(device);
        Ok(Self {
            port: Some(SerialPort::open(device, baudrate, false)?),
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

    pub fn grab(&mut self) {
        self.set_angle(0, SERVO0_GRAB, 1000);
        self.set_angle(1, SERVO1_GRAB, 1000);
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
        sleep_ms(1500);

        self.set_angle(2, ID2_ANGLE_CLOSE, 1000);
        sleep_ms(1000);

        self.set_angle(0, SERVO0_LIFT, 1000);
        self.set_angle(1, SERVO1_LIFT, 1000);
        sleep_ms(1200);
    }

    pub fn release_pos(&mut self) {
        self.set_angle(0, SERVO0_READY, 1000);
        self.set_angle(1, SERVO1_READY, 1000);
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
    }

    pub fn release(&mut self) {
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
    }

    pub fn grab_pos(&mut self) {
        self.set_angle(0, SERVO0_READY, 1000);
        self.set_angle(1, SERVO1_READY, 1000);
        self.set_angle(2, ID2_ANGLE_OPEN, 1000);
    }

    pub fn show(&mut self) {
        self.set_angle(0, SERVO0_LIFT, 1000);
        self.set_angle(1, SERVO1_LIFT, 1000);
        self.set_angle(2, ID2_ANGLE_CLOSE, 1000);
    }

    fn send_command(&mut self, command: &str) {
        let Some(port) = &mut self.port else {
            return; // mock/test:无硬件,静默丢弃舵机指令
        };
        if let Err(err) = port.write_all_drain(command.as_bytes()) {
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
