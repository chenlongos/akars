use crate::serial::SerialPort;
use std::io;
use std::thread;
use std::time::Duration;

const FRAME_SOF0: u8 = 0xAA;
const FRAME_SOF1: u8 = 0x55;

const CMD_INIT: u8 = 0x01;
const CMD_CONFIG: u8 = 0x02;
const CMD_SET_SPEED: u8 = 0x10;
const CMD_STOP: u8 = 0x11;
const CMD_BRAKE: u8 = 0x12;
const CMD_RESET: u8 = 0xFF;

const RSP_ACK: u8 = 0x80;

#[derive(Debug, Clone)]
pub struct MotorConfig {
    pub device: String,
    pub speed_scale: i32,
    pub ppr: u16,
    pub pwm_freq: u16,
    pub min_speed: i32,
}

impl Default for MotorConfig {
    fn default() -> Self {
        Self {
            device: "/dev/ttyS1".to_string(),
            speed_scale: 85,
            ppr: 4680,
            pwm_freq: 20000,
            min_speed: 15,
        }
    }
}

pub struct Motor {
    port: Option<SerialPort>,
    speed_scale: i32,
    min_speed: i32,
}

impl Motor {
    pub fn open(config: &MotorConfig) -> io::Result<Self> {
        // The UART pads need muxing before the port will physically transmit;
        // the device tree enables UART1 but leaves its pins unrouted.
        crate::pinmux::configure_for_device(&config.device);
        let port = SerialPort::open(&config.device, 115200, true)?;
        eprintln!("[dbg][motor] port opened, sleeping 100ms");
        thread::sleep(Duration::from_millis(100));
        port.flush();
        eprintln!("[dbg][motor] flushed, sending CMD_INIT");

        let mut motor = Self {
            port: Some(port),
            speed_scale: config.speed_scale,
            min_speed: config.min_speed,
        };
        motor.cmd_init();
        eprintln!("[dbg][motor] CMD_INIT done, sending CMD_CONFIG");
        motor.cmd_config(config.ppr, config.pwm_freq);
        eprintln!("[dbg][motor] CMD_CONFIG done");
        Ok(motor)
    }

    pub fn forward(&mut self, speed: i32) {
        self.drive(speed, speed);
    }

    pub fn backward(&mut self, speed: i32) {
        self.drive(-speed, -speed);
    }

    pub fn left(&mut self, speed: i32) {
        self.drive(-speed, speed);
    }

    pub fn right(&mut self, speed: i32) {
        self.drive(speed, -speed);
    }

    pub fn brake(&mut self) {
        self.cmd_brake(0);
        self.cmd_brake(1);
    }

    pub fn standby(&mut self) {
        self.cmd_stop(0);
        self.cmd_stop(1);
    }

    pub fn drive(&mut self, left_speed: i32, right_speed: i32) {
        let left = to_pwm(map_deadzone(left_speed, self.min_speed), self.speed_scale);
        let right = to_pwm(map_deadzone(right_speed, self.min_speed), self.speed_scale);
        self.cmd_set_speeds(left, right);
    }

    fn cmd_init(&mut self) {
        self.send_frame(CMD_INIT, &[]);
        match self.recv_frame(Duration::from_millis(500)) {
            Ok(Some((RSP_ACK, _))) => eprintln!("[motor] CMD_INIT: got ACK, ESP32 is responding"),
            Ok(Some((cmd, _))) => {
                eprintln!("[motor] CMD_INIT: unexpected reply 0x{cmd:02X} (not ACK)")
            }
            Ok(None) => eprintln!(
                "[motor] CMD_INIT: NO REPLY (timeout) — ESP32 not talking on this port/wiring"
            ),
            Err(err) => eprintln!("[motor] CMD_INIT: serial read error: {err}"),
        }
    }

    fn cmd_config(&mut self, ppr: u16, pwm_freq: u16) {
        let mut payload = [0u8; 4];
        put_be16(&mut payload[0..2], ppr);
        put_be16(&mut payload[2..4], pwm_freq);
        self.send_frame(CMD_CONFIG, &payload);
        let _ = self.recv_frame(Duration::from_millis(500));
    }

    fn cmd_set_speeds(&mut self, left_pwm: i16, right_pwm: i16) {
        self.cmd_set_speed(0, left_pwm);
        self.cmd_set_speed(1, right_pwm);
    }

    fn cmd_set_speed(&mut self, motor_id: u8, pwm: i16) {
        let raw = pwm as u16;
        let payload = [motor_id, (raw >> 8) as u8, raw as u8];
        self.send_frame(CMD_SET_SPEED, &payload);
    }

    fn cmd_stop(&mut self, motor_id: u8) {
        self.send_frame(CMD_STOP, &[motor_id]);
    }

    fn cmd_brake(&mut self, motor_id: u8) {
        self.send_frame(CMD_BRAKE, &[motor_id]);
    }

    fn send_frame(&mut self, cmd: u8, payload: &[u8]) {
        if payload.len() > u8::MAX as usize {
            return;
        }
        let len = payload.len() as u8;
        let mut frame = Vec::with_capacity(payload.len() + 5);
        frame.push(FRAME_SOF0);
        frame.push(FRAME_SOF1);
        frame.push(cmd);
        frame.push(len);
        frame.extend_from_slice(payload);
        frame.push(checksum(cmd, len, payload));

        if let Some(port) = &mut self.port {
            if let Err(err) = port.write_all_drain(&frame) {
                eprintln!("[motor] write failed: {err}");
            }
        }
    }

    fn recv_frame(&mut self, timeout: Duration) -> io::Result<Option<(u8, Vec<u8>)>> {
        let Some(port) = &mut self.port else {
            return Ok(None);
        };

        let deadline = std::time::Instant::now() + timeout;
        loop {
            let remaining = deadline.saturating_duration_since(std::time::Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            if port.read_byte(remaining)? != Some(FRAME_SOF0) {
                continue;
            }
            if port.read_byte(Duration::from_millis(50))? != Some(FRAME_SOF1) {
                continue;
            }
            let Some(cmd) = port.read_byte(Duration::from_millis(50))? else {
                return Ok(None);
            };
            let Some(len) = port.read_byte(Duration::from_millis(50))? else {
                return Ok(None);
            };
            let mut payload = vec![0u8; len as usize];
            for byte in &mut payload {
                let Some(value) = port.read_byte(Duration::from_millis(50))? else {
                    return Ok(None);
                };
                *byte = value;
            }
            let Some(chk) = port.read_byte(Duration::from_millis(50))? else {
                return Ok(None);
            };
            if checksum(cmd, len, &payload) == chk {
                return Ok(Some((cmd, payload)));
            }
        }
    }
}

impl Drop for Motor {
    fn drop(&mut self) {
        self.send_frame(CMD_RESET, &[]);
    }
}

fn checksum(cmd: u8, len: u8, payload: &[u8]) -> u8 {
    payload.iter().fold(cmd ^ len, |acc, b| acc ^ *b)
}

fn put_be16(dst: &mut [u8], value: u16) {
    dst[0] = (value >> 8) as u8;
    dst[1] = value as u8;
}

fn map_deadzone(value: i32, min_speed: i32) -> i32 {
    if value == 0 {
        return 0;
    }
    let sign = value.signum();
    let mag = value.abs().min(100);
    let mapped = min_speed + (mag - 1) * (100 - min_speed) / 99;
    sign * mapped.min(100)
}

fn to_pwm(speed: i32, scale: i32) -> i16 {
    let clamped = speed.clamp(-100, 100);
    (clamped * scale / 100) as i16
}

#[cfg(test)]
mod tests {
    use super::{checksum, map_deadzone, to_pwm};

    #[test]
    fn checksum_matches_protocol() {
        assert_eq!(checksum(0x02, 4, &[0x12, 0x48, 0x4E, 0x20]), 0x32);
    }

    #[test]
    fn maps_deadzone_like_cpp_driver() {
        assert_eq!(map_deadzone(0, 15), 0);
        assert_eq!(map_deadzone(1, 15), 15);
        assert_eq!(map_deadzone(-1, 15), -15);
        assert_eq!(map_deadzone(100, 15), 100);
    }

    #[test]
    fn maps_speed_to_pwm() {
        assert_eq!(to_pwm(100, 150), 150);
        assert_eq!(to_pwm(-100, 150), -150);
        assert_eq!(to_pwm(50, 150), 75);
    }
}
