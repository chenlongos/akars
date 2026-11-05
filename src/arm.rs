use crate::config;
use crate::serial::SerialPort;
use std::io;
use std::thread;
use std::time::{Duration, Instant};

// 舵机角度实测标定。整条抓取序列就四组姿势,对照着改:
//
//   姿势              舵机0   舵机1   舵机2(爪)
//   待抓/归位          READY   READY   OPEN       grab_pos()
//   伸到最低           GRAB    GRAB    OPEN       grab() 第 1 步
//   夹住(仍在最低位)   GRAB    GRAB    CLOSE      grab() 第 2 步
//   抬起/持球找桶      LIFT    LIFT    CLOSE      grab() 第 3 步,一直保持到桶口才 release()
//
// 注意"抬"几乎全是舵机1 干的(READY 185 → LIFT),舵机0 只是从 GRAB 回到 LIFT 位。
pub(crate) const ID2_ANGLE_OPEN: f32 = 140.0;
pub(crate) const ID2_ANGLE_CLOSE: f32 = 90.0;
pub(crate) const ANGLE_MAX: f32 = 270.0;
pub(crate) const PULSE_MIN: i32 = 500;
pub(crate) const PULSE_MAX: i32 = 2500;

pub(crate) const SERVO0_READY: f32 = 150.0;
pub(crate) const SERVO1_READY: f32 = 185.0;
pub(crate) const SERVO0_GRAB: f32 = 245.0;
pub(crate) const SERVO1_GRAB: f32 = 185.0;
// LIFT 当前与 READY 同值 —— 是有意的,不是复制粘贴漏改:持球姿势就是归位姿势。
pub(crate) const SERVO0_LIFT: f32 = 150.0;
pub(crate) const SERVO1_LIFT: f32 = 185.0;

// 夹爪(舵机2)闭合到位后是否读回角度、据此判断有没有夹到球。
// 夹到球时爪口被球顶住,舵机到不了闭合目标角,读回角度停在闭合角之上;
// 空合则一路走到闭合角。容差是"闭合线以上多少度仍算空合",逐车标定。
pub(crate) const GRAB_VERIFY: bool = true;
pub(crate) const GRAB_CLOSE_TOLERANCE_DEG: f32 = 15.0;

// 夹爪舵机编号(0/1 是手臂两轴,2 是爪)。
const CLAW_SERVO_ID: i32 = 2;
// 读回位置的应答超时(ms)。ZP10S 的 PRAD 是同步应答,正常 <50ms。
const READ_POSITION_TIMEOUT_MS: u64 = 200;

/// 抓取一格之后的判定。
///
/// 只看抬臂结束时夹爪的实测角度:夹着球时爪口被球顶住,舵机没法走到闭合目标角,
/// 读回角度明显大于闭合角;空合(含抬臂途中掉球,闭合指令会把爪彻底夹闭)则已是
/// 闭合角。机器人据此决定是继续"追桶→放球",还是当作没抓到、回追球重来。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GrabVerdict {
    /// 夹住了:读回角度没走到闭合线,被球挡住。
    Held { pulse: i32 },
    /// 空合:读回角度已闭合到位,球不在爪上。
    Missed { pulse: i32 },
    /// 没读到回包(固件不响应 PRAD / 串口超时 / 关闭了校验):无法判定,
    /// 按"夹住"处理,与旧版本行为一致。
    Unknown,
}

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
        let clamped = angle.clamp(0.0, config::get().arm.angle_max);
        let pulse = angle_to_pulse(clamped);
        self.send_command(&format!("#{servo_id:03}P{pulse:04}T{time_ms}!"));
    }

    pub fn release_torque(&mut self, servo_id: i32) {
        self.send_command(&format!("#{servo_id:03}PULK"));
    }

    pub fn restore_torque(&mut self, servo_id: i32) {
        self.send_command(&format!("#{servo_id:03}PULR"));
    }

    /// 跑完整条抓取序列,并返回夹爪闭合后的判定(见 [`GrabVerdict`])。
    pub fn grab(&mut self) -> GrabVerdict {
        let a = config::get().arm;
        self.set_angle(0, a.servo0_grab, a.grab_move_time_ms);
        self.set_angle(1, a.servo1_grab, a.grab_move_time_ms);
        self.set_angle(2, a.id2_angle_open, a.grab_move_time_ms);
        sleep_ms(a.grab_settle_pre_close_ms);

        self.set_angle(2, a.id2_angle_close, a.grab_move_time_ms);
        sleep_ms(a.grab_settle_close_ms);

        self.set_angle(0, a.servo0_lift, a.grab_move_time_ms);
        self.set_angle(1, a.servo1_lift, a.grab_move_time_ms);
        sleep_ms(a.grab_settle_lift_ms);

        // 判定放在抬臂之后:夹爪闭合那一刻夹住不等于抬起来还在 —— 球在抬臂
        // 过程中滑落/被抖掉时,爪口会继续被夹到闭合角。闭合指令一直生效,
        // 掉球后舵机就会把爪彻底闭合,所以抬完再读才是"最终"的夹持状态。
        self.claw_verdict()
    }

    /// 读回夹爪闭合角度并判定是否夹到球。读不到位置时退化到 `Unknown`。
    fn claw_verdict(&mut self) -> GrabVerdict {
        let a = config::get().arm;
        let verdict = claw_verdict_from_pulse(self.read_pulse(CLAW_SERVO_ID), &a);
        match verdict {
            GrabVerdict::Held { pulse } => eprintln!(
                "[arm] claw blocked at {:.0}° (pulse {pulse}, close target {:.0}°) -> ball held",
                pulse_to_angle(pulse, &a),
                a.id2_angle_close
            ),
            GrabVerdict::Missed { pulse } => eprintln!(
                "[arm] claw closed to {:.0}° (pulse {pulse}, close target {:.0}°) -> empty",
                pulse_to_angle(pulse, &a),
                a.id2_angle_close
            ),
            GrabVerdict::Unknown => {
                eprintln!("[arm] claw position unreadable -> assuming a ball is held")
            }
        }
        verdict
    }

    /// 读回舵机当前脉宽。ZP10S 手册 p.26:`#002PRAD!` → `#002P1500!`。
    ///
    /// 无回包、回包不是位置、或 `[arm] grab_verify = false` 时返回 `None`。
    pub fn read_pulse(&mut self, servo_id: i32) -> Option<i32> {
        if !config::get().arm.grab_verify {
            return None;
        }
        // mock:无硬件,没有位置可读
        let port = self.port.as_mut()?;
        // 先丢掉残留回包,否则会把上一次的位置当成这一次的。
        port.flush();
        let command = format!("#{servo_id:03}PRAD!");
        if let Err(err) = port.write_all_drain(command.as_bytes()) {
            eprintln!("[arm] read position failed: {err}");
            return None;
        }

        let deadline = Instant::now() + Duration::from_millis(READ_POSITION_TIMEOUT_MS);
        let mut reply = String::new();
        while reply.len() < 31 {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            match port.read_byte(remaining) {
                Ok(Some(byte)) => {
                    reply.push(byte as char);
                    if byte == b'!' {
                        break; // 帧尾
                    }
                }
                Ok(None) => break,
                Err(err) => {
                    eprintln!("[arm] read position failed: {err}");
                    break;
                }
            }
        }
        parse_position_reply(&reply, servo_id)
    }

    pub fn release_pos(&mut self) {
        let a = config::get().arm;
        self.set_angle(0, a.servo0_ready, a.grab_move_time_ms);
        self.set_angle(1, a.servo1_ready, a.grab_move_time_ms);
        self.set_angle(2, a.id2_angle_open, a.grab_move_time_ms);
    }

    pub fn release(&mut self) {
        let a = config::get().arm;
        self.set_angle(2, a.id2_angle_open, a.grab_move_time_ms);
    }

    pub fn grab_pos(&mut self) {
        let a = config::get().arm;
        self.set_angle(0, a.servo0_ready, a.grab_move_time_ms);
        self.set_angle(1, a.servo1_ready, a.grab_move_time_ms);
        self.set_angle(2, a.id2_angle_open, a.grab_move_time_ms);
    }

    pub fn show(&mut self) {
        let a = config::get().arm;
        self.set_angle(0, a.servo0_lift, a.grab_move_time_ms);
        self.set_angle(1, a.servo1_lift, a.grab_move_time_ms);
        self.set_angle(2, a.id2_angle_close, a.grab_move_time_ms);
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

/// 解析位置回包,形如 `#002P1500!` → 1500。
///
/// 只认本舵机 ID,并且 `P` 后面必须落在脉宽量程内 —— `#002P!`(ID 检测)、
/// `#002PMOD1!`(模式)这类回包同样是 `#`+`P`,直接 atoi 会读成 0。
fn parse_position_reply(reply: &str, servo_id: i32) -> Option<i32> {
    let rest = reply.trim().strip_prefix('#')?;
    if rest.get(0..3)?.parse::<i32>().ok()? != servo_id {
        return None;
    }
    let tail = rest.get(3..)?.strip_prefix('P')?;
    let digits: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
    let pulse = digits.parse::<i32>().ok()?;
    let a = config::get().arm;
    (a.pulse_min..=a.pulse_max)
        .contains(&pulse)
        .then_some(pulse)
}

/// 脉宽 → 角度(`angle_to_pulse` 的逆映射)。
fn pulse_to_angle(pulse: i32, cfg: &config::ArmConfig) -> f32 {
    let span = (cfg.pulse_max - cfg.pulse_min).max(1) as f32;
    ((pulse - cfg.pulse_min) as f32 / span) * cfg.angle_max
}

/// 由读回脉宽判定夹爪是否夹到球。读不到(`None`)时无法判定 → `Unknown`。
fn claw_verdict_from_pulse(read: Option<i32>, cfg: &config::ArmConfig) -> GrabVerdict {
    let Some(pulse) = read else {
        return GrabVerdict::Unknown;
    };
    let angle = pulse_to_angle(pulse, cfg);
    if angle <= cfg.id2_angle_close + cfg.grab_close_tolerance_deg {
        GrabVerdict::Missed { pulse }
    } else {
        GrabVerdict::Held { pulse }
    }
}

fn angle_to_pulse(angle: f32) -> i32 {
    let a = config::get().arm;
    let span = (a.pulse_max - a.pulse_min) as f32;
    ((a.pulse_min as f32 + (angle / a.angle_max) * span).round() as i32)
        .clamp(a.pulse_min, a.pulse_max)
}

fn sleep_ms(ms: u64) {
    thread::sleep(Duration::from_millis(ms));
}

#[cfg(test)]
mod tests {
    use super::{
        angle_to_pulse, claw_verdict_from_pulse, parse_position_reply, pulse_to_angle, GrabVerdict,
    };
    use crate::config;

    #[test]
    fn converts_angles_to_pulses() {
        assert_eq!(angle_to_pulse(0.0), 500);
        assert_eq!(angle_to_pulse(270.0), 2500);
        assert_eq!(angle_to_pulse(135.0), 1500);
    }

    #[test]
    fn pulse_to_angle_inverts_the_mapping() {
        let cfg = config::Config::default().arm;
        for angle in [0.0, 40.0, 90.0, 140.0, 270.0] {
            let back = pulse_to_angle(angle_to_pulse(angle), &cfg);
            assert!((back - angle).abs() < 1.0, "{angle}° 往返成 {back}°");
        }
    }

    /// 只认位置回包:`#002P1500!` 正常解析,ID 不符/非位置/越界一律拒绝,
    /// 否则 `#002P!` 会读成 0 而把"没夹到"误判成"读到了"。
    #[test]
    fn parses_only_real_position_replies() {
        assert_eq!(parse_position_reply("#002P1500!", 2), Some(1500));
        assert_eq!(parse_position_reply("#002P800!\r\n", 2), Some(800));
        assert_eq!(parse_position_reply("#002P!", 2), None);
        assert_eq!(parse_position_reply("#002PMOD1!", 2), None);
        assert_eq!(
            parse_position_reply("#001P1500!", 2),
            None,
            "别的舵机的回包"
        );
        assert_eq!(parse_position_reply("#002P9999!", 2), None, "超出脉宽量程");
        assert_eq!(parse_position_reply("", 2), None);
        assert_eq!(parse_position_reply("garbage", 2), None);
    }

    /// 判定只看闭合线:走到闭合角(含容差)以内 = 空合;被球顶在闭合线之上 = 夹住。
    #[test]
    fn claw_verdict_reads_the_closed_angle() {
        let mut cfg = config::Config::default().arm;
        cfg.id2_angle_close = 40.0;
        cfg.grab_close_tolerance_deg = 15.0;

        let close_pulse = angle_to_pulse(cfg.id2_angle_close);
        assert_eq!(
            claw_verdict_from_pulse(Some(close_pulse), &cfg),
            GrabVerdict::Missed { pulse: close_pulse }
        );
        // 容差以内(45° < 40+15)仍算空合:伺服回差/标定误差不该算成夹住。
        let slack_pulse = angle_to_pulse(cfg.id2_angle_close + cfg.grab_close_tolerance_deg - 2.0);
        assert!(matches!(
            claw_verdict_from_pulse(Some(slack_pulse), &cfg),
            GrabVerdict::Missed { .. }
        ));
        // 被球顶住:离闭合角还差一截(90°),就是夹到了。
        let blocked = angle_to_pulse(90.0);
        assert_eq!(
            claw_verdict_from_pulse(Some(blocked), &cfg),
            GrabVerdict::Held { pulse: blocked }
        );
        // 读不到位置:无法判定,交给调用方按"夹住"保持旧行为。
        assert_eq!(claw_verdict_from_pulse(None, &cfg), GrabVerdict::Unknown);
    }
}
