//! 运行时控制参数:底盘运动、抓取时序、机械臂姿势、电机映射。
//!
//! 这些参数过去写死在 `robot.rs` / `arm.rs` 的 `const` 上,改一次就得重新编译。
//! 现在把它们外置到一份 TOML 配置(板端默认 `/root/akars.toml`),启动时读一次并
//! 装进全局 [`OnceLock`];调参只改配置文件、不用重编译。
//!
//! 设计要点:
//! - 默认值直接取自 `robot.rs` / `arm.rs` / `bucket.rs` 里原来的 `const`(单一真值来
//!   源),所以文件缺失、解析失败或字段缺省时逐项回退,行为与旧版本完全一致。
//! - 目标桶色只暴露颜色名(`[bucket] bucket_color = "blue"`);色相窗口、饱和度/亮度
//!   下限这些检测内情留在 `bucket.rs` 的预设表里,不摆到配置文件上。
//! - 控制函数通过 [`get`] 读取,不改签名;[`get`] 在未 [`init`] 时返回默认值,
//!   因此单元测试无需初始化即可确定性地跑到默认参数上。

use crate::arm::{
    ANGLE_MAX, GRAB_CLOSE_TOLERANCE_DEG, GRAB_VERIFY, ID2_ANGLE_CLOSE, ID2_ANGLE_OPEN, PULSE_MAX,
    PULSE_MIN, SERVO0_GRAB, SERVO0_LIFT, SERVO0_READY, SERVO1_GRAB, SERVO1_LIFT, SERVO1_READY,
};
use crate::bucket::{
    BUCKET_APPROACH_SPD, BUCKET_AREA_BRAKE, BUCKET_AREA_DEPOSIT, BUCKET_BRAKE_MS, BUCKET_BRAKE_SPD,
    BUCKET_CENTER_DEAD_ZONE, BUCKET_CONFIRM_CNT, BUCKET_DEPOSIT_BACKOFF_MS, BUCKET_DEPOSIT_MS,
    BUCKET_HOME_MS, BUCKET_K_TURN, BUCKET_LOST_FRAMES, BUCKET_MAX_BIAS, BUCKET_MIN_AREA_FULLRES,
    BUCKET_SEARCH_SPD,
};
use crate::motor::MotorConfig;
use crate::robot::{
    AREA_FILTER_ALPHA, BACKWARD_SPEED, BRAKE_TRAVEL_MARGIN, CENTER_MARGIN, CHASE_FAR_AREA,
    CHASE_MID_AREA, CHASE_SPEED_FAR, CHASE_SPEED_MID, CHASE_SPEED_NEAR, CORRECT_SETTLE_MS,
    GRAB_AREA, GRAB_AREA_MAX, GRAB_BACKOFF_MS, GRAB_CLOSE_MS, GRAB_CONFIRM_MS, GRAB_HOME_MS,
    GRAB_HYSTERESIS_FRAC, GRAB_RELEASE_MS, GRAB_TURN_COUNT, GRAB_TURN_GAP_MS, GRAB_TURN_MS,
    GRAB_TURN_SPEED, IDLE_SPEED, LANE_BREAK_PX, LOST_GRACE_FRAMES, MARGIN_PROXIMITY_GAIN,
    PROXIMITY_DAMP_MIN, REFERENCE_FRAME_WIDTH, TURN_INPLACE_AREA, TURN_PULSE_GAIN_MS,
    TURN_PULSE_MAX_MS, TURN_PULSE_MIN_MS, TURN_PULSE_SPEED, TURN_PULSE_SPEED_DIFF,
};
use serde::Deserialize;
use std::path::Path;
use std::sync::OnceLock;

/// 底盘运动参数:直线锁定、修正脉冲、接近速度分级、搜索/后退等。
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default)]
pub struct ChassisConfig {
    /// 参考帧宽度,像素类阈值以此为基准按分辨率缩放。
    pub reference_frame_width: f32,
    /// 基础中心死区(像素,基于 reference_frame_width)。
    pub center_margin: i32,
    /// 中心死区随接近放大的系数。
    pub margin_proximity_gain: f32,
    /// 车道滞环的跑偏判定阈值(像素)。
    pub lane_break_px: i32,
    /// 原地转修正脉冲的转向幅值(角速度)。
    pub turn_pulse_speed: i32,
    /// 差速段(远处)修正脉冲的转向幅值,比原地转轻一档。
    pub turn_pulse_speed_diff: i32,
    /// 修正脉冲时长增益:时长 = 增益 × 归一化偏差 × 接近阻尼。
    pub turn_pulse_gain_ms: f32,
    /// 修正脉冲时长下限(克服静摩擦的最短时长)。
    pub turn_pulse_min_ms: u64,
    /// 修正脉冲时长上限。
    pub turn_pulse_max_ms: u64,
    /// 脉冲后的静置时长(相位裕度,等转动惯性停稳)。
    pub correct_settle_ms: u64,
    /// 预测刹车的收油余量(以“一帧行程”为单位)。
    pub brake_travel_margin: f32,
    /// 接近阻尼下限:球越近脉冲压得越短,此处为下限。
    pub proximity_damp_min: f32,
    /// 原地转的面积阈值:area_ratio ≥ 此值时丢掉前进分量。
    pub turn_inplace_area: f32,
    /// 远/中分级的面积阈值:小于此值用 FAR 档。
    pub chase_far_area: f32,
    /// 中/近分级的面积阈值:小于此值用 MID 档。
    pub chase_mid_area: f32,
    /// 远处接近速度。
    pub chase_speed_far: i32,
    /// 中处接近速度。
    pub chase_speed_mid: i32,
    /// 近处接近速度(实测能推得动整车的最低档以上)。
    pub chase_speed_near: i32,
    /// 丢球后原地慢转搜索的速度。
    pub idle_speed: i32,
    /// 太近时的后退速度。
    pub backward_speed: i32,
    /// 全局转向极性:轮子通道与物理左右反接的底盘置 true。
    ///
    /// 一个开关同时翻转所有闭环差的左右方向:追球修正、丢球搜索、追桶趋近、
    /// 找桶旋转(接线反接补偿),让同一种补偿一致生效;正常接线保持 false。
    /// 抓取盲转方向由 `[grab] grab_turn_direction` 单独指定,不随本开关走。
    pub turn_reverse: bool,
}

impl Default for ChassisConfig {
    fn default() -> Self {
        Self {
            reference_frame_width: REFERENCE_FRAME_WIDTH,
            center_margin: CENTER_MARGIN,
            margin_proximity_gain: MARGIN_PROXIMITY_GAIN,
            lane_break_px: LANE_BREAK_PX,
            turn_pulse_speed: TURN_PULSE_SPEED,
            turn_pulse_speed_diff: TURN_PULSE_SPEED_DIFF,
            turn_pulse_gain_ms: TURN_PULSE_GAIN_MS,
            turn_pulse_min_ms: TURN_PULSE_MIN_MS,
            turn_pulse_max_ms: TURN_PULSE_MAX_MS,
            correct_settle_ms: CORRECT_SETTLE_MS,
            brake_travel_margin: BRAKE_TRAVEL_MARGIN,
            proximity_damp_min: PROXIMITY_DAMP_MIN,
            turn_inplace_area: TURN_INPLACE_AREA,
            chase_far_area: CHASE_FAR_AREA,
            chase_mid_area: CHASE_MID_AREA,
            chase_speed_far: CHASE_SPEED_FAR,
            chase_speed_mid: CHASE_SPEED_MID,
            chase_speed_near: CHASE_SPEED_NEAR,
            idle_speed: IDLE_SPEED,
            backward_speed: BACKWARD_SPEED,
            turn_reverse: false,
        }
    }
}

/// 抓取盲转阶段的转向。机械臂/爪子相对相机的偏心方向随装配而变,不同小车需要
/// 不同的方向,故做成显式的 left/right,而不是跟着底盘接线极性走。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GrabTurnDirection {
    /// 盲转阶段左转(左轮回退、右轮前进)。
    #[default]
    Left,
    /// 盲转阶段右转(左轮前进、右轮回退)。
    Right,
}

/// 抓取状态机各段的阈值与时长。
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default)]
pub struct GrabConfig {
    /// “够近可抓”的面积阈值。
    pub grab_area: f32,
    /// “太近了、先后退”的面积阈值,须大于 grab_area;相等时每次抓取起手都会
    /// 先退再复判(退到带内才盲转),实际抓取距离由复判而不是盲退时长决定。
    pub grab_area_max: f32,
    /// 抓取前盲转微调的次数(补偿爪子偏心)。
    pub grab_turn_count: i32,
    /// 抓取前盲转微调的速度。
    pub grab_turn_speed: i32,
    /// 抓取前盲转微调的转向:left 左转、right 右转,按爪子相对相机的偏心方向选。
    pub grab_turn_direction: GrabTurnDirection,
    /// 面积比 EMA 系数(0~1):越小越平滑,单帧框抖动不直接碰阈值线。
    pub area_filter_alpha: f32,
    /// “够近”触发线的滞回带宽(占 grab_area 的比例):带内抖动不清驻留计时。
    pub grab_hysteresis_frac: f32,
    /// “够近且居中”须连续保持的驻留时长(毫秒)。用时间不用帧数,
    /// 否则确认窗随帧率跑(旧 grab_confirm_threshold 帧数口径已废弃)。
    pub grab_confirm_ms: u64,
    /// 丢失宽限帧数,超过才判定真丢并开始搜索。
    pub lost_grace_frames: i32,
    /// 太近时后退的时长。
    pub grab_backoff_ms: u64,
    /// 每次盲转微调的时长。
    pub grab_turn_ms: u64,
    /// 两次盲转之间的停稳间隔。
    pub grab_turn_gap_ms: u64,
    /// 爪子闭合后等待伺服到位的时长。
    pub grab_close_ms: u64,
    /// 松开后等待伺服到位的时长。
    pub grab_release_ms: u64,
    /// 爪子归位后回到追球前的等待时长。
    pub grab_home_ms: u64,
}

impl Default for GrabConfig {
    fn default() -> Self {
        Self {
            grab_area: GRAB_AREA,
            grab_area_max: GRAB_AREA_MAX,
            grab_turn_count: GRAB_TURN_COUNT,
            grab_turn_speed: GRAB_TURN_SPEED,
            grab_turn_direction: GrabTurnDirection::default(),
            area_filter_alpha: AREA_FILTER_ALPHA,
            grab_hysteresis_frac: GRAB_HYSTERESIS_FRAC,
            grab_confirm_ms: GRAB_CONFIRM_MS,
            lost_grace_frames: LOST_GRACE_FRAMES,
            grab_backoff_ms: GRAB_BACKOFF_MS,
            grab_turn_ms: GRAB_TURN_MS,
            grab_turn_gap_ms: GRAB_TURN_GAP_MS,
            grab_close_ms: GRAB_CLOSE_MS,
            grab_release_ms: GRAB_RELEASE_MS,
            grab_home_ms: GRAB_HOME_MS,
        }
    }
}

/// 目标桶色。配置文件里只写颜色名(`bucket_color = "blue"`)。
///
/// 色相窗口、饱和度/亮度下限是实现细节,放在 `bucket.rs` 的预设表里,由本枚举去查表 ——
/// 让使用者去调 H/S/V 只会调坏:窗口得盖住真实的深蓝/浅蓝,还被网球的黄绿挡着。
///
/// 只提供与网球(H 约 55~70)分得开的颜色:黄色正好压在球身上,选了会把手里那颗球当
/// 成桶,所以刻意不提供。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BucketColor {
    /// 蓝色(默认)。
    #[default]
    Blue,
    /// 红色。
    Red,
    /// 绿色。
    Green,
}

/// 追桶/放桶参数:是否追桶、目标颜色、两级趋近速度、中心死区、找桶与放球时序。
///
/// 闭环比追球简单得多(桶是静止大目标),所以这里的值大多沿用参考实现;换颜色、
/// 换桶尺寸都是改这里,不用重编译。
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default)]
pub struct BucketConfig {
    /// 抓到球后是否进入“追桶 → 放球”循环;false 退回旧行为(抓完原地松爪再追球)。
    pub chase_bucket: bool,
    /// 目标桶色:只写颜色名,色相窗口那种内情不进配置文件。
    pub bucket_color: BucketColor,
    /// 连通域最小面积(全分辨率像素数,色度网格上按采样率折算)。
    pub min_area_fullres: i32,
    /// 连续多少帧看到桶才确认(防抖)。
    pub confirm_frames: i32,
    /// 趋近中连续多少帧看不到桶 → 退回搜索。
    pub lost_frames: i32,
    /// 找不到桶时原地旋转搜索的速度。
    pub search_speed: i32,
    /// 趋近速度(远方一档)。
    pub approach_speed: i32,
    /// 制动区(面积 ≥ area_brake)的蠕行速度。须高于“下单但车不动”的下限,且不高于
    /// approach_speed;加载时会被钳到合法区间(见 bucket.rs 的 `approach_speeds`)。
    pub brake_speed: i32,
    /// 进入制动区的面积比(降到蠕行速度,避免撞桶)。
    pub area_brake: f32,
    /// 够近可放球的面积比。
    pub area_deposit: f32,
    /// 比例转向增益:bias = k_turn × 偏差 / 半宽。
    pub k_turn: f32,
    /// 转向 bias 的上限(单轮加减的最大幅值)。
    pub max_bias: i32,
    /// 中心死区(全分辨率像素),偏差在此内不转。
    pub center_dead_zone: i32,
    /// 放球前的硬刹时长(ms)。
    pub brake_ms: u64,
    /// 松爪后等舵机走完的时长(ms)。
    pub deposit_ms: u64,
    /// 机械臂归位后、回追球前的等待(ms)。
    pub home_ms: u64,
    /// 放完球后的倒退时长(ms):车头几乎贴着桶,不退就原地旋转会蹭到桶壁。
    /// 只求退开、不求停准,所以按帧推进即可(不是抓球那个跑满墙钟的后退脉冲)。
    /// 0 = 不退。速度沿用 `[chassis] backward_speed`。
    pub deposit_backoff_ms: u64,
}

impl Default for BucketConfig {
    fn default() -> Self {
        Self {
            chase_bucket: true,
            bucket_color: BucketColor::default(),
            min_area_fullres: BUCKET_MIN_AREA_FULLRES,
            confirm_frames: BUCKET_CONFIRM_CNT,
            lost_frames: BUCKET_LOST_FRAMES,
            search_speed: BUCKET_SEARCH_SPD,
            approach_speed: BUCKET_APPROACH_SPD,
            brake_speed: BUCKET_BRAKE_SPD,
            area_brake: BUCKET_AREA_BRAKE,
            area_deposit: BUCKET_AREA_DEPOSIT,
            k_turn: BUCKET_K_TURN,
            max_bias: BUCKET_MAX_BIAS,
            center_dead_zone: BUCKET_CENTER_DEAD_ZONE,
            brake_ms: BUCKET_BRAKE_MS,
            deposit_ms: BUCKET_DEPOSIT_MS,
            home_ms: BUCKET_HOME_MS,
            deposit_backoff_ms: BUCKET_DEPOSIT_BACKOFF_MS,
        }
    }
}

/// 机械臂舵机姿势、脉标定与抓取动作时序。
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default)]
pub struct ArmConfig {
    /// 爪子张开角度(舵机2)。
    pub id2_angle_open: f32,
    /// 爪子闭合角度(舵机2)。
    pub id2_angle_close: f32,
    /// 舵机全量程角度,用于角度→脉宽映射。
    pub angle_max: f32,
    /// 脉宽下限(对应 0°)。
    pub pulse_min: i32,
    /// 脉宽上限(对应 angle_max)。
    pub pulse_max: i32,
    /// 舵机0 待抓/归位角度。
    pub servo0_ready: f32,
    /// 舵机1 待抓/归位角度。
    pub servo1_ready: f32,
    /// 舵机0 伸到最低(抓取)角度。
    pub servo0_grab: f32,
    /// 舵机1 伸到最低(抓取)角度。
    pub servo1_grab: f32,
    /// 舵机0 抬起(持球)角度。
    pub servo0_lift: f32,
    /// 舵机1 抬起(持球)角度。
    pub servo1_lift: f32,
    /// 抓取完成后是否读回夹爪闭合角度,据此判断有没有真的夹到球。
    pub grab_verify: bool,
    /// 空合判定容差(度):读回角度在闭合角以上此值以内仍算"空合"。
    pub grab_close_tolerance_deg: f32,
    /// 每条 set_angle 指令的运动时长(time_ms)。
    pub grab_move_time_ms: i32,
    /// grab() 第 1 步(伸到最低、张爪)后的等待。
    pub grab_settle_pre_close_ms: u64,
    /// grab() 第 2 步(闭爪)后的等待。
    pub grab_settle_close_ms: u64,
    /// grab() 第 3 步(抬起)后的等待。
    pub grab_settle_lift_ms: u64,
}

impl Default for ArmConfig {
    fn default() -> Self {
        Self {
            id2_angle_open: ID2_ANGLE_OPEN,
            id2_angle_close: ID2_ANGLE_CLOSE,
            angle_max: ANGLE_MAX,
            pulse_min: PULSE_MIN,
            pulse_max: PULSE_MAX,
            servo0_ready: SERVO0_READY,
            servo1_ready: SERVO1_READY,
            servo0_grab: SERVO0_GRAB,
            servo1_grab: SERVO1_GRAB,
            servo0_lift: SERVO0_LIFT,
            servo1_lift: SERVO1_LIFT,
            grab_verify: GRAB_VERIFY,
            grab_close_tolerance_deg: GRAB_CLOSE_TOLERANCE_DEG,
            grab_move_time_ms: 1000,
            grab_settle_pre_close_ms: 1500,
            grab_settle_close_ms: 1000,
            grab_settle_lift_ms: 1200,
        }
    }
}

/// 电机映射参数(`MotorConfig` 的可外置子集,`device` 仍走命令行)。
#[derive(Clone, Copy, Debug, PartialEq, Deserialize)]
#[serde(default)]
pub struct MotorFileConfig {
    /// PWM 满量程缩放(百分比)。
    pub speed_scale: i32,
    /// 死区映射的速度下限。
    pub min_speed: i32,
    /// 编码器每转脉冲数。
    pub ppr: u16,
    /// PWM 频率。
    pub pwm_freq: u16,
}

impl Default for MotorFileConfig {
    fn default() -> Self {
        let motor = MotorConfig::default();
        Self {
            speed_scale: motor.speed_scale,
            min_speed: motor.min_speed,
            ppr: motor.ppr,
            pwm_freq: motor.pwm_freq,
        }
    }
}

/// 顶层配置:TOML 的五个 section。整份及每个 section 都允许缺省,缺省即回退到
/// 内置默认值。
#[derive(Clone, Copy, Debug, Default, PartialEq, Deserialize)]
#[serde(default)]
pub struct Config {
    pub chassis: ChassisConfig,
    pub grab: GrabConfig,
    pub arm: ArmConfig,
    pub bucket: BucketConfig,
    pub motor: MotorFileConfig,
}

static CONFIG: OnceLock<Config> = OnceLock::new();

/// 板端默认配置路径,与二进制 `/root/akars` 同目录。
pub const DEFAULT_CONFIG_PATH: &str = "/root/akars.toml";

/// 把加载好的配置装进全局,供各控制函数读取。仅第一次调用生效。
pub fn init(cfg: Config) {
    let _ = CONFIG.set(cfg);
}

/// 读取当前配置;未 [`init`] 时返回默认值(单元测试走这条路径)。
pub fn get() -> Config {
    CONFIG.get().copied().unwrap_or_default()
}

/// 从 `path` 加载 TOML 配置。
///
/// 文件不存在、读取失败或解析失败都不 panic,而是打印提示并回退到 [`Config::default`],
/// 保证程序总能起来。解析成功时,缺省字段由 `#[serde(default)]` 逐项填回默认值。
pub fn load(path: &Path) -> Config {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(err) => {
            if err.kind() == std::io::ErrorKind::NotFound {
                eprintln!(
                    "[config] no config file at {}, using built-in defaults",
                    path.display()
                );
            } else {
                eprintln!(
                    "[config] failed to read {}: {err}; using built-in defaults",
                    path.display()
                );
            }
            return Config::default();
        }
    };

    match toml::from_str::<Config>(&text) {
        Ok(cfg) => {
            eprintln!("[config] loaded {}", path.display());
            cfg
        }
        Err(err) => {
            eprintln!(
                "[config] failed to parse {}: {err}; using built-in defaults",
                path.display()
            );
            Config::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{BucketColor, BucketConfig, Config, GrabTurnDirection, get};

    /// 缺省字段回退到内置默认,给出的字段生效。
    #[test]
    fn partial_override_keeps_defaults() {
        let toml = r#"
[chassis]
turn_pulse_speed = 33
"#;
        // 只给 chassis.turn_pulse_speed,其它 chassis 字段以及 grab/arm/motor 整段都缺省。
        let cfg: Config = toml::from_str(toml).unwrap();
        assert_eq!(cfg.chassis.turn_pulse_speed, 33);
        assert_eq!(
            cfg.chassis.turn_pulse_speed_diff,
            Config::default().chassis.turn_pulse_speed_diff
        );
        assert_eq!(cfg.grab, Config::default().grab);
        assert_eq!(cfg.arm, Config::default().arm);
    }

    /// 未 init 时 get() 走默认值,保证控制函数的单测确定性。
    #[test]
    fn get_returns_default_before_init() {
        assert_eq!(get(), Config::default());
    }

    /// `[bucket]` 段:给出的字段生效,其余逐项回退默认 —— 追桶参数改配置文件即可。
    #[test]
    fn bucket_section_overrides_defaults() {
        let cfg: Config = toml::from_str(
            "[bucket]\nbucket_color = \"red\"\napproach_speed = 30\narea_deposit = 0.85\n",
        )
        .unwrap();
        assert_eq!(cfg.bucket.bucket_color, BucketColor::Red);
        assert_eq!(cfg.bucket.approach_speed, 30);
        assert_eq!(cfg.bucket.area_deposit, 0.85);
        assert_eq!(
            cfg.bucket.search_speed,
            BucketConfig::default().search_speed
        );
        assert_eq!(cfg.bucket.home_ms, BucketConfig::default().home_ms);
    }

    /// 目标色只认颜色名,缺省是蓝。
    #[test]
    fn bucket_colour_parses_and_defaults() {
        assert_eq!(Config::default().bucket.bucket_color, BucketColor::Blue);
        let cfg: Config = toml::from_str("[bucket]\nbucket_color = \"green\"\n").unwrap();
        assert_eq!(cfg.bucket.bucket_color, BucketColor::Green);
    }

    /// 拼错的颜色名会让整份文件解析失败(加载器打印警告后全走默认),而不是静默
    /// 挑一个别的窗口 —— 挑错了车会对着一面墙追很久。
    #[test]
    fn unknown_bucket_colour_is_rejected() {
        assert!(toml::from_str::<Config>("[bucket]\nbucket_color = \"purple\"\n").is_err());
    }

    /// 仓库里的 `akars.example.toml` 必须能整份解析:serde 会静默忽略拼错的键名,
    /// 只有真解析一次才能发现(例如把 min_area_fullres 打成 trl_min_area_fullres)。
    #[test]
    fn example_toml_parses_and_keeps_bucket_values() {
        let cfg: Config = toml::from_str(include_str!("../akars.example.toml"))
            .expect("akars.example.toml 解析失败");
        // 抽几个键确认确实落进了结构体(改示例里的标定值时同步更新这里)。
        assert_eq!(cfg.bucket.bucket_color, BucketColor::Blue);
        assert_eq!(cfg.bucket.min_area_fullres, 1000);
        assert_eq!(cfg.bucket.k_turn, 20.0);
        // 只写部分键的文件仍应保留其它 section 的默认值。
        assert!(cfg.bucket.chase_bucket);
    }

    /// chase_bucket 在 `[bucket]` 段,缺省为开,TOML 显式写 false 才关掉追桶循环。
    #[test]
    fn chase_bucket_parses_and_defaults() {
        assert!(Config::default().bucket.chase_bucket);
        let cfg: Config = toml::from_str("[bucket]\nchase_bucket = false\n").unwrap();
        assert!(!cfg.bucket.chase_bucket);
    }

    /// grab_turn_direction 接受 "left"/"right",缺省回退到 Left。
    #[test]
    fn grab_turn_direction_parses_and_defaults() {
        let cfg: Config = toml::from_str("[grab]\ngrab_turn_direction = \"right\"\n").unwrap();
        assert_eq!(cfg.grab.grab_turn_direction, GrabTurnDirection::Right);
        assert_eq!(
            Config::default().grab.grab_turn_direction,
            GrabTurnDirection::Left
        );
    }
}
