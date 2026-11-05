use crate::arm::{Arm, GrabVerdict};
use crate::bucket::{self, BucketOutcome};
use crate::camera::UsbCamera;
use crate::config;
use crate::detector::Detection;
use crate::image_bridge;
use crate::motor::Motor;
use crate::tpu::{InferTiming, InferenceConfig, ModelInputFormat, YoloModel};
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

static STOP_REQUESTED: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Debug)]
pub struct RobotConfig {
    pub inference: InferenceConfig,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self {
            inference: InferenceConfig::default(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RobotStatus {
    ChaseTennis,
    GrabTennis,
}

/// 抓取序列的各阶段。每段只在进入时发一次底盘/机械臂指令,之后按时间戳推进,
/// 主循环在整段抓取期间照常取帧、推理、打日志(不像脉冲版那样整段 sleep 掉)。
#[derive(Clone, Copy, Debug)]
enum GrabPhase {
    Idle,
    /// 太近:先后退一点再抓,免得把球顶飞。脉冲按墙钟定时一次跑完
    /// (见 `backoff_pulse`),结束立即刹车。
    Backoff,
    /// 后退脉冲后的静止观测窗:车已停住,等 EMA 吃进几帧新读数再复判距离。
    /// 这一段不动,所以帧节奏不会变成位移。
    BackoffSettle {
        until: Instant,
    },
    /// 抓取前盲转微调(补偿爪子相对相机的偏心,方向由 grab_turn_direction 决定)。
    /// 整段在一次调用里按墙钟跑完,见 `blind_turn_sequence`。
    Turn,
    /// 爪子闭合,等伺服到位。
    Closing {
        until: Instant,
    },
    /// 松开,等伺服到位。
    Releasing {
        until: Instant,
    },
    /// 爪子归位,到位后回到追球。
    Homing {
        until: Instant,
    },
}

/// 转向阶段。直道是连续的(不给差速),修正是离散的(脉冲 + 静置)—— 见 servo_step 的注释。
#[derive(Clone, Copy, Debug)]
enum SteerPhase {
    /// 偏差在车道内:两轮同速走直线。
    Straight,
    /// 跑偏了,正在发修正脉冲;dir = +1 右转 / -1 左转。
    Correcting { until: Instant, dir: i32 },
    /// 脉冲后静置:只走直线,等转动惯性停稳再评估。
    Settling { until: Instant },
}

/// “够近且居中”的抓取门控:EMA 压单帧框抖动,滞回防阈值线来回穿越,
/// 驻留用时间而不是帧数,消除帧率依赖(同一配置应同一距离触发)。
#[derive(Clone, Copy, Debug, Default)]
struct GrabGate {
    /// 滤波后的面积比;`0.0` 表示尚未播下种子,第一次观测直接取当帧值。
    ema: f32,
    /// 滞回状态:`ema ≥ grab_area` 进,`< grab_area − band` 出。
    engaged: bool,
    /// 连续满足 `engaged && centered` 的起始时刻;条件断了就清。
    since: Option<Instant>,
}

impl GrabGate {
    /// 把一帧原始面积推进滤波。抓取状态机期间也调用:只观测、不改转向。
    fn observe(&mut self, raw_area: f32, cfg: &config::Config) {
        let alpha = cfg.grab.area_filter_alpha.clamp(0.01, 1.0);
        self.ema = if self.ema > 0.0 {
            alpha.mul_add(raw_area, (1.0 - alpha) * self.ema)
        } else {
            raw_area
        };
    }

    /// 基于已观测的 ema 做滞回 + 驻留判定,`true` 表示驻留到点可交接抓取。
    fn step(&mut self, centered: bool, cfg: &config::Config, now: Instant) -> bool {
        let band = cfg.grab.grab_area * cfg.grab.grab_hysteresis_frac.clamp(0.0, 0.9);
        if self.engaged {
            if self.ema < cfg.grab.grab_area - band {
                self.engaged = false;
            }
        } else if self.ema >= cfg.grab.grab_area {
            self.engaged = true;
        }

        if self.engaged && centered {
            let since = *self.since.get_or_insert(now);
            return now.duration_since(since) >= Duration::from_millis(cfg.grab.grab_confirm_ms);
        }
        self.since = None;
        false
    }

    /// 交接/取消/丢球后整体复位:下一颗球的接近要重新计时、重新滤波。
    fn reset(&mut self) {
        *self = Self::default();
    }
}

#[derive(Clone, Copy, Debug)]
struct RobotState {
    status: RobotStatus,
    area_ratio: f32,
    ball_cx: i32,
    // “够近且居中”门控(EMA+滞回+时间窗),取代旧的连续帧硬计数。
    gate: GrabGate,
    // 本次抓取已执行的 Backoff 重试次数,`start_grab` 清零。
    backoff_retries: u32,
    // 最近一帧球框底边是否已贴出画面(球出镜头)。裁切区里面积读数不可信,
    // Backoff 复判把它当独立的"仍然太近"信号用。
    ball_clipped: bool,
    // 连续漏检帧数;在宽限期内先保持不动等球重现,超过阈值才判定真丢并搜索。
    miss_count: i32,
    // 转向阶段:直线 / 修正脉冲 / 静置。
    steer_phase: SteerPhase,
    // 抓取状态机当前阶段。
    grab_phase: GrabPhase,
    // 「抓起来 → 追蓝桶 → 放进桶」流程的状态机。
    bucket: bucket::BucketTrack,
}

impl Default for RobotState {
    fn default() -> Self {
        Self {
            status: RobotStatus::ChaseTennis,
            area_ratio: 0.0,
            ball_cx: 0,
            gate: GrabGate::default(),
            backoff_retries: 0,
            ball_clipped: false,
            miss_count: 0,
            steer_phase: SteerPhase::Straight,
            grab_phase: GrabPhase::Idle,
            bucket: bucket::BucketTrack::default(),
        }
    }
}

// 参考帧宽度。像素类阈值(如中心死区)以此为基准按分辨率缩放,保证不同分辨率下行为一致。
pub(crate) const REFERENCE_FRAME_WIDTH: f32 = 640.0;
// 基础中心死区(像素,基于 640 宽)。球中心偏离画面中心在此范围内即视为“已居中”,不再转向,避免对着中心反复微调。
pub(crate) const CENTER_MARGIN: i32 = 50;
// 中心死区随接近放大的系数。有效死区 = CENTER_MARGIN × (1 + 此值 × area_ratio);
// 抓球前不再追求亚像素级居中(抓取自带左转补偿),近处放宽容差直接消除小幅 hunting。
pub(crate) const MARGIN_PROXIMITY_GAIN: f32 = 2.0;

// ── 直线锁定(lane keeping) ───────────────────────────────────────────────
// 三轮车(前二主动轮 + 后一从动轮),相机落在主动轮连线上 ⇒ 两轮同速下发就是严格直线
// (同样的 PWM 给到两侧,ESP32 各自的转速 PID 会把它保持住)。
//
// 转向只在**跑偏**时发生:偏差超过 BREAK 起一个修正脉冲,脉冲结束(或刹车/过头)后静置
// 一段时间,再回到直线段重新评估。直道阶段一个差速都不给,所以不会"扭着过去"。
// 三个抗摆头的机制缺一不可:
//   1. 车道滞环(只在真跑偏时动作,不给检测抖动任何反应);
//   2. 脉冲 + 静置(离散校正,不在运动中去测量);
//   3. 预测刹车(用上一帧的行程量提前收油)。
pub(crate) const LANE_BREAK_PX: i32 = 80;

// ── 修正脉冲(跑偏后的离散校正) ──────────────────────────────────────────
// **幅值恒定、时长正比于偏差** —— 对"最小动作就是一大格"的底盘来说,这是唯一能做出
// 细修正的办法:用时间平均,而不是用更小的幅值(后者会被死区映射吃掉/转不动)。
// 时长沿用原 TURN_PULSE_* 那套(clamp 后 55~250ms),它们本来就是调出来的。
//
// 幅值分两档 —— 两种修正的手感不是一回事,实测结论也不同:
//   - 原地转(近处,bias = 0):22 = 原代码 TURN_SPEED,实测能转,别再动;
//   - 差速段(远处,bias > 0):同一个 22 会甩出一个又大又快的弧(外轮 forward+22),
//     到头来"幅度过大",所以单独给小一档。调它**不会**影响原地转。
//
// 这两个是**角速度**(转多少),不是行进速度(跑多快)。想提速别加它们 ——
// 加了只会甩得更猛、过头更多,而车并不会更快到球跟前。
//
// ── 帧率补偿(22 → 20) ─────────────────────────────────────────────────
// 上面那套是按 25fps 标定的,实测主循环只有 18fps。帧间隔 40ms → 55.6ms,脉冲结束
// 之后到下一帧决策之间的"滑行"多转 39%,这正是过头的来源,所以原地转速要按比例下调。
//
// 两条算法给的是同一个数:
//   1. 帧率等比:  22 × 18/25 = 15.84
//   2. 总转角不变: R' = R × (T+40)/(T+55.6),T∈[55,250] → 18.9 ~ 20.9
// 两者下沿都跌破实测静摩擦下限 20(见上面 CHASE_SPEED 那段的下限约束),
// 所以托底取 20 —— 更低的转速会"看着在下令、车不动"。
pub(crate) const TURN_PULSE_SPEED: i32 = 20;
pub(crate) const TURN_PULSE_SPEED_DIFF: i32 = 10;
pub(crate) const TURN_PULSE_GAIN_MS: f32 = 250.0;
pub(crate) const TURN_PULSE_MIN_MS: u64 = 55;
pub(crate) const TURN_PULSE_MAX_MS: u64 = 250;
// 脉冲后的静置:这段时间只走直线、不评估不修正,等转动惯性停稳再采下一帧。
//
// 这不是"低帧率时代遗留的 sleep",**而是相位裕度**:20fps + 约一帧延迟下,如果边转边测,
// 读到的偏差永远是"上一帧的旧值 + 这一帧还在转"的叠加,于是必然过头、左右摆。
// 摆头是在把这段静置删掉、改成"每帧连续修"之后才出现的 —— 别再把这条去掉。
pub(crate) const CORRECT_SETTLE_MS: u64 = 100;
// 预测刹车的收油点:剩余偏差 ≥ 此值 × “一帧行程”时继续转,≤ 1 个行程时提前结束脉冲。
// 2.0 = 多留一个行程的余量,抵掉轮子减速本身也要时间这件事。
pub(crate) const BRAKE_TRAVEL_MARGIN: f32 = 2.0;
// 接近阻尼下限。原地旋转时越近的球在画面里划过越快;
// 脉冲时长 × clamp(1 - area_ratio, 此值, 1),球越大(越近)脉冲压得越短,避免近处过冲。
pub(crate) const PROXIMITY_DAMP_MIN: f32 = 0.35;

// 原地转的距离阈值。area_ratio ≥ 此值时,修正脉冲把前进分量丢掉,两轮等幅反向
// (±TURN_PULSE_SPEED)—— 只转不走,免得近距离"边转边顶",用前进量把偏差推成位移。
// 两轮的差仍是 2×TURN_PULSE_SPEED,所以角速度不变,少的只是前冲。
// 直线/静置段不走这里(照常按 approach_speed 前进),所以接近不会因此停住。
pub(crate) const TURN_INPLACE_AREA: f32 = 0.20;

// ── 接近速度分级(面积比越大越近,越慢) ──────────────────────────────────
// 静止球场景下最直接的提速手段:远处快跑,近了减速,免得冲到跟前才发现要抓。
// 下限约束:20 是实测能推得动整车的最低档,低于它会"看着在下令、车不动"
// (映射后 PWM 太小,过不了静摩擦)。所以 NEAR 只能往上加,不能往下减。
// 上限约束:25cm 小车从 50cm 外冲过来刹不住,分级的意义就是提前把速度收下来 ——
// 越靠下的一档越要保守,冲过头是直接顶飞球。
//
// 映射后的实际 PWM(speed → map_deadzone(min_speed=15) → ×speed_scale 85%):
//   40 → 40% / 32 → 34% / 26 → 30%      (满量程 85%)
// 改之前是 26/22/20 → 30%/28%/26%,三档全挤在满量程三分之一附近 —— 这就是车慢的根源,
// 提速的空间一直在,只是当初按"能推到球就行"调的。
pub(crate) const CHASE_FAR_AREA: f32 = 0.15;
pub(crate) const CHASE_MID_AREA: f32 = 0.30;
pub(crate) const CHASE_SPEED_FAR: i32 = 40;
pub(crate) const CHASE_SPEED_MID: i32 = 32;
pub(crate) const CHASE_SPEED_NEAR: i32 = 26;

// ── 抓取状态机各段时长(按帧推进,不再阻塞主循环) ────────────────────────
// 太近时后退的时长。
pub(crate) const GRAB_BACKOFF_MS: u64 = 200;
// 抓取前每次盲转微调的时长。
pub(crate) const GRAB_TURN_MS: u64 = 250;
// 两次盲转微调之间的停稳间隔,以及爪子各动作后的等待。
pub(crate) const GRAB_TURN_GAP_MS: u64 = 100;
pub(crate) const GRAB_CLOSE_MS: u64 = 200;
pub(crate) const GRAB_RELEASE_MS: u64 = 100;
pub(crate) const GRAB_HOME_MS: u64 = 100;

// 球框面积占画面比 ≥ 此值 → 认为“够近可抓”,是触发抓取的主要距离阈值。
pub(crate) const GRAB_AREA: f32 = 0.40;
// 面积比 ≥ 此值 → “太近了”,先后退一点再抓,避免撞飞球。必须 > GRAB_AREA。
pub(crate) const GRAB_AREA_MAX: f32 = 0.55;

// 抓取前小幅盲转微调的次数(补偿爪子相对相机的安装偏心,把球挪到爪子正前方)。
pub(crate) const GRAB_TURN_COUNT: i32 = 3;
// 确认驻留窗(时间):"够近且居中"须连续保持这么久才真正执行抓取。用时间不用
// 帧数:帧数窗在 11fps 与 18fps 下驻留差近一倍,抓取距离会跟着相机卡跑。
pub(crate) const GRAB_CONFIRM_MS: u64 = 300;
// 面积比 EMA 系数:越大越贴当帧,越小越平滑(滞后 ≈ 1/α 帧)。单帧框宽/高
// 被量化和回归噪声抖动 ±几 px,直接比阈值会让确认计时反复复位。
pub(crate) const AREA_FILTER_ALPHA: f32 = 0.4;
// 触发线的滞回带宽,按 GRAB_AREA 的比例计:ema ≥ 阈值进 engage,< 阈值×(1−此值)
// 才退出。带内抖动不清计时,防阈值线来回穿越造成的抖档。
pub(crate) const GRAB_HYSTERESIS_FRAC: f32 = 0.1;
// 抓取起手"太近→再退"的重试上限;封顶后仍判太近(或在裁切区)就放弃本次抓取
// 回追球重新逼近,不再硬着头皮盲转。
pub(crate) const GRAB_BACKOFF_RETRY_MAX: u32 = 2;
// 球底"出镜头"判定的贴边余量(px):bbox 底边距画面底边不足此值即视为已被裁。
// 进入裁切区后面积比不再满足 1/z²、读数失真,唯一安全动作是后退脱离。
pub(crate) const BOTTOM_CLIP_PX: f32 = 4.0;
// 球框长宽差判裁切的阈值(px):球是刚体球,正常成像 w≈h(实测全框时差 ≤8px);
// 底部被画面边缘削掉后框高变矮,w−h 拉开(实测 8.4cm 贴脸裁切时 395×339,差 56px)。
// 比贴边信号更早也更稳,两条信号取并。
pub(crate) const BOTTOM_CLIP_ASPECT_PX: f32 = 20.0;
// 盲转脉冲期间重发目标速度的间隔(ms)。ESP32 锁存目标速度,重发是失效保护;
// 但脉冲的结束点必须由墙钟决定,不能靠"下一帧再发停止指令"。
const BLIND_TURN_RESEND_MS: u64 = 50;
// 丢失宽限帧数。看到过球后若漏检,先原地保持不动等它重现,连续漏检超过此值才判定真丢并开始搜索,
// 避免转向时的偶发漏检(运动模糊/擦边)立刻触发盲搜把球搞丢。
pub(crate) const LOST_GRACE_FRAMES: i32 = 8;

// 没检测到球时原地慢转搜索的速度。转得快=扫过一圈快,但盲转过头会甩过球再转回来,
// 所以它是"够快能扫到"的档,不是越快越好。
//
// ── 帧率补偿(24 → 20) ─────────────────────────────────────────────────
// 搜索是**连续旋转**(不是脉冲),所以"转过头"的量 = 转速 × 从球入画面到控制生效的
// 那几个帧。帧率 25→18 后帧间隔 40ms→55.6ms,同样的转速下这段多转 39% —— 这就是
// "球在眼前还是被甩过去"的来源,所以转速要按比例下调。
//   帧率等比: 24 × 18/25 = 17.28
// 下沿仍受实测静摩擦下限 20 约束(见上面 CHASE_SPEED 那段),故托底取 20。
pub(crate) const IDLE_SPEED: i32 = 20;
// 抓取前盲转微调的速度。**别单独动它**:它是和 GRAB_TURN_MS 配成一对标定出来的
// ——"3 次 × 250ms"对应爪子偏心需要补的那个角度。只加速度不改时长,转过的角度按比例
// 变大,球就被顶到爪子旁边而不是正前方了。要提速得两个一起按同比例改。
pub(crate) const GRAB_TURN_SPEED: i32 = 18;
// 太近(≥ GRAB_AREA_MAX)时的后退速度。快一点只是退得更远些,退多了下一轮自己会再靠上来。
pub(crate) const BACKWARD_SPEED: i32 = 24;

pub fn install_signal_handlers() {
    unsafe {
        crate::linux::signal(crate::linux::SIGINT, signal_handler);
        crate::linux::signal(crate::linux::SIGTERM, signal_handler);
    }
}

extern "C" fn signal_handler(_: i32) {
    STOP_REQUESTED.store(true, Ordering::SeqCst);
}

pub fn stop_requested() -> bool {
    STOP_REQUESTED.load(Ordering::SeqCst)
}

pub fn run_tennis_hunter(
    mut camera: UsbCamera,
    mut model: YoloModel,
    mut motor: Motor,
    mut arm: Arm,
    config: RobotConfig,
) {
    let mut robot = RobotState::default();
    // 上一帧看到的居中偏差(像素)。预测刹车要用"两帧之间偏差缩了多少"当作一帧转动的
    // 行程量,所以它必须逐帧穿过主循环;上一帧没看到球时是 None,刹车自动让位给纯比例。
    let mut prev_offset: Option<i32> = None;
    let mut total_time = Duration::ZERO;
    let mut frame_count = 0u64;
    let mut frame_idx = 0u64;
    let (model_w, model_h) = model.input_dimensions();
    let input_format = model.input_format();
    let input_len = input_format.buffer_len(model_w, model_h).unwrap_or(0);
    if input_len == 0 {
        eprintln!("[camera] invalid model input size: {}x{}", model_w, model_h);
        motor.standby();
        return;
    }
    eprintln!("[tpu] model input format: {input_format:?}");
    let mut input_buffer = vec![0u8; input_len];

    arm.grab_pos();

    // 一直跑到收到 SIGINT/SIGTERM 为止:调用方传入的帧数上限不再生效。
    while !stop_requested() {
        frame_idx += 1;

        // ── 桶流程进行中:独占一帧跑桶色检测(不推理),由 robot.bucket.is_active() 触发。
        if robot.bucket.is_active() {
            let bucket_start = Instant::now();
            let frame = match camera.get_frame() {
                Ok(frame) => frame,
                Err(err) => {
                    eprintln!("[bucket] frame failed: {err}");
                    sleep_us(100_000);
                    continue;
                }
            };
            let capture_us = bucket_start.elapsed().as_micros() as i64;
            let control_start = Instant::now();
            let outcome = robot.bucket.step(
                bucket::FrameView {
                    pixels: &frame.pixels,
                    width: frame.width as i32,
                    height: frame.height as i32,
                    planar: crate::camera::PlanarFormat::I422,
                },
                &mut motor,
                &mut arm,
                Instant::now(),
            );
            let control_time = control_start.elapsed();
            eprintln!(
                "[time] frame={} capture={:.1} ms [bucket]",
                frame_idx,
                capture_us as f32 / 1000.0
            );
            let frame_time = Duration::from_micros(capture_us.max(0) as u64) + control_time;
            total_time += frame_time;
            frame_count += 1;
            let fps = if frame_time.as_secs_f32() > 0.0 {
                1.0 / frame_time.as_secs_f32()
            } else {
                0.0
            };
            let avg_fps = if total_time.as_secs_f32() > 0.0 {
                frame_count as f32 / total_time.as_secs_f32()
            } else {
                0.0
            };
            eprintln!(
                "[FPS] {fps:.2} avg: {avg_fps:.2} ({:.1}ms, control={:.1}ms)",
                frame_time.as_secs_f32() * 1000.0,
                control_time.as_secs_f32() * 1000.0
            );

            if outcome == BucketOutcome::Finished {
                prev_offset = None;
                robot.status = RobotStatus::ChaseTennis;
                robot.miss_count = 0;
                robot.gate.reset();
                robot.steer_phase = SteerPhase::Straight;
                robot.ball_cx = 0;
                robot.area_ratio = 0.0;
            }
            continue;
        }

        // ── 取帧:内核 ioctl 4 返回 YUV422 planar(I422),用户态转换后喂张量。
        let capture_start = Instant::now();
        let frame = match camera.get_frame() {
            Ok(frame) => frame,
            Err(err) => {
                eprintln!("[camera] failed to get frame: {err}");
                sleep_us(100_000);
                continue;
            }
        };
        let capture_us = capture_start.elapsed().as_micros() as i64;

        let preprocess_start = Instant::now();
        let preprocess_result = match input_format {
            ModelInputFormat::RgbPlanar => image_bridge::yuv422p_to_rgb_planar(
                &frame.pixels,
                &mut input_buffer,
                model_w,
                model_h,
            ),
            ModelInputFormat::Yuv420Planar => image_bridge::yuv422p_to_yuv420p_scaled(
                &frame.pixels,
                &mut input_buffer,
                frame.width as i32,
                frame.height as i32,
                model_w,
                model_h,
            ),
            ModelInputFormat::YuvNv12 => {
                image_bridge::yuv422p_to_nv12(&frame.pixels, &mut input_buffer, model_w, model_h)
            }
            ModelInputFormat::YuvNv21 => {
                image_bridge::yuv422p_to_nv21(&frame.pixels, &mut input_buffer, model_w, model_h)
            }
        };
        if let Err(err) = preprocess_result {
            eprintln!("[pre] YUV preprocess failed: {err}");
            continue;
        }
        let preprocess_us = preprocess_start.elapsed().as_micros() as i64;

        let image_w = frame.width as i32;
        let image_h = frame.height as i32;
        let mut timing = InferTiming::default();
        let inference_result = match input_format {
            ModelInputFormat::RgbPlanar => model.infer_rgb_planar_timed(
                &input_buffer,
                image_w,
                image_h,
                config.inference,
                Some(&mut timing),
            ),
            ModelInputFormat::Yuv420Planar
            | ModelInputFormat::YuvNv12
            | ModelInputFormat::YuvNv21 => model.infer_yuv420_timed(
                &input_buffer,
                image_w,
                image_h,
                config.inference,
                Some(&mut timing),
            ),
        };

        let detections = match inference_result {
            Ok(detections) => detections,
            Err(err) => {
                eprintln!("[detect] inference failed: {err}");
                continue;
            }
        };

        eprintln!(
            "[time] frame={} capture={:.1} pre={:.1} fwd={:.1} post={:.1} ms",
            frame_idx,
            capture_us as f32 / 1000.0,
            preprocess_us as f32 / 1000.0,
            timing.forward_us as f32 / 1000.0,
            timing.postprocess_us as f32 / 1000.0,
        );

        let control_start = Instant::now();
        prev_offset = handle_detections(
            &detections,
            image_w,
            image_h,
            prev_offset,
            &mut robot,
            &mut motor,
            &mut arm,
        );

        let control_time = control_start.elapsed();
        let frame_time = Duration::from_micros(
            capture_us.max(0) as u64
                + preprocess_us.max(0) as u64
                + timing.forward_us.max(0) as u64
                + timing.postprocess_us.max(0) as u64,
        ) + control_time;
        total_time += frame_time;
        frame_count += 1;
        let fps = if frame_time.as_secs_f32() > 0.0 {
            1.0 / frame_time.as_secs_f32()
        } else {
            0.0
        };
        let avg_fps = if total_time.as_secs_f32() > 0.0 {
            frame_count as f32 / total_time.as_secs_f32()
        } else {
            0.0
        };
        eprintln!(
            "[FPS] {fps:.2} avg: {avg_fps:.2} ({:.1}ms, control={:.1}ms)",
            frame_time.as_secs_f32() * 1000.0,
            control_time.as_secs_f32() * 1000.0
        );
    }

    motor.standby();
}

/// 取 bbox 面积最大的检测当目标球(NMS 后通常只有一个,留多候选容错)。
fn best_ball(detections: &[Detection]) -> Option<&Detection> {
    detections.iter().max_by(|a, b| {
        let area_a = a.bbox.w * a.bbox.h;
        let area_b = b.bbox.w * b.bbox.h;
        area_a.total_cmp(&area_b)
    })
}

/// 球框面积 / 画面面积。尺度无关的距离代理(真实距离 ∝ 1/√area)。
fn ball_area_ratio(best: &Detection, image_w: i32, image_h: i32) -> f32 {
    let image_area = (image_w.max(1) * image_h.max(1)) as f32;
    (best.bbox.w * best.bbox.h) / image_area
}

/// 球框底边是否已贴出画面底边(球出镜头)。触球近距离下框底碰到画面边缘,
/// 面积被裁剪后不再满足 1/z²,距离读数失真且只能单调偏大,是"贴脸"的硬信号。
/// 两路信号取并:①框底贴到画面底边余量线;②框被削矮(w−h 超过长宽差阈值,
/// 球是刚体球、正常成像 w≈h,只有底部出画才会拉开这个差)。
fn ball_bottom_clipped(best: &Detection, image_h: i32) -> bool {
    let y2 = best.bbox.y + best.bbox.h / 2.0;
    let touches_bottom = y2 >= image_h as f32 - BOTTOM_CLIP_PX;
    let aspect_squashed = best.bbox.w - best.bbox.h > BOTTOM_CLIP_ASPECT_PX;
    touches_bottom || aspect_squashed
}

/// 处理一帧检测,返回这一帧的居中偏差(没看到球时 None,留给下一帧算行程量)。
fn handle_detections(
    detections: &[Detection],
    image_w: i32,
    image_h: i32,
    prev_offset: Option<i32>,
    robot: &mut RobotState,
    motor: &mut Motor,
    arm: &mut Arm,
) -> Option<i32> {
    let cfg = config::get();

    // 抓取序列进行中:主循环照常取帧/推理/打日志,这里只按时间戳推进各阶段;
    // 但每帧仍观测面积 —— Backoff 到期的闭环复判要用(只测量,不转向)。
    if !matches!(robot.grab_phase, GrabPhase::Idle) {
        if let Some(b) = best_ball(detections) {
            robot.ball_clipped = ball_bottom_clipped(b, image_h);
            let raw = ball_area_ratio(b, image_w, image_h);
            robot.gate.observe(raw, &cfg);
            if servo_log() {
                eprintln!(
                    "[grab] observe phase={:?} raw={:.3} ema={:.3} clipped={}",
                    robot.grab_phase, raw, robot.gate.ema, robot.ball_clipped
                );
            }
        } else {
            robot.ball_clipped = false;
        }
        advance_grab(robot, motor, arm, bucket_enabled());
        return None;
    }

    let Some(best) = best_ball(detections) else {
        robot.gate.reset();
        robot.steer_phase = SteerPhase::Straight;
        robot.miss_count += 1;

        if robot.miss_count <= cfg.grab.lost_grace_frames {
            // 宽限期内:刚才还看得到球,先停住等它重现,别急着盲搜。
            eprintln!(
                "[detect] missed {}/{} frames, holding",
                robot.miss_count, cfg.grab.lost_grace_frames
            );
            motor.standby();
        } else {
            // 真丢了:朝球最后出现的一侧转,而不是固定方向盲搜。极性与追球
            // 共用 `[chassis] turn_reverse`,保证通道反接的底盘仍朝正确一侧搜索。
            robot.status = RobotStatus::ChaseTennis;
            let last_offset = robot.ball_cx - image_w / 2;
            let polarity = if cfg.chassis.turn_reverse { -1 } else { 1 };
            let s = cfg.chassis.idle_speed * polarity;
            if last_offset >= 0 {
                eprintln!("[detect] ball lost, searching right (last seen right)");
                motor.drive(s, -s);
            } else {
                eprintln!("[detect] ball lost, searching left (last seen left)");
                motor.drive(-s, s);
            }
        }
        return None;
    };
    robot.miss_count = 0;

    let area_ratio = ball_area_ratio(best, image_w, image_h);
    let ball_cx = best.bbox.x as i32;
    let center = image_w / 2;

    // 滤波先于居中判定:近距离放大用的裕度也吃 ema,两者同一时刻口径一致。
    robot.gate.observe(area_ratio, &cfg);
    let center_margin = proximity_center_margin(image_w, robot.gate.ema);
    let offset = ball_cx - center;
    let centered = offset.abs() <= center_margin;

    robot.area_ratio = area_ratio;
    robot.ball_cx = ball_cx;

    eprintln!(
        "[detect] area={area_ratio:.3} ema={:.3} cx={ball_cx} conf={:.3} engaged={} centered={centered}",
        robot.gate.ema, best.score, robot.gate.engaged
    );

    // 球底出镜头 = 面积读数失真的贴脸区,和"太近"同义。复用既有的分段
    // Backoff 脉冲后退,不另设连续倒车机制。只拦未 engage 的逼近:已 engage
    // 的车在逐帧刹车、即将交接抓取序列(其太近分支自有后退复判),在这里
    // 抢刀会把滑进裁切区的正常抓取砍成"追-退"死循环。
    if ball_bottom_clipped(best, image_h) && !robot.gate.engaged {
        eprintln!(
            "[grab] ball bottom clipped (ema={:.3}), backing off",
            robot.gate.ema
        );
        robot.status = RobotStatus::GrabTennis;
        robot.steer_phase = SteerPhase::Straight;
        motor.brake();
        begin_backoff(robot);
        return Some(offset);
    }

    // 确认驻留:时间窗到点才交接抓取状态机,单帧抖动由 EMA+滞回吸收,
    // 不再像旧硬计数那样被一帧清零。驻留期间主动刹车:CMD_STOP 会滑行,
    // 过冲随入停速度和地面摩擦漂,是"同阈值距离却飘"的主项之一。
    let fired = robot.gate.step(centered, &cfg, Instant::now());
    if robot.gate.engaged && centered {
        robot.status = RobotStatus::GrabTennis;
        robot.steer_phase = SteerPhase::Straight; // 已站定,滞环状态跟着复位
        motor.brake();
        if fired {
            start_grab(robot, robot.gate.ema);
        }
        return Some(offset);
    }

    robot.status = RobotStatus::ChaseTennis;
    // servo_step 吃的是 ema;engage 后直线段停前进(见 straight_forward)。
    servo_step(
        offset,
        prev_offset,
        image_w,
        robot.gate.ema,
        robot.gate.engaged,
        robot,
        motor,
    );
    Some(offset)
}

/// 每帧一次的转向决策,三个阶段:
/// - **Straight**:两轮同值 → 严格直线走,一个差速都不给;
/// - **Correcting**:跑偏了,发一个定幅值的修正脉冲,时长正比于偏差;
/// - **Settling**:脉冲后静置,只走直线,等转动惯性停稳再回去评估(相位裕度)。
fn servo_step(
    offset: i32,
    prev_offset: Option<i32>,
    image_w: i32,
    area_ratio: f32,
    engaged: bool,
    robot: &mut RobotState,
    motor: &mut Motor,
) {
    advance_steer_phase(
        robot,
        offset,
        prev_offset,
        image_w,
        area_ratio,
        Instant::now(),
    );

    let forward = straight_forward(area_ratio, engaged);
    // 脉冲的前进分量:远处边走边修,近了归零 → 两轮等幅反向,原地转。
    let bias = turn_forward_bias(forward, area_ratio);
    // 全局转向极性:`[chassis] turn_reverse` 置 true 时镜像左右,追球/搜索/追桶共用。
    let polarity = if config::get().chassis.turn_reverse {
        -1
    } else {
        1
    };
    let turn = match robot.steer_phase {
        SteerPhase::Correcting { dir, .. } => steer_turn(dir, turn_amplitude(area_ratio), polarity),
        SteerPhase::Straight | SteerPhase::Settling { .. } => 0,
    };

    if servo_log() {
        eprintln!(
            "[servo] offset={offset} phase={:?} turn={turn} fwd={forward} bias={bias} brake={:.2} L={} R={}",
            robot.steer_phase,
            brake_scale(offset, prev_offset),
            bias + turn,
            bias - turn
        );
    }

    if turn == 0 {
        // 非脉冲段不看 bias:直线段该走还得走,不然近处永远靠不上去。
        if forward == 0 {
            // engage 后目标已钉死在触球前:站等居中时用刹死而不是 standby,
            // CMD_STOP 滑行会把确认驻留期间剩下的距离全部滑进触球。
            if engaged {
                motor.brake();
            } else {
                motor.standby();
            }
        } else {
            // 同值下发 = 严格直线。
            motor.drive(forward, forward);
        }
        return;
    }
    // 远处保留前进分量:两轮同向滚动,不会因为"原地转那点力矩过不了静摩擦"而卡住;
    // 近处(bias = 0)等幅反向 = 原地转,转完由直线段负责往前靠。
    motor.drive(bias + turn, bias - turn);
}

/// 把名义转向 `dir`(±1)与全局极性 `polarity`(±1)合成轮用的转向量。
/// `polarity` 来自 `[chassis] turn_reverse`:通道与物理左右反接的底盘用它把
/// 追球 / 搜索两条闭回路一起翻回正确方向(见 servo_step / handle_detections)。
/// 抓取盲转不接本极性,方向由 `[grab] grab_turn_direction` 单独指定。
fn steer_turn(dir: i32, amplitude: i32, polarity: i32) -> i32 {
    dir * amplitude * polarity
}

/// 直线段的前进速度:正常追球按面积分档;抓取门控已 engage 时归零。
/// engage 意味着"距离已经够了",剩下只差居中 —— 那应该原地转去等,
/// 边等边蹭会把触发距离变成"等居中的时长"的函数,和 grab_area 脱钩。
fn straight_forward(area_ratio: f32, engaged: bool) -> i32 {
    if engaged {
        0
    } else {
        approach_speed(area_ratio)
    }
}

/// 修正脉冲里保留多少前进分量:远处给全量(边走边修的差速),近处归零(原地转)。
fn turn_forward_bias(forward: i32, area_ratio: f32) -> i32 {
    if area_ratio >= config::get().chassis.turn_inplace_area {
        0
    } else {
        forward
    }
}

/// 修正脉冲的转向幅值,与 `turn_forward_bias` 同一个分界:近处原地转用实测过的
/// `TURN_PULSE_SPEED`,远处差速段用小一档的 `TURN_PULSE_SPEED_DIFF`。
fn turn_amplitude(area_ratio: f32) -> i32 {
    let chassis = config::get().chassis;
    if area_ratio >= chassis.turn_inplace_area {
        chassis.turn_pulse_speed
    } else {
        chassis.turn_pulse_speed_diff
    }
}

/// 调参开关:AKARS_SERVO_LOG=1 时每帧打印转向决策(默认关,省串口带宽)。
fn servo_log() -> bool {
    std::env::var("AKARS_SERVO_LOG").is_ok()
}

/// 功能开关：「抓起来 → 追桶 → 放进桶」流程，由配置文件
/// `[bucket] chase_bucket` 决定（缺省开）；临时调试还可用 `AKARS_NO_BUCKET`
/// 环境变量强制关掉，不必改卡上的配置。
fn bucket_enabled() -> bool {
    config::get().bucket.chase_bucket && std::env::var("AKARS_NO_BUCKET").is_err()
}

/// 推进转向阶段。
fn advance_steer_phase(
    robot: &mut RobotState,
    offset: i32,
    prev_offset: Option<i32>,
    image_w: i32,
    area_ratio: f32,
    now: Instant,
) {
    match robot.steer_phase {
        SteerPhase::Straight => {
            if offset.abs() > scaled_px(config::get().chassis.lane_break_px, image_w) {
                let dir = if offset >= 0 { 1 } else { -1 };
                robot.steer_phase = SteerPhase::Correcting {
                    until: now + Duration::from_millis(pulse_ms(offset, image_w, area_ratio)),
                    dir,
                };
            }
        }
        SteerPhase::Correcting { until, .. } => {
            // 提前结束脉冲的两种情况:预测刹车说到位了,或者偏差已经翻号(转过头)。
            let braked = brake_scale(offset, prev_offset) <= 0.0;
            let flipped = matches!(
                prev_offset,
                Some(prev) if prev != 0 && offset != 0 && prev.signum() != offset.signum()
            );
            if braked || flipped || now >= until {
                robot.steer_phase = SteerPhase::Settling {
                    until: now + Duration::from_millis(config::get().chassis.correct_settle_ms),
                };
            }
        }
        SteerPhase::Settling { until } => {
            if now >= until {
                robot.steer_phase = SteerPhase::Straight;
            }
        }
    }
}

/// 修正脉冲时长:幅值恒定,时长正比于偏差(时间平均 = 细修正),夹在实测区间内。
fn pulse_ms(offset: i32, image_w: i32, area_ratio: f32) -> u64 {
    let chassis = config::get().chassis;
    let half_width = (image_w / 2).max(1) as f32;
    let norm = (offset.abs() as f32 / half_width).min(1.0);
    let damp = (1.0 - area_ratio.max(0.0)).clamp(chassis.proximity_damp_min, 1.0);
    (chassis.turn_pulse_gain_ms * norm * damp).round().clamp(
        chassis.turn_pulse_min_ms as f32,
        chassis.turn_pulse_max_ms as f32,
    ) as u64
}

/// 按参考帧宽等比缩放一个像素阈值。
fn scaled_px(px: i32, image_w: i32) -> i32 {
    ((px as f32) * image_w as f32 / config::get().chassis.reference_frame_width)
        .round()
        .max(1.0) as i32
}

/// 预测刹车系数:1.0 = 全速转,0.0 = 停止转向。
///
/// 两帧之间偏差缩小的量,就是"转一帧能划掉多少像素"的实测值(`travel`)——不用标定
/// 焦距或轮径。剩余偏差不足一帧行程时继续全速转,下一帧必然读到过冲,而修正永远迟到
/// 一帧,这就是远处的转过头。所以在 `travel × BRAKE_TRAVEL_MARGIN` 处开始收油、
/// 到 `travel` 处完全停转,让本体惯性把它停在中心附近。
///
/// 上一帧没看到球时没有可靠的行程量,返回 1.0(退回纯比例控制)。
fn brake_scale(offset: i32, prev_offset: Option<i32>) -> f32 {
    let Some(prev) = prev_offset else {
        return 1.0;
    };
    let travel = (prev.abs() - offset.abs()) as f32;
    if travel <= 0.0 {
        // 没在收敛(还没转起来,或被前进带来的方位变化抵消)→ 不刹车。
        return 1.0;
    }
    let margin = config::get().chassis.brake_travel_margin;
    let room = offset.abs() as f32;
    ((room - travel) / (travel * (margin - 1.0))).clamp(0.0, 1.0)
}

/// 接近速度分级:远处快跑,近了减速。进入抓取距离(≥ GRAB_AREA)后返回 0,
/// 那时只该对准、不该再往前顶。
fn approach_speed(area_ratio: f32) -> i32 {
    let cfg = config::get();
    if area_ratio >= cfg.grab.grab_area {
        0
    } else if area_ratio < cfg.chassis.chase_far_area {
        cfg.chassis.chase_speed_far
    } else if area_ratio < cfg.chassis.chase_mid_area {
        cfg.chassis.chase_speed_mid
    } else {
        cfg.chassis.chase_speed_near
    }
}

/// 后退脉冲结束后的静止观测窗(ms):车已刹停,等抓取序列期间持续观测的
/// EMA 吃进几帧新读数再复判。没有这段窗,复判用的还是后退前的贴脸读数,
/// 就会一段接一段无限退。
const BACKOFF_SETTLE_MS: u64 = 200;

/// 一段后退脉冲:墙钟定时跑满 `grab_backoff_ms`,结束立即刹车。
///
/// 旧实现由逐帧状态机推进(每帧发一次后退、下一帧才判断到期),而 ESP32 锁存
/// 目标速度 —— 帧一卡轮子就跟着多转一段,"偶尔大幅后退"就是这么来的。
fn backoff_pulse(motor: &mut Motor) {
    let speed = -config::get().chassis.backward_speed;
    drive_for(motor, speed, speed, config::get().grab.grab_backoff_ms);
    motor.brake();
}

/// 直接进入一段后退脉冲(球底出镜头时的贴脸脱离路径;内部重置门控,把
/// 这轮当全新的"太近"处理)。分段节奏与 start_grab 的太近分支一致。
fn begin_backoff(robot: &mut RobotState) {
    robot.gate.reset();
    robot.backoff_retries = 0;
    robot.steer_phase = SteerPhase::Straight;
    robot.grab_phase = GrabPhase::Backoff;
}

/// 确认够近且居中后进入抓取序列。太近的话先安排一段后退,否则直接开始左转微调。
fn start_grab(robot: &mut RobotState, area_ratio: f32) {
    let cfg = config::get();
    robot.gate.reset();
    robot.backoff_retries = 0;
    robot.steer_phase = SteerPhase::Straight;

    if area_ratio >= cfg.grab.grab_area_max {
        eprintln!("[grab] too close, backing up first");
        robot.grab_phase = GrabPhase::Backoff;
    } else {
        robot.grab_phase = GrabPhase::Turn;
    }
}

/// 抓取序列走完最后一格之后的去向。
///
/// `verdict` 是抬臂结束时读回的夹爪判定:`Missed`(空合)时球根本不在爪上,
/// 无论 `bucket_mode` 都不能往下走 —— 去追桶就是抱着空爪跑一圈,
/// 所以直接走开爪归位的收尾,回追球重新逼近。
///
/// `bucket_mode` 为真且夹到了球:球已在手上,不进放开的三段收尾,而是直接转入
/// “追蓝桶 → 放进桶”流程(`robot.bucket.start()`),由主循环的 `bucket.is_active()`
/// 分支接管。为假(旧行为):回初始位置 → 原地松爪 → 继续追球。
///
/// 单独抽出来是因为 `arm.grab()` 阻塞约 3.7s(`arm.rs:62-75`),想验证这一段就得绕开它。
fn finish_grab(robot: &mut RobotState, now: Instant, bucket_mode: bool, verdict: GrabVerdict) {
    if matches!(verdict, GrabVerdict::Missed { .. }) {
        // 空合:开爪归位后回追球,重新走一轮 追球 → 抓球,不进桶流程。
        eprintln!("[grab] empty claw -> reopen, resume chase instead of the next stage");
        robot.gate.reset();
        robot.steer_phase = SteerPhase::Straight;
        robot.grab_phase = GrabPhase::Closing {
            until: now + Duration::from_millis(config::get().grab.grab_close_ms),
        };
        return;
    }
    if bucket_mode {
        // 与桶流程唯一的分岔点:不松爪,夹着球去找桶。
        eprintln!("[bucket] ball held -> searching for bucket");
        robot.grab_phase = GrabPhase::Idle;
        robot.status = RobotStatus::ChaseTennis;
        robot.gate.reset();
        robot.miss_count = 0;
        robot.steer_phase = SteerPhase::Straight;
        robot.bucket.start();
        return;
    }
    robot.grab_phase = GrabPhase::Closing {
        until: now + Duration::from_millis(config::get().grab.grab_close_ms),
    };
}

/// Backoff 到期的复判结果。旧行为是"盲退固定时长就直接盲转",退没退到位
/// 全看电池/地面;现在用抓取序列期间持续观测的 ema 重新量一次再分支。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BackoffNext {
    /// 仍然太近(面积达上限或球底仍出镜头)且重试未封顶:再退一段。
    ReBackoff,
    /// 落在可抓带内:进盲转。
    Turn,
    /// 退过头/球丢了/封顶仍贴脸:取消本次抓取,回追球重新逼近。
    Abort,
}

fn backoff_next(area: f32, clipped: bool, cfg: &config::Config, retries: u32) -> BackoffNext {
    // 裁切区里面积被钉在饱和值附近不再变化,ema 可能远低于 grab_area_max;
    // 所以"球底出镜头"单独就是"仍然太近",与面积判定取并。
    let too_close = area >= cfg.grab.grab_area_max || clipped;
    if too_close && retries < GRAB_BACKOFF_RETRY_MAX {
        BackoffNext::ReBackoff
    } else if too_close {
        // 重试封顶仍贴脸:盲转硬抓比放弃危险得多,回追球重新平滑逼近。
        BackoffNext::Abort
    } else if area < cfg.grab.grab_area {
        BackoffNext::Abort
    } else {
        BackoffNext::Turn
    }
}

/// 推进抓取状态机:每段只在进入时发一次指令,到期后切换下一段。
/// 底盘指令按帧重发(ESP32 锁存目标转速,重发只是保险,不会叠加)。
///
/// `bucket_mode` 决定抓完这一格球后的去向:真 → 转入追桶/放桶;假 → 原地松爪回追球。
fn advance_grab(robot: &mut RobotState, motor: &mut Motor, arm: &mut Arm, bucket_mode: bool) {
    let now = Instant::now();
    let cfg = config::get();

    match robot.grab_phase {
        GrabPhase::Idle => {}

        GrabPhase::Backoff => {
            // 后退脉冲必须按墙钟跑完:位移是硬输出,不能由帧节奏决定。
            backoff_pulse(motor);
            robot.grab_phase = GrabPhase::BackoffSettle {
                until: Instant::now() + Duration::from_millis(BACKOFF_SETTLE_MS),
            };
        }

        GrabPhase::BackoffSettle { until } => {
            if now < until {
                return;
            }
            // 车已静止:用这几帧观测到的 ema 复判距离。旧版这里直接进盲转,
            // 盲退位移就是抓取距离的变量。
            motor.brake();
            match backoff_next(
                robot.gate.ema,
                robot.ball_clipped,
                &cfg,
                robot.backoff_retries,
            ) {
                BackoffNext::ReBackoff => {
                    robot.backoff_retries += 1;
                    eprintln!(
                        "[grab] still too close (ema={:.3} clipped={}), backing up again {}/{}",
                        robot.gate.ema,
                        robot.ball_clipped,
                        robot.backoff_retries,
                        GRAB_BACKOFF_RETRY_MAX
                    );
                    robot.grab_phase = GrabPhase::Backoff;
                }
                BackoffNext::Turn => robot.grab_phase = GrabPhase::Turn,
                BackoffNext::Abort => {
                    // 退过头、未再看到球、或封顶仍贴脸:别硬抓,回追球重新逼近。
                    eprintln!(
                        "[grab] backoff gave up (ema={:.3} clipped={}) -> re-approach",
                        robot.gate.ema, robot.ball_clipped
                    );
                    robot.gate.reset();
                    robot.grab_phase = GrabPhase::Idle;
                    robot.status = RobotStatus::ChaseTennis;
                    robot.steer_phase = SteerPhase::Straight;
                }
            }
        }

        GrabPhase::Turn => {
            // 盲转是开环的:转多少全看脉冲跑满多少毫秒,所以整段必须由墙钟定时。
            // 旧实现把"转 grab_turn_ms / 停 grab_turn_gap_ms"交给逐帧状态机推进,
            // 而帧周期(~60ms,相机卡顿时更久)直接量化了脉冲长度 —— 同一条命令
            // 有时只转半截、有时一股脑转过去,落爪位置跟着漂。
            blind_turn_sequence(motor);
            // arm.grab() 阻塞约 3.7s,结束时夹爪闭合、机械臂停在 LIFT 位。
            // 返回值是抬臂后读回的夹爪角度:夹到球才会走向桶/放球,空合(含抬臂
            // 途中掉球)则回追球重来。
            let verdict = arm.grab();
            // 上面两段都是阻塞的,`now` 已过期,收尾计时从当前重新起算。
            finish_grab(robot, Instant::now(), bucket_mode, verdict);
        }

        GrabPhase::Closing { until } => {
            if now < until {
                return;
            }
            // 先回初始位置。抓取结束时机械臂停在 LIFT 位,而 LIFT 与 READY 是同
            // 一组角度(舵机0/1 均为 150/185,见 arm.rs 的标定表),所以这一步不产生
            // 位移;grab_pos() 会把爪子一起置为 OPEN —— 球就是在这里放开的。
            arm.grab_pos();
            robot.grab_phase = GrabPhase::Releasing {
                until: now + Duration::from_millis(cfg.grab.grab_release_ms),
            };
        }

        GrabPhase::Releasing { until } => {
            if now < until {
                return;
            }
            // 爪子在上一步已经张开,这里是幂等重发,确保夹爪确实松开。
            arm.release();
            robot.grab_phase = GrabPhase::Homing {
                until: now + Duration::from_millis(cfg.grab.grab_home_ms),
            };
        }

        GrabPhase::Homing { until } => {
            if now < until {
                return;
            }
            eprintln!("[grab] ball released, resuming chase");
            robot.grab_phase = GrabPhase::Idle;
            robot.status = RobotStatus::ChaseTennis;
            robot.gate.reset();
            robot.miss_count = 0;
        }
    }
}

/// 抓取盲转阶段左右轮的目标转速:方向直接由 `[grab] grab_turn_direction` 给出,
/// `Left` 为左轮回退、右轮前进(前-后/后-前),`Right` 镜像成左轮前进、右轮回退。
/// 不叠加底盘接线极性(那是追球/搜索的事)。
fn grab_turn_wheels(speed: i32, direction: config::GrabTurnDirection) -> (i32, i32) {
    match direction {
        config::GrabTurnDirection::Right => (speed, -speed),
        config::GrabTurnDirection::Left => (-speed, speed),
    }
}

/// 盲转微调的一步:按当前方向转 `ms`,或者静置 `ms` 等惯性停稳。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TurnStep {
    Drive(u64),
    Settle(u64),
}

/// 盲转微调的时间表。纯数据,单元测试直接钉住段序与时长。
///
/// 每段都是固定时长,而不是"转到看不见球为止"之类的闭环:爪子与相机之间有
/// 装配常量偏心,补的就是这个常量。末尾多留一段静置,停稳后再下爪。
fn blind_turn_plan(cfg: &config::Config) -> Vec<TurnStep> {
    let turns = cfg.grab.grab_turn_count.max(1);
    let mut plan = Vec::with_capacity(turns as usize * 2);
    for i in 0..turns {
        if i > 0 {
            plan.push(TurnStep::Settle(cfg.grab.grab_turn_gap_ms));
        }
        plan.push(TurnStep::Drive(cfg.grab.grab_turn_ms));
    }
    plan.push(TurnStep::Settle(cfg.grab.grab_turn_gap_ms));
    plan
}

/// 按墙钟驱动一段,期间按固定节奏重发目标速度。结束点由 deadline 决定,
/// 而不是"下一次调用时再判断"——脉冲长度因此与主循环帧节奏无关。
fn drive_for(motor: &mut Motor, left: i32, right: i32, ms: u64) {
    let deadline = Instant::now() + Duration::from_millis(ms);
    loop {
        motor.drive(left, right);
        let now = Instant::now();
        if now >= deadline {
            return;
        }
        thread::sleep((deadline - now).min(Duration::from_millis(BLIND_TURN_RESEND_MS)));
    }
}

/// 执行整段盲转微调。每段结束主动 brake 而不是 standby 滑行:滑行会把上一段的
/// 残余角动量带进下一段,每段实际转量就随"上一段结束时轮子转多快"漂。
fn blind_turn_sequence(motor: &mut Motor) {
    let cfg = config::get();
    let (left, right) = grab_turn_wheels(cfg.grab.grab_turn_speed, cfg.grab.grab_turn_direction);
    for step in blind_turn_plan(&cfg) {
        match step {
            TurnStep::Drive(ms) => {
                drive_for(motor, left, right, ms);
                motor.brake();
            }
            TurnStep::Settle(ms) => thread::sleep(Duration::from_millis(ms)),
        }
    }
}

fn scaled_center_margin(image_w: i32) -> i32 {
    let chassis = config::get().chassis;
    ((chassis.center_margin as f32) * image_w as f32 / chassis.reference_frame_width)
        .round()
        .max(1.0) as i32
}

fn proximity_center_margin(image_w: i32, area_ratio: f32) -> i32 {
    let base = scaled_center_margin(image_w) as f32;
    let gain = config::get().chassis.margin_proximity_gain;
    (base * (1.0 + gain * area_ratio.max(0.0))).round().max(1.0) as i32
}

fn sleep_us(us: u64) {
    thread::sleep(Duration::from_micros(us));
}

#[cfg(test)]
mod tests {
    use super::{
        BackoffNext, CHASE_SPEED_FAR, CHASE_SPEED_MID, CHASE_SPEED_NEAR, CORRECT_SETTLE_MS,
        GRAB_AREA, GRAB_BACKOFF_RETRY_MAX, GrabGate, GrabPhase, LANE_BREAK_PX, RobotState,
        RobotStatus, SteerPhase, TURN_INPLACE_AREA, TURN_PULSE_GAIN_MS, TURN_PULSE_MAX_MS,
        TURN_PULSE_MIN_MS, TURN_PULSE_SPEED, TURN_PULSE_SPEED_DIFF, TurnStep, advance_grab,
        advance_steer_phase, approach_speed, backoff_next, ball_bottom_clipped, blind_turn_plan,
        blind_turn_sequence, brake_scale, finish_grab, grab_turn_wheels, proximity_center_margin,
        pulse_ms, scaled_center_margin, scaled_px, steer_turn, straight_forward, turn_amplitude,
        turn_forward_bias,
    };
    use crate::arm::{Arm, GrabVerdict};
    use crate::config::{self, GrabTurnDirection};
    use crate::detector::{Box2d, Detection};
    use crate::motor::Motor;
    use std::time::{Duration, Instant};

    const W: i32 = 640;

    /// 推进一帧转向阶段(无历史偏差)。
    fn step(robot: &mut RobotState, offset: i32, area_ratio: f32, now: Instant) {
        advance_steer_phase(robot, offset, None, W, area_ratio, now);
    }

    #[test]
    fn scales_center_margin() {
        assert_eq!(scaled_center_margin(640), 50);
        assert_eq!(scaled_center_margin(320), 25);
    }

    #[test]
    fn margin_widens_with_proximity() {
        // Far away (tiny ball) stays near the base margin.
        assert_eq!(proximity_center_margin(640, 0.0), 50);
        // Close up (big ball) widens the dead zone to stop hunting.
        assert!(proximity_center_margin(640, 0.4) > scaled_center_margin(640));
    }

    #[test]
    fn scales_pixel_thresholds_with_frame_width() {
        assert_eq!(scaled_px(LANE_BREAK_PX, 640), LANE_BREAK_PX);
        assert_eq!(scaled_px(LANE_BREAK_PX, 320), LANE_BREAK_PX / 2);
    }

    #[test]
    fn small_drift_never_starts_a_correction() {
        // 车道内的偏差(含检测抖动)一律不修正 —— 这是不"摇头"的第一道保证。
        let now = Instant::now();
        let mut robot = RobotState::default();
        step(&mut robot, 0, 0.05, now);
        step(&mut robot, LANE_BREAK_PX / 2, 0.05, now);
        step(&mut robot, -LANE_BREAK_PX, 0.05, now);
        assert!(matches!(robot.steer_phase, SteerPhase::Straight));
    }

    #[test]
    fn drifting_out_starts_a_pulse_towards_the_ball() {
        let now = Instant::now();
        let mut robot = RobotState::default();
        step(&mut robot, -LANE_BREAK_PX - 1, 0.05, now);
        match robot.steer_phase {
            SteerPhase::Correcting { dir, .. } => assert_eq!(dir, -1, "球在左 → 左转"),
            other => panic!("expected Correcting, got {other:?}"),
        }

        let mut robot = RobotState::default();
        step(&mut robot, LANE_BREAK_PX + 1, 0.05, now);
        match robot.steer_phase {
            SteerPhase::Correcting { dir, .. } => assert_eq!(dir, 1, "球在右 → 右转"),
            other => panic!("expected Correcting, got {other:?}"),
        }
    }

    #[test]
    fn pulse_ends_then_settles_then_straightens() {
        let now = Instant::now();
        let mut robot = RobotState::default();
        step(&mut robot, 300, 0.05, now);
        let SteerPhase::Correcting { until, .. } = robot.steer_phase else {
            panic!("expected Correcting");
        };

        // 脉冲未到点:继续修,不能中途改主意。
        step(&mut robot, 300, 0.05, now + Duration::from_millis(1));
        assert!(matches!(robot.steer_phase, SteerPhase::Correcting { .. }));

        // 到点 → 静置(这段时间只走直线,等惯性停稳)。
        step(&mut robot, 300, 0.05, until);
        let SteerPhase::Settling {
            until: settle_until,
        } = robot.steer_phase
        else {
            panic!("expected Settling, got {:?}", robot.steer_phase);
        };
        assert_eq!(
            settle_until - until,
            Duration::from_millis(CORRECT_SETTLE_MS)
        );

        // 静置中即使偏差还很大也不修 —— 这就是相位裕度,防止边转边测导致左右摆。
        step(&mut robot, 300, 0.05, until + Duration::from_millis(1));
        assert!(matches!(robot.steer_phase, SteerPhase::Settling { .. }));

        // 静置结束 → 回到直线段(下一帧再决定要不要补一枪)。
        step(&mut robot, 300, 0.05, settle_until);
        assert!(matches!(robot.steer_phase, SteerPhase::Straight));
    }

    #[test]
    fn pulse_ends_early_when_braked_or_overshot() {
        // 预测刹车说"下一帧就要过线了" → 立刻收,别转满。
        let now = Instant::now();
        let mut robot = RobotState::default();
        robot.steer_phase = SteerPhase::Correcting {
            until: now + Duration::from_millis(250),
            dir: 1,
        };
        // 上一帧偏差 300、这一帧 100(travel 200 ≥ room 100)→ 刹车到 0 → 结束脉冲。
        advance_steer_phase(&mut robot, 100, Some(300), W, 0.05, now);
        assert!(matches!(robot.steer_phase, SteerPhase::Settling { .. }));

        // 偏差翻号(转过头)→ 同样立刻收。
        let mut robot = RobotState::default();
        robot.steer_phase = SteerPhase::Correcting {
            until: now + Duration::from_millis(250),
            dir: 1,
        };
        advance_steer_phase(&mut robot, -40, Some(120), W, 0.05, now);
        assert!(matches!(robot.steer_phase, SteerPhase::Settling { .. }));
    }

    #[test]
    fn pulse_ms_is_proportional_and_clamped() {
        // 偏差越大脉冲越长。
        let small = pulse_ms(100, W, 0.0);
        let big = pulse_ms(300, W, 0.0);
        assert!(big > small);
        // 拉满半屏 = 增益值,且不超上限。
        assert_eq!(pulse_ms(1000, W, 0.0), TURN_PULSE_MAX_MS);
        assert!((pulse_ms(W / 2, W, 0.0) as f32 - TURN_PULSE_GAIN_MS).abs() < 1.0);
        // 下限:极小的偏差也不会短到"来不及克服静摩擦"。
        assert_eq!(pulse_ms(1, W, 0.0), TURN_PULSE_MIN_MS);
    }

    #[test]
    fn pulse_shrinks_near_the_ball() {
        // 同一个偏差,球越近(area 越大)脉冲越短 —— 近处画面的角速度更快,容易过冲。
        let far = pulse_ms(300, W, 0.0);
        let near = pulse_ms(300, W, 0.5);
        assert!(near < far);
    }

    #[test]
    fn brake_scale_eases_off_as_the_ball_arrives() {
        // travel = 100px/帧:还剩 200px(2 个行程)→ 继续转。
        assert_eq!(brake_scale(200, Some(300)), 1.0);
        // 只剩 100px(1 个行程)→ 收油,交给惯性。
        assert_eq!(brake_scale(100, Some(200)), 0.0);
        // 更少 → 仍是 0(不会反向)。
        assert_eq!(brake_scale(40, Some(200)), 0.0);
        // 两者之间线性过渡。
        let mid = brake_scale(150, Some(250)); // travel 100, room 150
        assert!(mid > 0.0 && mid < 1.0);
    }

    #[test]
    fn brake_scale_needs_two_consecutive_sightings() {
        // 上一帧没看到球 → 没有可靠行程量,不刹车。
        assert_eq!(brake_scale(100, None), 1.0);
        // 偏差在变大(还没转起来 / 被前进带来的方位变化抵消)→ 不刹车。
        assert_eq!(brake_scale(120, Some(60)), 1.0);
    }

    #[test]
    fn brake_engages_earlier_when_turning_faster() {
        // 同一个剩余偏差,慢转(行程小)不刹车、快转(行程大)已经该停了 —— 远处转得快,
        // 于是自动更早收油,不用为远近各调一套参数。
        let slow = brake_scale(100, Some(110)); // travel 10
        let fast = brake_scale(100, Some(200)); // travel 100
        assert_eq!(slow, 1.0);
        assert_eq!(fast, 0.0);
    }

    #[test]
    fn turn_pulse_keeps_forward_bias_far_away() {
        // 远处:边走边修,前进分量照给 —— 两轮同向,只是快慢不同。
        assert_eq!(turn_forward_bias(CHASE_SPEED_FAR, 0.10), CHASE_SPEED_FAR);
        assert_eq!(
            turn_forward_bias(CHASE_SPEED_MID, TURN_INPLACE_AREA - 0.01),
            CHASE_SPEED_MID
        );
    }

    #[test]
    fn turn_pulse_turns_in_place_from_the_threshold_on() {
        // 0.2 起(含 0.2~0.4 整段)丢掉前进分量 → 两轮等幅反向 = 原地转。
        for area in [TURN_INPLACE_AREA, 0.25, 0.30, 0.35, GRAB_AREA] {
            let bias = turn_forward_bias(approach_speed(area), area);
            assert_eq!(bias, 0, "area={area} 应该原地转");
            assert_eq!(
                (bias + TURN_PULSE_SPEED, bias - TURN_PULSE_SPEED),
                (TURN_PULSE_SPEED, -TURN_PULSE_SPEED)
            );
        }
    }

    #[test]
    fn differential_turn_is_gentler_than_in_place() {
        // 差速段(远处)单独用小一档幅值,原地转那档不受影响。
        assert_eq!(turn_amplitude(0.10), TURN_PULSE_SPEED_DIFF);
        assert_eq!(
            turn_amplitude(TURN_INPLACE_AREA - 0.01),
            TURN_PULSE_SPEED_DIFF
        );
        assert_eq!(turn_amplitude(TURN_INPLACE_AREA), TURN_PULSE_SPEED);
        assert_eq!(turn_amplitude(0.30), TURN_PULSE_SPEED);
        assert!(
            TURN_PULSE_SPEED_DIFF < TURN_PULSE_SPEED,
            "差速段要比原地转轻"
        );
        assert!(TURN_PULSE_SPEED_DIFF > 0);
    }

    #[test]
    fn differential_turn_keeps_both_wheels_rolling_forward() {
        // 差速段的整个意义就是两轮同向滚动(不靠静摩擦硬顶),幅值不能大到让内轮反转。
        for area in [0.0, 0.10, 0.15, 0.19] {
            let forward = approach_speed(area);
            let bias = turn_forward_bias(forward, area);
            let turn = turn_amplitude(area);
            assert!(bias - turn > 0, "area={area} 内轮反转了:{bias} - {turn}");
            assert!(bias + turn > 0);
        }
    }

    #[test]
    fn in_place_turn_does_not_slow_the_rotation() {
        // 原地转和带前进分量的差速转,两轮之差同为 2×turn —— 角速度一样,少的只是前冲。
        let area = 0.25;
        let bias = turn_forward_bias(approach_speed(area), area);
        let (l, r) = (bias + TURN_PULSE_SPEED, bias - TURN_PULSE_SPEED);
        assert_eq!(l - r, 2 * TURN_PULSE_SPEED);
        assert_eq!(l + r, 0, "原地转:两轮等幅反向,没有净前进");
        // 阈值必须落在抓取距离之内,否则"近了原地转"永远不会发生。
        assert!(TURN_INPLACE_AREA < GRAB_AREA);
    }

    /// `arm.grab()` 之后的收尾:进入 Closing,再由 Closing→Releasing→Homing 把球放开
    /// 并回到追球。这条链路是"抓完回到初始位置后放开,然后继续抓"的唯一实现。
    #[test]
    fn grab_tail_goes_home_then_releases() {
        let mut robot = RobotState::default();
        let now = Instant::now();
        finish_grab(&mut robot, now, false, GrabVerdict::Held { pulse: 1200 });

        // 不松爪、也不停在 Idle:先走归位/放开的三段。
        assert!(matches!(robot.grab_phase, GrabPhase::Closing { .. }));
    }

    /// 桶流程开启时,抓完最后一格不原地松爪:球保持在手上,直接交接给追桶状态机。
    #[test]
    fn finish_grab_hands_off_to_bucket_when_enabled() {
        let mut robot = RobotState::default();
        finish_grab(
            &mut robot,
            Instant::now(),
            true,
            GrabVerdict::Held { pulse: 1200 },
        );
        // 不进 Closing/Releasing:抓取状态机复位,球夹着,交给桶流程。
        assert!(matches!(robot.grab_phase, GrabPhase::Idle));
        assert!(robot.bucket.is_active());
    }

    /// 空合(夹爪一路闭合到位):球没夹住,即使开了桶流程也不能去追桶 ——
    /// 得开爪归位后回追球,重新走一轮 追球 → 抓球。
    #[test]
    fn empty_claw_skips_the_bucket_and_rechases() {
        let mut robot = RobotState {
            status: RobotStatus::GrabTennis,
            gate: GrabGate {
                ema: 0.5,
                engaged: true,
                since: Some(Instant::now()),
            },
            ..RobotState::default()
        };
        let (mut motor, mut arm) = (Motor::mock(), Arm::mock());
        finish_grab(
            &mut robot,
            Instant::now(),
            true,
            GrabVerdict::Missed { pulse: 790 },
        );

        assert!(!robot.bucket.is_active(), "空合不能进桶流程");
        assert!(
            matches!(robot.grab_phase, GrabPhase::Closing { .. }),
            "应走开爪归位的收尾"
        );
        assert!(!robot.gate.engaged && robot.gate.since.is_none(), "门控应复位");

        // 收尾三段推完后必须回到追球,而不是停住。
        for phase in [
            GrabPhase::Closing {
                until: Instant::now() - Duration::from_millis(1),
            },
            GrabPhase::Releasing {
                until: Instant::now() - Duration::from_millis(1),
            },
            GrabPhase::Homing {
                until: Instant::now() - Duration::from_millis(1),
            },
        ] {
            robot.grab_phase = phase;
            advance_grab(&mut robot, &mut motor, &mut arm, true);
        }
        assert!(matches!(robot.grab_phase, GrabPhase::Idle));
        assert!(matches!(robot.status, RobotStatus::ChaseTennis));
        assert!(!robot.bucket.is_active());
    }

    /// 收尾三段走完必须回到追球状态 —— 否则抓一次就停机,不再抓下一个球。
    #[test]
    fn grab_release_tail_resumes_chase() {
        let mut robot = RobotState {
            status: RobotStatus::GrabTennis,
            miss_count: 5,
            // 残留的确认态:门控已 engage 且驻留计时已起头。
            gate: GrabGate {
                ema: 0.5,
                engaged: true,
                since: Some(Instant::now()),
            },
            ..RobotState::default()
        };
        let (mut motor, mut arm) = (Motor::mock(), Arm::mock());

        // 每个阶段都把 until 设成已过期,一次调用推进一步。
        robot.grab_phase = GrabPhase::Closing {
            until: Instant::now() - Duration::from_millis(1),
        };
        advance_grab(&mut robot, &mut motor, &mut arm, false);
        assert!(matches!(robot.grab_phase, GrabPhase::Releasing { .. }));

        robot.grab_phase = GrabPhase::Releasing {
            until: Instant::now() - Duration::from_millis(1),
        };
        advance_grab(&mut robot, &mut motor, &mut arm, false);
        assert!(matches!(robot.grab_phase, GrabPhase::Homing { .. }));

        robot.grab_phase = GrabPhase::Homing {
            until: Instant::now() - Duration::from_millis(1),
        };
        advance_grab(&mut robot, &mut motor, &mut arm, false);

        assert!(
            matches!(robot.grab_phase, GrabPhase::Idle),
            "应回到空闲,准备抓下一个球"
        );
        assert!(
            matches!(robot.status, RobotStatus::ChaseTennis),
            "应恢复追球"
        );
        assert_eq!(robot.miss_count, 0);
        assert!(
            !robot.gate.engaged && robot.gate.since.is_none() && robot.gate.ema == 0.0,
            "门控应整体复位,带新球重新计时/滤波"
        );
    }

    /// 后退脉冲一次调用跑满配置时长,然后进入静止观测窗 —— 退多少只由
    /// `grab_backoff_ms` 决定。旧实现单次调用只驱动 0ms、停在 Backoff 等下一帧,
    /// 帧一卡就多退一段(现场表现为"偶尔大幅后退")。
    #[test]
    fn backoff_pulse_runs_whole_duration_in_one_call() {
        let cfg = config::Config::default();
        let mut robot = RobotState {
            grab_phase: GrabPhase::Backoff,
            status: RobotStatus::GrabTennis,
            ..RobotState::default()
        };
        let (mut motor, mut arm) = (Motor::mock(), Arm::mock());
        let start = Instant::now();
        advance_grab(&mut robot, &mut motor, &mut arm, false);
        assert!(
            start.elapsed() >= Duration::from_millis(cfg.grab.grab_backoff_ms),
            "一次调用应把后退脉冲跑满,实际 {:?}",
            start.elapsed()
        );
        assert!(
            matches!(robot.grab_phase, GrabPhase::BackoffSettle { .. }),
            "退完先进观测窗,不该同一帧就去复判"
        );
    }

    /// 观测窗内不推进(车静止,只累积新读数);到期才按 ema 复判。
    #[test]
    fn backoff_settle_waits_for_fresh_frames_then_rejudges() {
        let cfg = config::Config::default();
        let (mut motor, mut arm) = (Motor::mock(), Arm::mock());
        let band = (cfg.grab.grab_area + cfg.grab.grab_area_max) / 2.0;

        let mut waiting = RobotState {
            grab_phase: GrabPhase::BackoffSettle {
                until: Instant::now() + Duration::from_secs(60),
            },
            gate: GrabGate {
                ema: band,
                ..GrabGate::default()
            },
            ..RobotState::default()
        };
        advance_grab(&mut waiting, &mut motor, &mut arm, false);
        assert!(
            matches!(waiting.grab_phase, GrabPhase::BackoffSettle { .. }),
            "观测窗未到期不该推进"
        );

        let mut expired = RobotState {
            grab_phase: GrabPhase::BackoffSettle {
                until: Instant::now() - Duration::from_millis(1),
            },
            gate: GrabGate {
                ema: band,
                ..GrabGate::default()
            },
            ..RobotState::default()
        };
        advance_grab(&mut expired, &mut motor, &mut arm, false);
        assert!(
            matches!(expired.grab_phase, GrabPhase::Turn),
            "落在可抓带内应进盲转"
        );
    }

    /// 盲转时间表:每段固定时长,段间静置,末尾留一段停稳再下爪。
    /// 旧实现把这套时序交给逐帧状态机推进,帧周期(~60ms,相机卡顿更久)
    /// 直接量化了脉冲长度 —— 同样的配置有时只转半截、有时一股脑转过去。
    #[test]
    fn blind_turn_plan_uses_fixed_duration_segments() {
        let cfg = config::Config::default();
        let plan = blind_turn_plan(&cfg);

        // 段序固定:转-停-转-停…-停。首段必须是转(否则先空等),末段必须是静置。
        assert_eq!(plan.first(), Some(&TurnStep::Drive(cfg.grab.grab_turn_ms)));
        assert_eq!(
            plan.last(),
            Some(&TurnStep::Settle(cfg.grab.grab_turn_gap_ms))
        );
        // 每段时长都取配置值,不掺任何"上一帧/下一帧"的量化。
        for step in &plan {
            match step {
                TurnStep::Drive(ms) => assert_eq!(*ms, cfg.grab.grab_turn_ms),
                TurnStep::Settle(ms) => assert_eq!(*ms, cfg.grab.grab_turn_gap_ms),
            }
        }
        // 有 count 次转就有 count 次停(段间 + 末尾各一次)。
        let count = cfg.grab.grab_turn_count.max(1) as usize;
        assert_eq!(plan.len(), count * 2);

        // 段数跟着 grab_turn_count 走。
        let mut more = cfg;
        more.grab.grab_turn_count = 4;
        assert_eq!(blind_turn_plan(&more).len(), 8);

        // 0 / 负数退化成一次:不能"转了但不抓"。
        let mut zero = cfg;
        zero.grab.grab_turn_count = 0;
        assert_eq!(
            blind_turn_plan(&zero),
            vec![
                TurnStep::Drive(cfg.grab.grab_turn_ms),
                TurnStep::Settle(cfg.grab.grab_turn_gap_ms),
            ]
        );
    }

    /// 整段盲转一次调用跑满全部计划时长 —— 脉冲长度由墙钟决定,
    /// 不随调用频次(帧节奏)伸缩。
    #[test]
    fn blind_turn_sequence_runs_whole_plan_in_one_call() {
        let cfg = config::Config::default();
        let turns = cfg.grab.grab_turn_count.max(1) as u64;
        let expected =
            Duration::from_millis(turns * (cfg.grab.grab_turn_ms + cfg.grab.grab_turn_gap_ms));
        let mut motor = Motor::mock();
        let start = Instant::now();
        blind_turn_sequence(&mut motor);
        assert!(
            start.elapsed() >= expected,
            "盲转必须一次性跑满 {expected:?},实际 {:?}",
            start.elapsed()
        );
    }

    /// RobotState 必须保持 Copy:主循环把它当值传递。
    #[test]
    fn robot_state_stays_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<RobotState>();
    }

    #[test]
    fn approach_speed_grades_with_distance() {
        assert_eq!(approach_speed(0.05), CHASE_SPEED_FAR);
        assert_eq!(approach_speed(0.20), CHASE_SPEED_MID);
        assert_eq!(approach_speed(0.35), CHASE_SPEED_NEAR);
        // 已进入抓取距离:只对准,不再前进。
        assert_eq!(approach_speed(GRAB_AREA), 0);
        assert!(CHASE_SPEED_FAR > CHASE_SPEED_MID && CHASE_SPEED_MID > CHASE_SPEED_NEAR);
        // 最低档不能低于实测能推得动整车的量级(老 CHASE_SPEED = 20)。
        assert!(CHASE_SPEED_NEAR >= 20);
    }

    /// 抓取盲转方向把两轮目标镜像:`Left` 左转、`Right` 右转。
    #[test]
    fn grab_turn_direction_mirrors_wheels() {
        // Left:左轮回退、右轮前进 = 左转。
        assert_eq!(grab_turn_wheels(18, GrabTurnDirection::Left), (-18, 18));
        // Right:左右互换 = 右转。
        assert_eq!(grab_turn_wheels(18, GrabTurnDirection::Right), (18, -18));
        // 两种方向互为镜像,幅值不变(角速度大小不受方向影响)。
        let (l, r) = grab_turn_wheels(18, GrabTurnDirection::Left);
        let (rl, rr) = grab_turn_wheels(18, GrabTurnDirection::Right);
        assert_eq!((rl, rr), (r, l));
    }

    /// `[chassis] turn_reverse` 镜像所有闭环差的接线极性(追球/搜索/追桶,
    /// 见 `bucket.rs`);抓取盲转方向独立,由 `[grab] grab_turn_direction` 决定。
    #[test]
    fn turn_reverse_mirrors_closed_loops_not_blind_grab() {
        // 追球:球在左 (dir=-1)。默认极性不翻转 → 名义左转。
        assert_eq!(steer_turn(-1, TURN_PULSE_SPEED, 1), -TURN_PULSE_SPEED);
        // 通道反接 (polarity=-1) → 转向量镜像,经反接接线后物理仍朝球一侧。
        assert_eq!(steer_turn(-1, TURN_PULSE_SPEED, -1), TURN_PULSE_SPEED);
        // 幅值与极性无关,只改方向。
        assert_eq!(steer_turn(1, 12, -1).abs(), 12);
        // 抓取盲转不吃 turn_reverse:同一方向给出固定两轮目标,与极性开关无关。
        assert_eq!(grab_turn_wheels(18, GrabTurnDirection::Left), (-18, 18));
        assert_eq!(grab_turn_wheels(18, GrabTurnDirection::Right), (18, -18));
    }

    // ── 抓取门控:EMA + 滞回 + 时间窗 ────────────────────────────────

    /// 门控类测试的公共配置:触发线 0.30,带宽 10%(退出线 0.27),驻留 300ms。
    /// 不用全局默认值,免被常量调参带走。
    fn gate_cfg() -> config::Config {
        let mut cfg = config::Config::default();
        cfg.grab.grab_area = 0.30;
        cfg.grab.grab_area_max = 0.35;
        cfg.grab.grab_hysteresis_frac = 0.1;
        cfg.grab.grab_confirm_ms = 300;
        cfg.grab.area_filter_alpha = 0.4;
        cfg
    }

    /// 面积在阈值线 ±0.01 交替抖动:旧的"连续 5 帧原始面积 ≥ grab_area"硬计数永远
    /// 被抖动帧清零、永不抓取;EMA+滞回+时间窗驻留仍能在驻留到点后触发。
    /// 这条对旧计数实现是必失败回归。
    #[test]
    fn grab_gate_hysteresis_survives_area_jitter() {
        let cfg = gate_cfg();
        let t0 = Instant::now();
        let mut gate = GrabGate::default();
        let mut fired = false;
        for i in 0..16u64 {
            let raw = if i % 2 == 0 { 0.29 } else { 0.31 };
            gate.observe(raw, &cfg);
            if gate.step(true, &cfg, t0 + Duration::from_millis(i * 50)) {
                fired = true;
                break;
            }
        }
        assert!(fired, "面积交替抖动不应无限复位驻留计时");
        assert!(gate.engaged);
    }

    /// 驻留窗按时间不按帧数:同一面积序列,10fps 与 20fps 的触发时刻相对首次
    /// engage 都是 300ms 墙钟(旧帧数计数在 10fps 要 500ms、20fps 只要 250ms)。
    #[test]
    fn grab_gate_dwell_is_time_not_frame_based() {
        let cfg = gate_cfg();
        let t0 = Instant::now();
        for step_ms in [100u64, 50] {
            let mut gate = GrabGate::default();
            let mut fired_at = None;
            for i in 0..12u64 {
                gate.observe(0.35, &cfg);
                if gate.step(true, &cfg, t0 + Duration::from_millis(i * step_ms)) {
                    fired_at = Some(i * step_ms);
                    break;
                }
            }
            assert_eq!(
                fired_at,
                Some(300),
                "step={step_ms}ms 都应在 engage 后 300ms 触发"
            );
        }
    }

    /// 滞回:ema 跌不过 `grab_area - band` 就不退出 engage;跌出带宽才退出并清计时。
    #[test]
    fn grab_gate_disengages_below_hysteresis_band() {
        let cfg = gate_cfg();
        let t0 = Instant::now();
        let mut gate = GrabGate::default();
        gate.observe(0.31, &cfg);
        gate.step(true, &cfg, t0);
        assert!(gate.engaged);
        // 带内回落(0.29 > 0.27):仍 engaged。
        gate.observe(0.29, &cfg);
        gate.step(true, &cfg, t0 + Duration::from_millis(100));
        assert!(gate.engaged, "阈值带内抖动靠滞回保持 engage");
        // 持续低于退出线:退出并清掉驻留计时。
        for i in 0..3 {
            gate.observe(0.10, &cfg);
            gate.step(true, &cfg, t0 + Duration::from_millis(200 + i * 50));
        }
        assert!(!gate.engaged, "ema 跌过 grab_area - band 应退出");
        assert!(gate.since.is_none(), "退出后驻留计时应清零");
    }

    /// 门控已 engage 但还没居中时,直线段应停止前进:原地转等居中,
    /// 不能边等边蹭近,否则触发距离不再对应 grab_area。
    #[test]
    fn engaged_straight_holds_forward() {
        // 正常追球:直线段按面积分档前进。
        assert_eq!(straight_forward(0.10, false), approach_speed(0.10));
        assert_eq!(straight_forward(0.30, false), approach_speed(0.30));
        // engage 后,不管面积多大,直线段停住。
        assert_eq!(straight_forward(0.30, true), 0);
        assert_eq!(straight_forward(0.10, true), 0);
    }

    /// Backoff 到期的复判决策表:仍太近(面积达上限或球底出镜头)→再退
    /// (有重试上限),退过头/封顶仍贴脸→取消,落在带内→盲转。
    #[test]
    fn backoff_next_decision_table() {
        let cfg = gate_cfg(); // area=0.30, max=0.35
        assert_eq!(backoff_next(0.42, false, &cfg, 0), BackoffNext::ReBackoff);
        assert_eq!(
            backoff_next(0.42, false, &cfg, GRAB_BACKOFF_RETRY_MAX),
            BackoffNext::Abort,
            "重试封顶仍太近:回追球重新逼近,不盲抓"
        );
        assert_eq!(
            backoff_next(0.10, true, &cfg, 0),
            BackoffNext::ReBackoff,
            "裁切区面积读数失真(ema 被钉低),球底出镜头单独就构成继续后退"
        );
        assert_eq!(
            backoff_next(0.10, true, &cfg, GRAB_BACKOFF_RETRY_MAX),
            BackoffNext::Abort,
            "封顶后仍贴脸也不得硬抓"
        );
        assert_eq!(backoff_next(0.32, false, &cfg, 0), BackoffNext::Turn);
        assert_eq!(
            backoff_next(0.25, false, &cfg, 0),
            BackoffNext::Abort,
            "退过头应回追球重新逼近"
        );
        assert_eq!(
            backoff_next(0.0, false, &cfg, 0),
            BackoffNext::Abort,
            "没看到球(未观测)不得硬抓"
        );
    }

    /// 球底贴出画面底边即判裁切;余量 BOTTOM_CLIP_PX 内算贴边。
    #[test]
    fn ball_bottom_clipped_boundary() {
        let mut d = Detection {
            bbox: Box2d {
                x: 320.0,
                y: 150.0,
                w: 300.0,
                h: 300.0,
            },
            ..Detection::default()
        }; // y2 = 300
        assert!(
            !ball_bottom_clipped(&d, 360),
            "框底远离底边、长宽等形不算裁切"
        );
        d.bbox.y = 208.0; // y2 = 358 = 360 - BOTTOM_CLIP_PX
        assert!(ball_bottom_clipped(&d, 360), "贴到余量线即算裁切");
        d.bbox.y = 207.0; // y2 = 357 < 358
        assert!(ball_bottom_clipped(&d, 360), "探进余量线以内同样算裁切");
        d.bbox.y = 205.0; // y2 = 355,距底边 5px > 4px 余量
        assert!(
            !ball_bottom_clipped(&d, 360),
            "余量外不判裁切,避免边缘抖动误触发"
        );
    }

    /// 长宽差信号:框被削矮(w−h 超 20px)时即使没贴到画面底边也算出镜头;
    /// 实测锚点:全框 303×295(差 8)不触发,裁切帧 395×339(差 56)触发。
    #[test]
    fn ball_clipped_by_aspect_distortion() {
        let mut d = Detection {
            bbox: Box2d {
                x: 320.0,
                y: 132.5,
                w: 303.0,
                h: 295.0,
            },
            ..Detection::default()
        }; // y2 = 280,armtest 最佳爪位,完整在框内
        assert!(
            !ball_bottom_clipped(&d, 360),
            "全框贴脸(差 8px)不该误触裁切判定"
        );
        d.bbox = Box2d {
            x: 320.0,
            y: 150.0,
            w: 395.0,
            h: 339.0,
        }; // y2 = 319,未贴底边,但 w−h = 56 > 20
        assert!(ball_bottom_clipped(&d, 360), "削矮畸变单独构成出镜头判定");
        d.bbox = Box2d {
            x: 320.0,
            y: 150.0,
            w: 300.0,
            h: 279.0,
        }; // w−h = 21,刚过阈值
        assert!(ball_bottom_clipped(&d, 360));
        d.bbox = Box2d {
            x: 320.0,
            y: 150.0,
            w: 300.0,
            h: 281.0,
        }; // w−h = 19,阈值内不触发
        assert!(!ball_bottom_clipped(&d, 360));
    }
}
