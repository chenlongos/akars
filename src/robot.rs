use crate::arm::Arm;
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
    /// 太近:先后退一点再抓,免得把球顶飞。
    Backoff {
        until: Instant,
    },
    /// 抓取前盲转微调(补偿爪子相对相机的偏心,方向由 grab_turn_direction 决定),
    /// remaining 是还需再转的次数。
    Turn {
        until: Instant,
        remaining: i32,
    },
    /// 两次盲转之间停稳。
    TurnGap {
        until: Instant,
        remaining: i32,
    },
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

#[derive(Clone, Copy, Debug)]
struct RobotState {
    status: RobotStatus,
    area_ratio: f32,
    ball_cx: i32,
    grab_confirm_count: i32,
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
            grab_confirm_count: 0,
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
// 连续满足“够近且居中”的帧数达到此值才真正执行抓取,防单帧误检导致误抓。
pub(crate) const GRAB_CONFIRM_THRESHOLD: i32 = 5;
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
                robot.grab_confirm_count = 0;
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
    // 抓取序列进行中:主循环照常取帧/推理/打日志,这里只按时间戳推进各阶段。
    if !matches!(robot.grab_phase, GrabPhase::Idle) {
        advance_grab(robot, motor, arm, bucket_enabled());
        return None;
    }

    let cfg = config::get();

    if detections.is_empty() {
        robot.grab_confirm_count = 0;
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
    }
    robot.miss_count = 0;

    let best = detections
        .iter()
        .max_by(|a, b| {
            let area_a = a.bbox.w * a.bbox.h;
            let area_b = b.bbox.w * b.bbox.h;
            area_a.total_cmp(&area_b)
        })
        .expect("non-empty detections");

    let image_area = (image_w.max(1) * image_h.max(1)) as f32;
    let area_ratio = (best.bbox.w * best.bbox.h) / image_area;
    let ball_cx = best.bbox.x as i32;
    let center = image_w / 2;
    let center_margin = proximity_center_margin(image_w, area_ratio);
    let offset = ball_cx - center;
    let centered = offset.abs() <= center_margin;

    robot.area_ratio = area_ratio;
    robot.ball_cx = ball_cx;

    eprintln!(
        "[detect] area={area_ratio:.3} cx={ball_cx} conf={:.3} centered={centered}",
        best.score
    );

    // 够近且居中:站定确认若干帧(防单帧误检),确认够了就交给抓取状态机。
    if area_ratio >= cfg.grab.grab_area && centered {
        robot.status = RobotStatus::GrabTennis;
        robot.grab_confirm_count += 1;
        robot.steer_phase = SteerPhase::Straight; // 已站定,滞环状态跟着复位
        motor.standby();

        if robot.grab_confirm_count >= cfg.grab.grab_confirm_threshold {
            start_grab(robot, area_ratio);
        }
        return Some(offset);
    }

    robot.grab_confirm_count = 0;
    robot.status = RobotStatus::ChaseTennis;
    servo_step(offset, prev_offset, image_w, area_ratio, robot, motor);
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

    let forward = approach_speed(area_ratio);
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
            motor.standby();
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

/// 功能开关：「抓起来 → 追蓝桶 → 放进桶」流程，由配置文件
/// `[grab] chase_bucket` 决定（缺省开）；临时调试还可用 `AKARS_NO_BUCKET`
/// 环境变量强制关掉，不必改卡上的配置。
fn bucket_enabled() -> bool {
    config::get().grab.chase_bucket && std::env::var("AKARS_NO_BUCKET").is_err()
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

/// 确认够近且居中后进入抓取序列。太近的话先安排一段后退,否则直接开始左转微调。
fn start_grab(robot: &mut RobotState, area_ratio: f32) {
    let now = Instant::now();
    let cfg = config::get();
    robot.grab_confirm_count = 0;
    robot.steer_phase = SteerPhase::Straight;

    if area_ratio >= cfg.grab.grab_area_max {
        eprintln!("[grab] too close, backing up first");
        robot.grab_phase = GrabPhase::Backoff {
            until: now + Duration::from_millis(cfg.grab.grab_backoff_ms),
        };
    } else {
        robot.grab_phase = GrabPhase::Turn {
            until: now + Duration::from_millis(cfg.grab.grab_turn_ms),
            remaining: cfg.grab.grab_turn_count - 1,
        };
    }
}

/// 抓取序列走完最后一格之后的去向。
///
/// `bucket_mode` 为真:球已在手上,不进放开的三段收尾,而是直接转入
/// “追蓝桶 → 放进桶”流程(`robot.bucket.start()`),由主循环的 `bucket.is_active()`
/// 分支接管。为假(旧行为):回初始位置 → 原地松爪 → 继续追球。
///
/// 单独抽出来是因为 `arm.grab()` 阻塞约 3.7s(`arm.rs:61-72`),想验证这一段就得绕开它。
fn finish_grab(robot: &mut RobotState, now: Instant, bucket_mode: bool) {
    if bucket_mode {
        // 与桶流程唯一的分岔点:不松爪,夹着球去找桶。
        eprintln!("[bucket] ball held -> searching for bucket");
        robot.grab_phase = GrabPhase::Idle;
        robot.status = RobotStatus::ChaseTennis;
        robot.grab_confirm_count = 0;
        robot.miss_count = 0;
        robot.steer_phase = SteerPhase::Straight;
        robot.bucket.start();
        return;
    }
    robot.grab_phase = GrabPhase::Closing {
        until: now + Duration::from_millis(config::get().grab.grab_close_ms),
    };
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

        GrabPhase::Backoff { until } => {
            if now < until {
                motor.backward(cfg.chassis.backward_speed);
                return;
            }
            motor.standby();
            robot.grab_phase = GrabPhase::Turn {
                until: now + Duration::from_millis(cfg.grab.grab_turn_ms),
                remaining: cfg.grab.grab_turn_count - 1,
            };
        }

        GrabPhase::Turn { until, remaining } => {
            if now < until {
                let (l, r) =
                    grab_turn_wheels(cfg.grab.grab_turn_speed, cfg.grab.grab_turn_direction);
                motor.drive(l, r);
                return;
            }
            motor.standby();
            robot.grab_phase = GrabPhase::TurnGap {
                until: now + Duration::from_millis(cfg.grab.grab_turn_gap_ms),
                remaining,
            };
        }

        GrabPhase::TurnGap { until, remaining } => {
            if now < until {
                return;
            }
            if remaining > 0 {
                robot.grab_phase = GrabPhase::Turn {
                    until: now + Duration::from_millis(cfg.grab.grab_turn_ms),
                    remaining: remaining - 1,
                };
            } else {
                // arm.grab() 阻塞约 3.7s,结束时夹爪闭合、机械臂停在 LIFT 位 —— 球在手上。
                arm.grab();
                finish_grab(robot, now, bucket_mode);
            }
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
            robot.grab_confirm_count = 0;
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
        CHASE_SPEED_FAR, CHASE_SPEED_MID, CHASE_SPEED_NEAR, CORRECT_SETTLE_MS, GRAB_AREA,
        GrabPhase, LANE_BREAK_PX, RobotState, RobotStatus, SteerPhase, TURN_INPLACE_AREA,
        TURN_PULSE_GAIN_MS, TURN_PULSE_MAX_MS, TURN_PULSE_MIN_MS, TURN_PULSE_SPEED,
        TURN_PULSE_SPEED_DIFF, advance_grab, advance_steer_phase, approach_speed, brake_scale,
        finish_grab, grab_turn_wheels, proximity_center_margin, pulse_ms, scaled_center_margin,
        scaled_px, steer_turn, turn_amplitude, turn_forward_bias,
    };
    use crate::arm::Arm;
    use crate::config::GrabTurnDirection;
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
        finish_grab(&mut robot, now, false);

        // 不松爪、也不停在 Idle:先走归位/放开的三段。
        assert!(matches!(robot.grab_phase, GrabPhase::Closing { .. }));
    }

    /// 桶流程开启时,抓完最后一格不原地松爪:球保持在手上,直接交接给追桶状态机。
    #[test]
    fn finish_grab_hands_off_to_bucket_when_enabled() {
        let mut robot = RobotState::default();
        finish_grab(&mut robot, Instant::now(), true);
        // 不进 Closing/Releasing:抓取状态机复位,球夹着,交给桶流程。
        assert!(matches!(robot.grab_phase, GrabPhase::Idle));
        assert!(robot.bucket.is_active());
    }

    /// 收尾三段走完必须回到追球状态 —— 否则抓一次就停机,不再抓下一个球。
    #[test]
    fn grab_release_tail_resumes_chase() {
        let mut robot = RobotState {
            status: RobotStatus::GrabTennis,
            miss_count: 5,
            grab_confirm_count: 5,
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
        assert_eq!(robot.grab_confirm_count, 0);
    }

    /// 空转中段的停稳间隔不该改道 —— 出口只发生在 remaining == 0 那一格。
    #[test]
    fn grab_tail_only_diverts_on_the_last_turn() {
        let mut robot = RobotState {
            grab_phase: GrabPhase::TurnGap {
                until: Instant::now() - Duration::from_millis(1),
                remaining: 2,
            },
            ..RobotState::default()
        };
        let (mut motor, mut arm) = (Motor::mock(), Arm::mock());
        advance_grab(&mut robot, &mut motor, &mut arm, false);
        assert!(matches!(
            robot.grab_phase,
            GrabPhase::Turn { remaining: 1, .. }
        ));
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
}
