use crate::arm::Arm;
use crate::camera::UsbCamera;
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
    pub max_frames: Option<u64>,
}

impl Default for RobotConfig {
    fn default() -> Self {
        Self {
            inference: InferenceConfig::default(),
            max_frames: None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum RobotStatus {
    ChaseTennis,
    GrabTennis,
}

#[derive(Clone, Copy, Debug)]
struct RobotState {
    status: RobotStatus,
    area_ratio: f32,
    ball_cx: i32,
    grab_confirm_count: i32,
    // 连续漏检帧数;在宽限期内先保持不动等球重现,超过阈值才判定真丢并搜索。
    miss_count: i32,
}

impl Default for RobotState {
    fn default() -> Self {
        Self {
            status: RobotStatus::ChaseTennis,
            area_ratio: 0.0,
            ball_cx: 0,
            grab_confirm_count: 0,
            miss_count: 0,
        }
    }
}

// 参考帧宽度。像素类阈值(如中心死区)以此为基准按分辨率缩放,保证不同分辨率下行为一致。
const REFERENCE_FRAME_WIDTH: f32 = 640.0;
// 基础中心死区(像素,基于 640 宽)。球中心偏离画面中心在此范围内即视为“已居中”,不再转向,避免对着中心反复微调。
const CENTER_MARGIN: i32 = 50;
// 转向脉冲正比于“居中误差”(offset),而非距离。
// 偏差达到半屏时对应 TURN_PULSE_GAIN_US;偏差越小脉冲越短(趋向 TURN_PULSE_MIN_US),让对准收敛而不是来回摆。
const TURN_PULSE_GAIN_US: f32 = 250_000.0;
// 接近阻尼下限。相机装在旋转中心前方,原地旋转时越近的球在画面里划过越快;
// 阻尼系数 = clamp(1 - area_ratio, 此值, 1),球越大(越近)脉冲压得越小,消除近距离的过冲/摇头。
const PROXIMITY_DAMP_MIN: f32 = 0.35;
// 中心死区随接近放大的系数。有效死区 = CENTER_MARGIN × (1 + 此值 × area_ratio);
// 抓球前不再追求亚像素级居中(抓取自带左转补偿),近处放宽容差直接消除小幅 hunting。
const MARGIN_PROXIMITY_GAIN: f32 = 2.0;

// 球框面积占画面比 ≥ 此值 → 认为“够近可抓”,是触发抓取的主要距离阈值。
const GRAB_AREA: f32 = 0.40;
// 面积比 ≥ 此值 → “太近了”,先后退一点再抓,避免撞飞球。必须 > GRAB_AREA。
const GRAB_AREA_MAX: f32 = 0.55;

// 抓取前小幅左转微调的次数(补偿爪子相对相机的安装偏心,把球挪到爪子正前方)。
const GRAB_LEFT_TURN_COUNT: i32 = 2;
// 连续满足“够近且居中”的帧数达到此值才真正执行抓取,防单帧误检导致误抓。
const GRAB_CONFIRM_THRESHOLD: i32 = 5;
// 丢失宽限帧数。看到过球后若漏检,先原地保持不动等它重现,连续漏检超过此值才判定真丢并开始搜索,
// 避免转向时的偶发漏检(运动模糊/擦边)立刻触发盲搜把球搞丢。
const LOST_GRACE_FRAMES: i32 = 8;

// 已居中但还没够近时的直线追球速度(1–100 等级,经 motor 死区映射后转 PWM)。
const CHASE_SPEED: i32 = 20;
// 原地转向对准速度。太小转不动,太大转过头。
const TURN_SPEED: i32 = 22;
// 没检测到球时原地慢转搜索的速度。
const IDLE_SPEED: i32 = 18;
// 抓取前左转微调的速度。
const GRAB_LEFT_TURN_SPEED: i32 = 18;
// 太近(≥ GRAB_AREA_MAX)时的后退速度。
const BACKWARD_SPEED: i32 = 18;

// 单次转向的最长时长上限,防大偏差时猛甩过头。
const TURN_PULSE_MAX_US: u64 = 250_000;
// 单次转向的最短时长下限。短于此电机来不及克服静摩擦(“转不动”),故设地板。
const TURN_PULSE_MIN_US: u64 = 55_000;
// 对准脉冲后的静置时长,让轮子停稳再取下一帧,减少运动模糊导致的漏检。
const ALIGN_SETTLE_US: u64 = 60_000;
// 太近时后退的时长。
const BACKWARD_PULSE_US: u64 = 200_000;
// 抓取前每次左转微调的时长。
const GRAB_LEFT_TURN_US: u64 = 250_000;

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

    while !stop_requested() {
        if let Some(limit) = config.max_frames {
            if frame_idx >= limit {
                break;
            }
        }

        frame_idx += 1;

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
            ModelInputFormat::Yuv420Planar => {
                image_bridge::yuv422p_to_yuv420p(&frame.pixels, &mut input_buffer, model_w, model_h)
            }
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
        handle_detections(
            &detections,
            image_w,
            image_h,
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
        println!(
            "[FPS] {:.2} avg: {:.2} ({:.1}ms, control={:.1}ms)",
            fps,
            avg_fps,
            frame_time.as_secs_f32() * 1000.0,
            control_time.as_secs_f32() * 1000.0
        );
    }

    motor.standby();
}

fn handle_detections(
    detections: &[Detection],
    image_w: i32,
    image_h: i32,
    robot: &mut RobotState,
    motor: &mut Motor,
    arm: &mut Arm,
) {
    if detections.is_empty() {
        robot.grab_confirm_count = 0;
        robot.miss_count += 1;

        if robot.miss_count <= LOST_GRACE_FRAMES {
            // 宽限期内:刚才还看得到球,先停住等它重现,别急着盲搜。
            eprintln!(
                "[detect] missed {}/{} frames, holding",
                robot.miss_count, LOST_GRACE_FRAMES
            );
            motor.standby();
        } else {
            // 真丢了:朝球最后出现的一侧转,而不是固定方向盲搜。
            robot.status = RobotStatus::ChaseTennis;
            let last_offset = robot.ball_cx - image_w / 2;
            if last_offset >= 0 {
                eprintln!("[detect] ball lost, searching right (last seen right)");
                motor.drive(IDLE_SPEED, -IDLE_SPEED);
            } else {
                eprintln!("[detect] ball lost, searching left (last seen left)");
                motor.drive(-IDLE_SPEED, IDLE_SPEED);
            }
        }
        return;
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
    let pulse_us = turn_pulse_us(offset, image_w, area_ratio);

    robot.area_ratio = area_ratio;
    robot.ball_cx = ball_cx;

    eprintln!(
        "[detect] area={area_ratio:.3} cx={ball_cx} conf={:.3} centered={centered}",
        best.score
    );

    if area_ratio >= GRAB_AREA && centered {
        robot.status = RobotStatus::GrabTennis;
        robot.grab_confirm_count += 1;

        if area_ratio >= GRAB_AREA_MAX {
            eprintln!("[grab] too close, backing up");
            motor.backward(BACKWARD_SPEED);
            sleep_us(BACKWARD_PULSE_US);
            motor.standby();
        } else {
            motor.standby();
        }

        if robot.grab_confirm_count >= GRAB_CONFIRM_THRESHOLD {
            if area_ratio >= GRAB_AREA_MAX {
                motor.backward(BACKWARD_SPEED);
                sleep_us(BACKWARD_PULSE_US);
                motor.standby();
            }

            for _ in 0..GRAB_LEFT_TURN_COUNT {
                motor.drive(-GRAB_LEFT_TURN_SPEED, GRAB_LEFT_TURN_SPEED);
                sleep_us(GRAB_LEFT_TURN_US);
                motor.standby();
                sleep_us(100_000);
            }

            arm.grab();
            sleep_us(2_000_00);
            arm.release();
            sleep_us(1_000_00);
            arm.grab_pos();
            sleep_us(1_000_00);

            robot.grab_confirm_count = 0;
            robot.status = RobotStatus::ChaseTennis;
        }
    } else if area_ratio >= GRAB_AREA && !centered {
        robot.grab_confirm_count = 0;
        align(offset, pulse_us, motor);
    } else {
        robot.grab_confirm_count = 0;
        robot.status = RobotStatus::ChaseTennis;
        if centered {
            motor.forward(CHASE_SPEED);
        } else {
            align(offset, pulse_us, motor);
        }
    }
}

fn align(offset: i32, pulse_us: u64, motor: &mut Motor) {
    if offset < 0 {
        motor.drive(-TURN_SPEED, TURN_SPEED);
    } else {
        motor.drive(TURN_SPEED, -TURN_SPEED);
    }
    sleep_us(pulse_us);
    motor.standby();
    // 等轮子停稳再让主循环取下一帧,避免运动模糊导致漏检丢球。
    sleep_us(ALIGN_SETTLE_US);
}

fn scaled_center_margin(image_w: i32) -> i32 {
    ((CENTER_MARGIN as f32) * image_w as f32 / REFERENCE_FRAME_WIDTH)
        .round()
        .max(1.0) as i32
}

fn proximity_center_margin(image_w: i32, area_ratio: f32) -> i32 {
    let base = scaled_center_margin(image_w) as f32;
    (base * (1.0 + MARGIN_PROXIMITY_GAIN * area_ratio.max(0.0)))
        .round()
        .max(1.0) as i32
}

fn turn_pulse_us(offset: i32, image_w: i32, area_ratio: f32) -> u64 {
    let half_width = (image_w / 2).max(1) as f32;
    let norm_offset = (offset.abs() as f32 / half_width).min(1.0);
    let proximity_damp = (1.0 - area_ratio).clamp(PROXIMITY_DAMP_MIN, 1.0);
    ((TURN_PULSE_GAIN_US * norm_offset * proximity_damp) as u64)
        .clamp(TURN_PULSE_MIN_US, TURN_PULSE_MAX_US)
}

fn sleep_us(us: u64) {
    thread::sleep(Duration::from_micros(us));
}

#[cfg(test)]
mod tests {
    use super::{
        proximity_center_margin, scaled_center_margin, turn_pulse_us, TURN_PULSE_MAX_US,
        TURN_PULSE_MIN_US,
    };

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
    fn clamps_turn_pulse() {
        // Zero offset floors to the minimum step.
        assert_eq!(turn_pulse_us(0, 640, 0.0), TURN_PULSE_MIN_US);
        // A full half-frame offset saturates to the maximum.
        assert_eq!(turn_pulse_us(1000, 640, 0.0), TURN_PULSE_MAX_US);
        // A larger offset yields a longer pulse than a smaller one.
        assert!(turn_pulse_us(160, 640, 0.0) > turn_pulse_us(60, 640, 0.0));
        // A nearby ball (large area) is damped below the same offset far away.
        assert!(turn_pulse_us(160, 640, 0.5) < turn_pulse_us(160, 640, 0.0));
    }
}
