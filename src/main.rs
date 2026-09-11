use akars::arm::{Arm, ArmMotionConfig};
use akars::camera::UsbCamera;
use akars::config::{load_arm_motion_config, load_robot_config, RobotConfigFile};
use akars::motor::Motor;
use akars::robot::{install_signal_handlers, run_tennis_hunter, RobotConfig};
use akars::tpu::open_model;
use akars::web::{serve, WebConfig};
use std::env;
use std::path::PathBuf;

const DEFAULT_ROBOT_CONFIG: &str = "./robot.toml";
const DEFAULT_ARM_MOTION_CONFIG: &str = "./arm.json";

/// 子命令 hunt 特有参数。
#[derive(Debug)]
struct HuntCli {
    model: PathBuf,
    frames: Option<u64>,
}

/// 子命令 detect 特有参数。
#[derive(Debug)]
struct DetectCli {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
}

/// 子命令 capture 特有参数。
#[derive(Debug)]
struct CaptureCli {
    output: PathBuf,
    warmup: u32,
}

/// 子命令 armtest 特有参数。
#[derive(Debug)]
struct ArmTestCli {
    servo0: f32,
    servo1: f32,
    servo2: f32,
    time_ms: i32,
}

#[derive(Debug)]
enum Command {
    Hunt(HuntCli),
    Serve,
    Detect(DetectCli),
    Capture(CaptureCli),
    ArmTest(ArmTestCli),
}

impl Default for HuntCli {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            frames: None,
        }
    }
}

impl Default for DetectCli {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            input: PathBuf::new(),
            output: PathBuf::from("detect_out.jpg"),
        }
    }
}

impl Default for CaptureCli {
    fn default() -> Self {
        Self {
            output: PathBuf::from("capture.jpg"),
            warmup: 30,
        }
    }
}

impl Default for ArmTestCli {
    fn default() -> Self {
        Self {
            servo0: 0.0,
            servo1: 0.0,
            servo2: 0.0,
            time_ms: 1000,
        }
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let mut args = args.into_iter().peekable();

    // ── 第一步：解析全局参数 --robot / --arm（放在子命令之前）─────────
    let mut robot_config_path = PathBuf::from(DEFAULT_ROBOT_CONFIG);
    let mut arm_motion_path = PathBuf::from(DEFAULT_ARM_MOTION_CONFIG);

    while let Some(arg) = args.peek() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--robot" => {
                args.next();
                let path = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--robot expects a file path");
                        print_usage();
                        std::process::exit(2);
                    }
                };
                robot_config_path = PathBuf::from(path);
            }
            "--arm" => {
                args.next();
                let path = match args.next() {
                    Some(v) => v,
                    None => {
                        eprintln!("--arm expects a file path");
                        print_usage();
                        std::process::exit(2);
                    }
                };
                arm_motion_path = PathBuf::from(path);
            }
            _ => break, // 遇到子命令或位置参数，停止解析全局参数
        }
    }

    // ── 第二步：加载配置文件 ────────────────────────────────────────
    let robot_config = match load_robot_config(&robot_config_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };
    let arm_motion = match load_arm_motion_config(&arm_motion_path) {
        Ok(cfg) => cfg,
        Err(e) => {
            eprintln!("{e}");
            std::process::exit(1);
        }
    };

    // ── 第三步：解析子命令 ────────────────────────────────────────
    let command = match parse_subcommand(&mut args) {
        Ok(cmd) => cmd,
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            std::process::exit(2);
        }
    };

    // ── 第四步：分派执行 ──────────────────────────────────────────
    match command {
        Command::Hunt(cli) => {
            install_signal_handlers();
            run_hunt(cli, robot_config, arm_motion);
        }
        Command::Serve => {
            run_serve(robot_config, arm_motion);
        }
        Command::Detect(cli) => run_detect(cli, &robot_config),
        Command::Capture(cli) => run_capture(cli, &robot_config),
        Command::ArmTest(cli) => run_armtest(cli, &robot_config, arm_motion),
    }
}

// ── 子命令解析 ──────────────────────────────────────────────────────────────

fn parse_subcommand(
    args: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
) -> Result<Command, String> {
    match args.peek().map(String::as_str) {
        Some("serve") => {
            args.next();
            // serve 不接受额外子命令参数（配置全从 TOML 读）
            // 但允许 serve 之后跟 --help
            while let Some(arg) = args.next() {
                if arg == "-h" || arg == "--help" {
                    print_serve_usage();
                    std::process::exit(0);
                }
                eprintln!("[serve] ignoring extra argument: {arg} (all settings come from robot.toml)");
            }
            Ok(Command::Serve)
        }
        Some("detect") => {
            args.next();
            parse_detect_args(args).map(Command::Detect)
        }
        Some("capture") => {
            args.next();
            parse_capture_args(args).map(Command::Capture)
        }
        Some("armtest") => {
            args.next();
            parse_armtest_args(args).map(Command::ArmTest)
        }
        Some("hunt") => {
            args.next();
            parse_hunt_args(args).map(Command::Hunt)
        }
        _ => parse_hunt_args(args).map(Command::Hunt),
    }
}

fn parse_hunt_args(
    args: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
) -> Result<HuntCli, String> {
    let mut cli = HuntCli::default();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--frames" => {
                cli.frames = Some(
                    args.next()
                        .ok_or_else(|| "--frames expects an integer".to_string())?
                        .parse()
                        .map_err(|_| "--frames expects an integer".to_string())?,
                );
            }
            value if value.starts_with('-') => return Err(format!("unknown option: {value}")),
            value => {
                if cli.model.as_os_str().is_empty() {
                    cli.model = PathBuf::from(value);
                } else {
                    return Err(format!("unexpected positional argument: {value}"));
                }
            }
        }
    }
    if cli.model.as_os_str().is_empty() {
        return Err("missing cvimodel path".to_string());
    }
    Ok(cli)
}

fn parse_detect_args(
    args: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
) -> Result<DetectCli, String> {
    let mut cli = DetectCli::default();
    let mut positional = 0;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_detect_usage();
                std::process::exit(0);
            }
            "-o" | "--out" => {
                cli.output = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out expects a path".to_string())?,
                );
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown detect option: {value}"))
            }
            value => {
                match positional {
                    0 => cli.model = PathBuf::from(value),
                    1 => cli.input = PathBuf::from(value),
                    _ => return Err(format!("unexpected detect argument: {value}")),
                }
                positional += 1;
            }
        }
    }
    if cli.model.as_os_str().is_empty() {
        return Err("missing cvimodel path".to_string());
    }
    if cli.input.as_os_str().is_empty() {
        return Err("missing input image path".to_string());
    }
    Ok(cli)
}

fn parse_capture_args(
    args: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
) -> Result<CaptureCli, String> {
    let mut cli = CaptureCli::default();
    let mut positional = 0;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_capture_usage();
                std::process::exit(0);
            }
            "-o" | "--out" => {
                cli.output = PathBuf::from(
                    args.next()
                        .ok_or_else(|| "--out expects a path".to_string())?,
                );
            }
            "--warmup" => {
                cli.warmup = args
                    .next()
                    .ok_or_else(|| "--warmup expects an integer".to_string())?
                    .parse()
                    .map_err(|_| "--warmup expects a non-negative integer".to_string())?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown capture option: {value}"))
            }
            value => {
                if positional == 0 {
                    cli.output = PathBuf::from(value);
                } else {
                    return Err(format!("unexpected capture argument: {value}"));
                }
                positional += 1;
            }
        }
    }
    Ok(cli)
}

fn parse_armtest_args(
    args: &mut std::iter::Peekable<std::vec::IntoIter<String>>,
) -> Result<ArmTestCli, String> {
    let mut cli = ArmTestCli::default();
    let mut positional: Vec<f32> = Vec::with_capacity(3);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_armtest_usage();
                std::process::exit(0);
            }
            "--time" => {
                cli.time_ms = args
                    .next()
                    .ok_or_else(|| "--time expects an integer".to_string())?
                    .parse()
                    .map_err(|_| "--time expects an integer (ms)".to_string())?;
            }
            value if value.starts_with('-') => {
                return Err(format!("unknown armtest option: {value}"))
            }
            value => {
                if positional.len() >= 3 {
                    return Err(format!("unexpected armtest argument: {value}"));
                }
                let angle = value
                    .parse::<f32>()
                    .map_err(|_| format!("invalid angle value: {value}"))?;
                positional.push(angle);
            }
        }
    }
    if positional.len() != 3 {
        return Err(
            "armtest expects exactly 3 angle arguments: <servo0> <servo1> <servo2>".to_string(),
        );
    }
    cli.servo0 = positional[0];
    cli.servo1 = positional[1];
    cli.servo2 = positional[2];
    Ok(cli)
}

// ── 子命令实现 ──────────────────────────────────────────────────────────────

fn run_hunt(cli: HuntCli, robot_config: RobotConfigFile, arm_motion: ArmMotionConfig) {
    // 摄像头
    let camera_device = &robot_config.camera.device;
    let camera = match UsbCamera::open(
        camera_device,
        robot_config.camera.frame_width,
        robot_config.camera.frame_height,
    ) {
        Ok(camera) => {
            let info = camera.info();
            eprintln!(
                "[camera] opened {}: {}x{} format={}",
                camera_device,
                info.width,
                info.height,
                info.format_name()
            );
            camera
        }
        Err(err) => {
            eprintln!("[camera] failed to open {}: {err}", camera_device);
            std::process::exit(1);
        }
    };

    // TPU 模型
    let model = match open_model(&cli.model) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("[tpu] failed to open model {}: {err}", cli.model.display());
            std::process::exit(1);
        }
    };
    eprintln!(
        "[dbg] model opened, opening motor {} ...",
        robot_config.motor.port
    );

    // 电机（配置全从 TOML 读）
    let motor_config = robot_config.to_motor_config();
    let motor = match Motor::open(&motor_config) {
        Ok(motor) => motor,
        Err(err) => {
            eprintln!("[motor] failed to open {}: {err}", motor_config.device);
            std::process::exit(1);
        }
    };
    eprintln!(
        "[dbg] motor opened, opening arm {} ...",
        robot_config.arm.port
    );

    // 机械臂
    let arm = match Arm::open(&robot_config.arm.port, robot_config.arm.baudrate, arm_motion) {
        Ok(arm) => arm,
        Err(err) => {
            eprintln!(
                "[arm] failed to open {}: {err}",
                robot_config.arm.port
            );
            std::process::exit(1);
        }
    };
    eprintln!("[dbg] arm opened, entering tennis hunter ...");

    let config = RobotConfig {
        inference: robot_config.to_inference_config(),
        max_frames: cli.frames,
    };

    run_tennis_hunter(camera, model, motor, arm, config);
}

fn run_serve(robot_config: RobotConfigFile, _arm_motion: ArmMotionConfig) {
    let web_cfg = WebConfig {
        listen: robot_config.web.listen,
        motor_device: robot_config.motor.port.clone(),
        arm_device: robot_config.arm.port.clone(),
        mock: robot_config.web.mock,
    };

    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .expect("failed to build tokio runtime");

    // 把 arm_motion 也注册到 web.rs 的 hardware 里
    // 目前 web.rs 里 Arm::open 用的是 default()，让我们直接用正确的
    // 这里 serve() 的内部实现会调 Arm::open，我们暂时不改 web.rs
    // 后续可以让 serve 接受 ArmMotionConfig 参数

    if let Err(err) = runtime.block_on(serve(web_cfg)) {
        eprintln!("[web] server failed: {err}");
        std::process::exit(1);
    }
}

fn run_detect(cli: DetectCli, robot_config: &RobotConfigFile) {
    let image = match std::fs::read(&cli.input) {
        Ok(bytes) => bytes,
        Err(err) => {
            eprintln!("[detect] failed to read {}: {err}", cli.input.display());
            std::process::exit(1);
        }
    };

    let mut model = match open_model(&cli.model) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("[tpu] failed to open model {}: {err}", cli.model.display());
            std::process::exit(1);
        }
    };

    let config = robot_config.to_inference_config();

    match model.detect_image(&image, &cli.output, config) {
        Ok(detections) => {
            eprintln!(
                "[detect] {} detection(s) on {}",
                detections.len(),
                cli.input.display()
            );
            for (i, d) in detections.iter().enumerate() {
                eprintln!(
                    "  #{i} class={} score={:.3} box=(cx={:.1}, cy={:.1}, w={:.1}, h={:.1})",
                    d.cls, d.score, d.bbox.x, d.bbox.y, d.bbox.w, d.bbox.h
                );
            }
            eprintln!(
                "[detect] annotated image written to {}",
                cli.output.display()
            );
        }
        Err(err) => {
            eprintln!("[detect] inference failed: {err}");
            std::process::exit(1);
        }
    }
}

fn run_capture(cli: CaptureCli, robot_config: &RobotConfigFile) {
    let camera_device = &robot_config.camera.device;
    let mut camera = match UsbCamera::open(
        camera_device,
        robot_config.camera.frame_width,
        robot_config.camera.frame_height,
    ) {
        Ok(camera) => {
            let info = camera.info();
            eprintln!(
                "[capture] opened {}: {}x{} format={}",
                camera_device,
                info.width,
                info.height,
                info.format_name()
            );
            camera
        }
        Err(err) => {
            eprintln!("[capture] failed to open {}: {err}", camera_device);
            std::process::exit(1);
        }
    };

    for i in 0..cli.warmup {
        if let Err(err) = camera.get_frame() {
            eprintln!("[capture] warm-up frame {i} failed: {err}");
            std::process::exit(1);
        }
    }
    if cli.warmup > 0 {
        eprintln!("[capture] discarded {} warm-up frame(s)", cli.warmup);
    }

    let frame = match camera.get_frame() {
        Ok(frame) => frame,
        Err(err) => {
            eprintln!("[capture] failed to capture frame: {err}");
            std::process::exit(1);
        }
    };

    if let Err(err) = std::fs::write(&cli.output, &frame.pixels) {
        eprintln!("[capture] failed to write {}: {err}", cli.output.display());
        std::process::exit(1);
    }

    eprintln!(
        "[capture] wrote {}x{} frame ({} bytes) to {}",
        frame.width,
        frame.height,
        frame.pixels.len(),
        cli.output.display()
    );
}

fn run_armtest(
    cli: ArmTestCli,
    robot_config: &RobotConfigFile,
    arm_motion: ArmMotionConfig,
) {
    let arm_port = &robot_config.arm.port;
    let mut arm = match Arm::open(arm_port, robot_config.arm.baudrate, arm_motion) {
        Ok(arm) => arm,
        Err(err) => {
            eprintln!("[armtest] failed to open {}: {err}", arm_port);
            std::process::exit(1);
        }
    };

    let time_ms = cli.time_ms.clamp(0, 10_000);
    arm.set_angle(0, cli.servo0, time_ms);
    arm.set_angle(1, cli.servo1, time_ms);
    arm.set_angle(2, cli.servo2, time_ms);

    eprintln!(
        "[armtest] sent: servo0={:.1} servo1={:.1} servo2={:.1} time_ms={}",
        cli.servo0, cli.servo1, cli.servo2, time_ms
    );
}

// ── Usage ────────────────────────────────────────────────────────────────────

fn print_usage() {
    eprintln!(
        "Usage:
  akars [--robot PATH] [--arm PATH] hunt <model.cvimodel> [--frames N]
  akars [--robot PATH] [--arm PATH] serve
  akars [--robot PATH] [--arm PATH] detect <model.cvimodel> <image> [--out PATH]
  akars [--robot PATH] [--arm PATH] capture [output] [--out PATH] [--warmup N]
  akars [--robot PATH] [--arm PATH] armtest <servo0> <servo1> <servo2> [--time MS]

Global options (before the subcommand):
  --robot PATH   Robot TOML config (default: ./robot.toml)
  --arm PATH     Arm motion JSON config (default: ./arm.json)

Config file templates:
  robot.example.toml   camera/motor/arm/tpu/web sections
  arm.example.json     grab_position / lift_position / gripper_open / gripper_close"
    );
}

fn print_capture_usage() {
    eprintln!(
        "Usage: akars capture [output] [--out PATH] [--warmup N]

Defaults:
  --out    capture.jpg
  --warmup 30"
    );
}

fn print_detect_usage() {
    eprintln!(
        "Usage: akars detect <model.cvimodel> <image> [--out PATH]

Defaults:
  --out    detect_out.jpg"
    );
}

fn print_serve_usage() {
    eprintln!(
        "Usage: akars serve

All server settings (listen address, mock mode) come from the [web] section of
the robot TOML config. Use --robot to point to a different file."
    );
}

fn print_armtest_usage() {
    eprintln!(
        "Usage: akars armtest <servo0> <servo1> <servo2> [--time MS]

Sets three arm servo angles directly for debugging.

Defaults:
  --time  1000"
    );
}
