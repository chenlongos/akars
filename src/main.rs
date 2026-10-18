use akars::arm::Arm;
use akars::camera::UsbCamera;
use akars::motor::{Motor, MotorConfig};
use akars::robot::{install_signal_handlers, run_tennis_hunter, RobotConfig};
use akars::tpu::{open_model, InferenceConfig};
use akars::web::{serve, WebConfig};
use std::env;
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Debug)]
struct Cli {
    model: PathBuf,
    camera: String,
    motor: String,
    arm: String,
    frames: Option<u64>,
    classes: i32,
    conf: f32,
    iou: f32,
}

/// Arguments for the standalone TPU inference test: run a model on a single
/// image file and write an annotated result image.
#[derive(Debug)]
struct DetectCli {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
    classes: i32,
    conf: f32,
    iou: f32,
}

/// Arguments for the camera capture test: grab a single frame and save it,
/// discarding the first few frames so the sensor's exposure/white-balance can
/// settle (warm up).
#[derive(Debug)]
struct CaptureCli {
    camera: String,
    output: PathBuf,
    warmup: u32,
    /// `--bucket`:额外跑一遍桶色检测,用来对着真桶标定颜色/距离,不用开整车。
    bucket: bool,
}

/// Arguments for manual arm debugging: set all three servo angles directly.
#[derive(Debug)]
struct ArmTestCli {
    arm: String,
    servo0: f32,
    servo1: f32,
    servo2: f32,
    time_ms: i32,
}

#[derive(Debug)]
enum Command {
    Hunt(Cli),
    Serve(WebConfig),
    Detect(DetectCli),
    Capture(CaptureCli),
    ArmTest(ArmTestCli),
}

impl Default for Cli {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            camera: "/dev/cvi-usb-camera0".to_string(),
            motor: "/dev/ttyS3".to_string(),
            arm: "/dev/ttyS2".to_string(),
            frames: None,
            classes: 1,
            conf: 0.5,
            iou: 0.5,
        }
    }
}

impl Default for DetectCli {
    fn default() -> Self {
        Self {
            model: PathBuf::new(),
            input: PathBuf::new(),
            output: PathBuf::from("detect_out.jpg"),
            classes: 1,
            conf: 0.5,
            iou: 0.5,
        }
    }
}

impl Default for CaptureCli {
    fn default() -> Self {
        Self {
            camera: "/dev/cvi-usb-camera0".to_string(),
            output: PathBuf::from("capture.jpg"),
            warmup: 30,
            bucket: false,
        }
    }
}

impl Default for ArmTestCli {
    fn default() -> Self {
        Self {
            arm: "/dev/ttyS2".to_string(),
            servo0: 0.0,
            servo1: 0.0,
            servo2: 0.0,
            time_ms: 1000,
        }
    }
}

fn main() {
    let command = match parse_cli(env::args().skip(1)) {
        Ok(command) => command,
        Err(message) => {
            eprintln!("{message}");
            print_usage();
            std::process::exit(2);
        }
    };

    match command {
        Command::Hunt(cli) => {
            install_signal_handlers();
            run_hunt(cli);
        }
        Command::Serve(config) => {
            let runtime = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()
                .expect("failed to build tokio runtime");
            if let Err(err) = runtime.block_on(serve(config)) {
                eprintln!("[web] server failed: {err}");
                std::process::exit(1);
            }
        }
        Command::Detect(cli) => run_detect(cli),
        Command::Capture(cli) => run_capture(cli),
        Command::ArmTest(cli) => run_armtest(cli),
    }
}

fn run_armtest(cli: ArmTestCli) {
    let mut arm = match Arm::open(&cli.arm, 115200) {
        Ok(arm) => arm,
        Err(err) => {
            eprintln!("[armtest] failed to open {}: {err}", cli.arm);
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

fn run_hunt(cli: Cli) {
    let camera = match UsbCamera::open(&cli.camera) {
        Ok(camera) => {
            let info = camera.info();
            eprintln!(
                "[camera] opened {}: {}x{} format={} connected={}",
                cli.camera, info.width, info.height, info.format, info.connected
            );
            camera
        }
        Err(err) => {
            eprintln!("[camera] failed to open {}: {err}", cli.camera);
            std::process::exit(1);
        }
    };

    let model = match open_model(&cli.model) {
        Ok(model) => model,
        Err(err) => {
            eprintln!("[tpu] failed to open model {}: {err}", cli.model.display());
            std::process::exit(1);
        }
    };
    eprintln!("[dbg] model opened, opening motor {} ...", cli.motor);

    let motor_config = MotorConfig {
        device: cli.motor.clone(),
        ..MotorConfig::default()
    };
    let motor = match Motor::open(&motor_config) {
        Ok(motor) => motor,
        Err(err) => {
            eprintln!("[motor] failed to open {}: {err}", cli.motor);
            std::process::exit(1);
        }
    };
    eprintln!("[dbg] motor opened, opening arm {} ...", cli.arm);

    let arm = match Arm::open(&cli.arm, 115200) {
        Ok(arm) => arm,
        Err(err) => {
            eprintln!("[arm] failed to open {}: {err}", cli.arm);
            std::process::exit(1);
        }
    };
    eprintln!("[dbg] arm opened, entering tennis hunter ...");

    // 抓取/放桶序列跨好几秒,帧数上限会在流程中途把它截断,所以这里不再生效。
    // --frames 仍然接受,让已有的调用方(rootfs test_runner.sh、benchmark)不改也能跑。
    if cli.frames.is_some() {
        eprintln!("[dbg] --frames is ignored; the hunter runs until stopped");
    }

    let config = RobotConfig {
        inference: InferenceConfig {
            classes_num: cli.classes,
            confidence_threshold: cli.conf,
            iou_threshold: cli.iou,
        },
    };

    run_tennis_hunter(camera, model, motor, arm, config);
}

fn run_detect(cli: DetectCli) {
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

    let config = InferenceConfig {
        classes_num: cli.classes,
        confidence_threshold: cli.conf,
        iou_threshold: cli.iou,
    };

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

fn run_capture(cli: CaptureCli) {
    let mut camera = match UsbCamera::open(&cli.camera) {
        Ok(camera) => {
            let info = camera.info();
            eprintln!(
                "[capture] opened {}: {}x{} format={} connected={}",
                cli.camera, info.width, info.height, info.format, info.connected
            );
            camera
        }
        Err(err) => {
            eprintln!("[capture] failed to open {}: {err}", cli.camera);
            std::process::exit(1);
        }
    };

    // Discard the first few frames so the sensor's auto exposure / white
    // balance can settle before we keep one.
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

    // ── 桶色检测(--bucket):对着真桶标定颜色/距离,不用开整车 ──
    if cli.bucket {
        let (w, h) = (frame.width as i32, frame.height as i32);
        // 本分支取帧路径固定返回 YUV422 planar。
        let planar = akars::camera::PlanarFormat::I422;
        match akars::bucket::detect_bucket(
            &frame.pixels,
            w,
            h,
            planar,
            akars::bucket::BUCKET_MIN_AREA_FULLRES,
        ) {
            Some(r) => eprintln!(
                "[bucket] found area={:.3} ({:.0}% of frame) box=({},{},{},{}) center=({},{})",
                r.area_ratio,
                r.area_ratio * 100.0,
                r.x,
                r.y,
                r.w,
                r.h,
                r.cx,
                r.cy
            ),
            None => eprintln!(
                "[bucket] not found (min_area={} full-res px)",
                akars::bucket::BUCKET_MIN_AREA_FULLRES
            ),
        }
        // 判不出来时最有用的一条:桶的实际色相落在哪。
        match akars::bucket::bucket_color_stats(&frame.pixels, w, h, planar) {
            Some((hue, sat, val, n)) => eprintln!(
                "[bucket] coloured pixels: mean H={hue} S={sat} V={val} (n={n}, 判据 H∈[{},{}] 且 S≥80 V≥50)",
                akars::bucket::BUCKET_HUE_MIN,
                akars::bucket::BUCKET_HUE_MAX
            ),
            None => eprintln!("[bucket] no coloured pixels in view (S/V 下限全灭)"),
        }
    }

    if let Err(err) = akars::image_bridge::save_yuv422p(
        &frame.pixels,
        frame.width as i32,
        frame.height as i32,
        &cli.output,
    ) {
        eprintln!("[capture] failed to write {}: {err}", cli.output.display());
        std::process::exit(1);
    }

    eprintln!(
        "[capture] wrote {}x{} frame to {}",
        frame.width,
        frame.height,
        cli.output.display()
    );
}

fn parse_cli(args: impl Iterator<Item = String>) -> Result<Command, String> {
    let mut args = args.peekable();
    match args.peek().map(String::as_str) {
        Some("serve") => {
            args.next();
            parse_serve_cli(args).map(Command::Serve)
        }
        Some("detect") => {
            args.next();
            parse_detect_cli(args).map(Command::Detect)
        }
        Some("capture") => {
            args.next();
            parse_capture_cli(args).map(Command::Capture)
        }
        Some("armtest") => {
            args.next();
            parse_armtest_cli(args).map(Command::ArmTest)
        }
        _ => parse_hunt_cli(args).map(Command::Hunt),
    }
}

fn parse_armtest_cli(args: impl Iterator<Item = String>) -> Result<ArmTestCli, String> {
    let mut cli = ArmTestCli::default();
    let mut args = args.peekable();
    let mut positional: Vec<f32> = Vec::with_capacity(3);

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_armtest_usage();
                std::process::exit(0);
            }
            "--arm" => cli.arm = take_value(&mut args, "--arm")?,
            "--time" => {
                cli.time_ms = take_value(&mut args, "--time")?
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

fn parse_hunt_cli(args: impl Iterator<Item = String>) -> Result<Cli, String> {
    let mut cli = Cli::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_usage();
                std::process::exit(0);
            }
            "--camera" => cli.camera = take_value(&mut args, "--camera")?,
            "--motor" => cli.motor = take_value(&mut args, "--motor")?,
            "--arm" => cli.arm = take_value(&mut args, "--arm")?,
            "--frames" => {
                cli.frames = Some(
                    take_value(&mut args, "--frames")?
                        .parse()
                        .map_err(|_| "--frames expects an integer".to_string())?,
                );
            }
            "--classes" => {
                cli.classes = take_value(&mut args, "--classes")?
                    .parse()
                    .map_err(|_| "--classes expects an integer".to_string())?;
            }
            "--conf" => {
                cli.conf = take_value(&mut args, "--conf")?
                    .parse()
                    .map_err(|_| "--conf expects a float".to_string())?;
            }
            "--iou" => {
                cli.iou = take_value(&mut args, "--iou")?
                    .parse()
                    .map_err(|_| "--iou expects a float".to_string())?;
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

fn parse_serve_cli(args: impl Iterator<Item = String>) -> Result<WebConfig, String> {
    let mut config = WebConfig::default();
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_web_usage();
                std::process::exit(0);
            }
            "--listen" => {
                config.listen = take_value(&mut args, "--listen")?
                    .parse::<SocketAddr>()
                    .map_err(|_| {
                        "--listen expects HOST:PORT, for example 0.0.0.0:8080".to_string()
                    })?;
            }
            "--motor" => config.motor_device = take_value(&mut args, "--motor")?,
            "--arm" => config.arm_device = take_value(&mut args, "--arm")?,
            "--mock" => config.mock = true,
            value if value.starts_with('-') => {
                return Err(format!("unknown serve option: {value}"))
            }
            value => return Err(format!("unexpected serve argument: {value}")),
        }
    }
    Ok(config)
}

fn parse_detect_cli(args: impl Iterator<Item = String>) -> Result<DetectCli, String> {
    let mut cli = DetectCli::default();
    let mut positional = 0;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_detect_usage();
                std::process::exit(0);
            }
            "-o" | "--out" => cli.output = PathBuf::from(take_value(&mut args, "--out")?),
            "--classes" => {
                cli.classes = take_value(&mut args, "--classes")?
                    .parse()
                    .map_err(|_| "--classes expects an integer".to_string())?;
            }
            "--conf" => {
                cli.conf = take_value(&mut args, "--conf")?
                    .parse()
                    .map_err(|_| "--conf expects a float".to_string())?;
            }
            "--iou" => {
                cli.iou = take_value(&mut args, "--iou")?
                    .parse()
                    .map_err(|_| "--iou expects a float".to_string())?;
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

fn parse_capture_cli(args: impl Iterator<Item = String>) -> Result<CaptureCli, String> {
    let mut cli = CaptureCli::default();
    let mut positional = 0;
    let mut args = args.peekable();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-h" | "--help" => {
                print_capture_usage();
                std::process::exit(0);
            }
            "--camera" => cli.camera = take_value(&mut args, "--camera")?,
            "-o" | "--out" => cli.output = PathBuf::from(take_value(&mut args, "--out")?),
            "--warmup" => {
                cli.warmup = take_value(&mut args, "--warmup")?
                    .parse()
                    .map_err(|_| "--warmup expects a non-negative integer".to_string())?;
            }
            "--bucket" => cli.bucket = true,
            value if value.starts_with('-') => {
                return Err(format!("unknown capture option: {value}"))
            }
            value => {
                match positional {
                    0 => cli.output = PathBuf::from(value),
                    _ => return Err(format!("unexpected capture argument: {value}")),
                }
                positional += 1;
            }
        }
    }
    Ok(cli)
}

fn take_value(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    option: &str,
) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{option} expects a value"))
}

fn print_usage() {
    eprintln!(
        "Usage:\n  akars <model.cvimodel> [--camera DEV] [--motor DEV] [--arm DEV] [--frames N] [--classes N] [--conf X] [--iou X]\n  akars serve [--listen HOST:PORT] [--motor DEV] [--arm DEV] [--mock]\n  akars detect <model.cvimodel> <image> [--out PATH] [--classes N] [--conf X] [--iou X]\n  akars capture [output.jpg] [--camera DEV] [--out PATH] [--warmup N]\n  akars armtest <servo0> <servo1> <servo2> [--arm DEV] [--time MS]"
    );
}

fn print_capture_usage() {
    eprintln!(
        "Usage: akars capture [output.jpg] [--camera DEV] [--out PATH] [--warmup N] [--bucket]\n\nGrabs a single JPEG frame from the camera, discarding the first N frames so the\nsensor's auto exposure / white balance can settle (warm up).\n\nWith --bucket it also runs the bucket-colour detector on that frame and reports the\nblob it would chase, the area ratio (放球阈值 0.90 / 制动阈值 0.70), and the mean\nH/S/V of the coloured pixels — so the colour thresholds can be calibrated against\nthe real bucket without driving the car.\n\nDefaults:\n  --camera /dev/cvi-usb-camera0\n  --out capture.jpg\n  --warmup 30"
    );
}

fn print_detect_usage() {
    eprintln!(
        "Usage: akars detect <model.cvimodel> <image> [--out PATH] [--classes N] [--conf X] [--iou X]\n\nRuns the TPU model on a single image and writes a copy with detection boxes drawn.\n\nDefaults:\n  --out detect_out.jpg\n  --classes 1\n  --conf 0.5\n  --iou 0.5"
    );
}

fn print_web_usage() {
    eprintln!(
        "Usage: akars serve [--listen HOST:PORT] [--motor DEV] [--arm DEV] [--mock]\n\nDefaults:\n  --listen 0.0.0.0:8080\n  --motor /dev/ttyS3\n  --arm /dev/ttyS2"
    );
}

fn print_armtest_usage() {
    eprintln!(
        "Usage: akars armtest <servo0> <servo1> <servo2> [--arm DEV] [--time MS]\n\nSets three arm servo angles directly for debugging.\n\nDefaults:\n  --arm /dev/ttyS2\n  --time 1000"
    );
}
