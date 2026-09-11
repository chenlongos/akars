use crate::arm::{Arm, ArmMotionConfig};
use crate::motor::{Motor, MotorConfig};
use axum::extract::{Query, State};
use axum::response::Html;
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};
use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

#[derive(Clone, Debug)]
pub struct WebConfig {
    pub listen: SocketAddr,
    pub motor_device: String,
    pub arm_device: String,
    pub mock: bool,
}

impl Default for WebConfig {
    fn default() -> Self {
        Self {
            listen: SocketAddr::from(([0, 0, 0, 0], 8080)),
            motor_device: "/dev/ttyS3".to_string(),
            arm_device: "/dev/ttyS2".to_string(),
            mock: false,
        }
    }
}

#[derive(Clone)]
struct AppState {
    config: WebConfig,
    hardware: Arc<Mutex<HardwareState>>,
}

struct HardwareState {
    mock: bool,
    motor: Option<Motor>,
    arm: Option<Arm>,
    motor_error: Option<String>,
    arm_error: Option<String>,
    left_speed: i32,
    right_speed: i32,
    last_action: String,
}

#[derive(Clone, Serialize)]
struct StatusSnapshot {
    mock: bool,
    motor_connected: bool,
    arm_connected: bool,
    motor_error: Option<String>,
    arm_error: Option<String>,
    left_speed: i32,
    right_speed: i32,
    last_action: String,
}

#[derive(Serialize)]
struct StatusResponse {
    ok: bool,
    status: StatusSnapshot,
}

#[derive(Serialize)]
struct CommandResponse {
    ok: bool,
    message: String,
    status: StatusSnapshot,
}

#[derive(Deserialize)]
struct DriveRequest {
    action: Option<String>,
    speed: Option<i32>,
    left: Option<i32>,
    right: Option<i32>,
}

#[derive(Deserialize)]
struct ControlQuery {
    action: String,
    speed: Option<i32>,
}

#[derive(Deserialize)]
struct ArmActionRequest {
    action: String,
    servo_id: Option<i32>,
    angle: Option<f32>,
    time_ms: Option<i32>,
}

pub async fn serve(config: WebConfig) -> io::Result<()> {
    let hardware = HardwareState::open(&config);
    let app_state = AppState {
        config: config.clone(),
        hardware: Arc::new(Mutex::new(hardware)),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/api/status", get(status))
        .route("/api/control", get(control))
        .route("/api/drive", post(drive))
        .route("/api/arm/action", post(arm_action))
        .route("/api/reconnect", post(reconnect))
        .with_state(app_state);

    eprintln!("[web] listening on http://{}", config.listen);
    let listener = tokio::net::TcpListener::bind(config.listen).await?;
    axum::serve(listener, app)
        .await
        .map_err(|err| io::Error::new(io::ErrorKind::Other, err))
}

async fn index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn status(State(state): State<AppState>) -> Json<StatusResponse> {
    let hardware = state.hardware.lock().expect("hardware mutex poisoned");
    Json(status_response(hardware.snapshot()))
}

async fn control(
    State(state): State<AppState>,
    Query(query): Query<ControlQuery>,
) -> Json<CommandResponse> {
    let request = DriveRequest {
        action: Some(query.action),
        speed: query.speed,
        left: None,
        right: None,
    };
    drive_inner(state, request)
}

async fn drive(
    State(state): State<AppState>,
    Json(request): Json<DriveRequest>,
) -> Json<CommandResponse> {
    drive_inner(state, request)
}

async fn arm_action(
    State(state): State<AppState>,
    Json(request): Json<ArmActionRequest>,
) -> Json<CommandResponse> {
    let mut hardware = state.hardware.lock().expect("hardware mutex poisoned");
    let result = hardware.apply_arm_action(request);
    Json(command_response(result, hardware.snapshot()))
}

async fn reconnect(State(state): State<AppState>) -> Json<CommandResponse> {
    let mut hardware = state.hardware.lock().expect("hardware mutex poisoned");
    hardware.connect(&state.config);
    let message = if hardware.ready() {
        Ok("hardware connected".to_string())
    } else {
        Err("hardware connection is incomplete".to_string())
    };
    Json(command_response(message, hardware.snapshot()))
}

fn drive_inner(state: AppState, request: DriveRequest) -> Json<CommandResponse> {
    let mut hardware = state.hardware.lock().expect("hardware mutex poisoned");
    let result = hardware.apply_drive(request);
    Json(command_response(result, hardware.snapshot()))
}

fn status_response(status: StatusSnapshot) -> StatusResponse {
    StatusResponse {
        ok: status.mock || (status.motor_connected && status.arm_connected),
        status,
    }
}

fn command_response(result: Result<String, String>, status: StatusSnapshot) -> CommandResponse {
    match result {
        Ok(message) => CommandResponse {
            ok: true,
            message,
            status,
        },
        Err(message) => CommandResponse {
            ok: false,
            message,
            status,
        },
    }
}

impl HardwareState {
    fn open(config: &WebConfig) -> Self {
        let mut state = Self {
            mock: config.mock,
            motor: None,
            arm: None,
            motor_error: None,
            arm_error: None,
            left_speed: 0,
            right_speed: 0,
            last_action: "boot".to_string(),
        };
        state.connect(config);
        state
    }

    fn connect(&mut self, config: &WebConfig) {
        self.mock = config.mock;
        self.left_speed = 0;
        self.right_speed = 0;
        self.motor = None;
        self.arm = None;
        self.motor_error = None;
        self.arm_error = None;

        if self.mock {
            self.last_action = "mock hardware ready".to_string();
            return;
        }

        let motor_config = MotorConfig {
            device: config.motor_device.clone(),
            ..MotorConfig::default()
        };
        match Motor::open(&motor_config) {
            Ok(motor) => self.motor = Some(motor),
            Err(err) => self.motor_error = Some(format!("{}: {err}", config.motor_device)),
        }

        match Arm::open(&config.arm_device, 115200, ArmMotionConfig::default()) {
            Ok(arm) => self.arm = Some(arm),
            Err(err) => self.arm_error = Some(format!("{}: {err}", config.arm_device)),
        }

        self.last_action = "connect".to_string();
    }

    fn ready(&self) -> bool {
        self.mock || (self.motor.is_some() && self.arm.is_some())
    }

    fn snapshot(&self) -> StatusSnapshot {
        StatusSnapshot {
            mock: self.mock,
            motor_connected: self.motor.is_some(),
            arm_connected: self.arm.is_some(),
            motor_error: self.motor_error.clone(),
            arm_error: self.arm_error.clone(),
            left_speed: self.left_speed,
            right_speed: self.right_speed,
            last_action: self.last_action.clone(),
        }
    }

    fn apply_drive(&mut self, request: DriveRequest) -> Result<String, String> {
        if let (Some(left), Some(right)) = (request.left, request.right) {
            let left = left.clamp(-100, 100);
            let right = right.clamp(-100, 100);
            self.drive(left, right)?;
            self.last_action = "drive".to_string();
            return Ok(format!("drive left={left} right={right}"));
        }

        let action = request.action.as_deref().unwrap_or("stop");
        let speed = request.speed.unwrap_or(50).abs().clamp(0, 100);
        match normalize_drive_action(action) {
            "forward" => {
                self.drive(speed, speed)?;
                self.last_action = "forward".to_string();
                Ok(format!("forward speed={speed}"))
            }
            "backward" => {
                self.drive(-speed, -speed)?;
                self.last_action = "backward".to_string();
                Ok(format!("backward speed={speed}"))
            }
            "left" => {
                self.drive(-speed, speed)?;
                self.last_action = "left".to_string();
                Ok(format!("left speed={speed}"))
            }
            "right" => {
                self.drive(speed, -speed)?;
                self.last_action = "right".to_string();
                Ok(format!("right speed={speed}"))
            }
            "brake" => {
                self.brake()?;
                self.last_action = "brake".to_string();
                Ok("brake".to_string())
            }
            "stop" => {
                self.stop()?;
                self.last_action = "stop".to_string();
                Ok("stop".to_string())
            }
            value => Err(format!("unknown drive action: {value}")),
        }
    }

    fn drive(&mut self, left: i32, right: i32) -> Result<(), String> {
        if let Some(motor) = &mut self.motor {
            motor.drive(left, right);
        } else if !self.mock {
            return Err("motor is not connected".to_string());
        }
        self.left_speed = left;
        self.right_speed = right;
        Ok(())
    }

    fn stop(&mut self) -> Result<(), String> {
        if let Some(motor) = &mut self.motor {
            motor.standby();
        } else if !self.mock {
            return Err("motor is not connected".to_string());
        }
        self.left_speed = 0;
        self.right_speed = 0;
        Ok(())
    }

    fn brake(&mut self) -> Result<(), String> {
        if let Some(motor) = &mut self.motor {
            motor.brake();
        } else if !self.mock {
            return Err("motor is not connected".to_string());
        }
        self.left_speed = 0;
        self.right_speed = 0;
        Ok(())
    }

    fn apply_arm_action(&mut self, request: ArmActionRequest) -> Result<String, String> {
        let action = request.action.trim();
        match action {
            "grab" => {
                self.with_arm(|arm| arm.grab())?;
                self.last_action = "arm grab".to_string();
                Ok("arm grab".to_string())
            }
            "release" => {
                self.with_arm(|arm| arm.release())?;
                self.last_action = "arm release".to_string();
                Ok("arm release".to_string())
            }
            "grab_pos" | "ready" => {
                self.with_arm(|arm| arm.grab_pos())?;
                self.last_action = "arm ready".to_string();
                Ok("arm ready".to_string())
            }
            "release_pos" => {
                self.with_arm(|arm| arm.release_pos())?;
                self.last_action = "arm release_pos".to_string();
                Ok("arm release_pos".to_string())
            }
            "show" => {
                self.with_arm(|arm| arm.show())?;
                self.last_action = "arm show".to_string();
                Ok("arm show".to_string())
            }
            "angle" | "set_angle" => {
                let servo_id = request
                    .servo_id
                    .ok_or_else(|| "servo_id is required".to_string())?;
                let angle = request
                    .angle
                    .ok_or_else(|| "angle is required".to_string())?;
                let time_ms = request.time_ms.unwrap_or(500).clamp(0, 10_000);
                self.with_arm(|arm| arm.set_angle(servo_id, angle, time_ms))?;
                self.last_action = format!("arm servo {servo_id} angle {angle:.1}");
                Ok(self.last_action.clone())
            }
            "torque_off" => {
                let servo_id = request
                    .servo_id
                    .ok_or_else(|| "servo_id is required".to_string())?;
                self.with_arm(|arm| arm.release_torque(servo_id))?;
                self.last_action = format!("arm servo {servo_id} torque off");
                Ok(self.last_action.clone())
            }
            "torque_on" => {
                let servo_id = request
                    .servo_id
                    .ok_or_else(|| "servo_id is required".to_string())?;
                self.with_arm(|arm| arm.restore_torque(servo_id))?;
                self.last_action = format!("arm servo {servo_id} torque on");
                Ok(self.last_action.clone())
            }
            value => Err(format!("unknown arm action: {value}")),
        }
    }

    fn with_arm(&mut self, command: impl FnOnce(&mut Arm)) -> Result<(), String> {
        if let Some(arm) = &mut self.arm {
            command(arm);
        } else if !self.mock {
            return Err("arm is not connected".to_string());
        }
        Ok(())
    }
}

fn normalize_drive_action(action: &str) -> &str {
    match action.trim() {
        "up" | "forward" => "forward",
        "down" | "backward" | "reverse" => "backward",
        "left" => "left",
        "right" => "right",
        "brake" => "brake",
        "stop" | "standby" => "stop",
        other => other,
    }
}

const INDEX_HTML: &str = r#"<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
  <title>AKA 控制台</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #101318;
      --panel: #171b22;
      --panel-2: #1e252f;
      --line: #313946;
      --text: #eef2f7;
      --muted: #a6b0bf;
      --accent: #2f7dd1;
      --ok: #2ab673;
      --bad: #e05252;
      --warn: #d49b2a;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100dvh;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      height: 56px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      padding: 0 16px;
      border-bottom: 1px solid var(--line);
      background: #151920;
      position: sticky;
      top: 0;
      z-index: 2;
    }
    h1 {
      margin: 0;
      font-size: 18px;
      font-weight: 750;
      letter-spacing: 0;
      white-space: nowrap;
    }
    main {
      width: min(1120px, 100%);
      margin: 0 auto;
      padding: 16px;
      display: grid;
      grid-template-columns: 1.15fr .85fr;
      gap: 14px;
    }
    section {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 14px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 15px;
      font-weight: 720;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      gap: 10px;
      align-items: center;
      color: var(--muted);
      font-size: 13px;
      min-width: 0;
    }
    .dot {
      width: 10px;
      height: 10px;
      border-radius: 50%;
      display: inline-block;
      background: var(--bad);
      flex: 0 0 auto;
    }
    .dot.ok { background: var(--ok); }
    .dot.warn { background: var(--warn); }
    .readout {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
      margin-bottom: 14px;
    }
    .metric {
      min-height: 70px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel-2);
      padding: 10px;
      display: grid;
      align-content: center;
      gap: 4px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
    }
    .metric b {
      font-size: 18px;
      overflow-wrap: anywhere;
    }
    .drive-pad {
      width: min(390px, 100%);
      margin: 0 auto;
      display: grid;
      grid-template-columns: repeat(3, minmax(72px, 1fr));
      grid-template-rows: repeat(3, 74px);
      gap: 10px;
    }
    button {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #222a35;
      color: var(--text);
      min-height: 42px;
      font: inherit;
      font-weight: 680;
      cursor: pointer;
      touch-action: manipulation;
    }
    button:active { transform: translateY(1px); }
    button.primary { background: var(--accent); border-color: #5b9ee5; }
    button.stop { background: #8f2c2c; border-color: #cd5555; }
    button.secondary { background: #27303c; }
    .up { grid-column: 2; grid-row: 1; }
    .left { grid-column: 1; grid-row: 2; }
    .stop { grid-column: 2; grid-row: 2; }
    .right { grid-column: 3; grid-row: 2; }
    .down { grid-column: 2; grid-row: 3; }
    .stack { display: grid; gap: 14px; }
    .row {
      display: grid;
      grid-template-columns: minmax(90px, 140px) 1fr;
      gap: 10px;
      align-items: center;
      padding: 8px 0;
      border-bottom: 1px solid rgba(166, 176, 191, .14);
    }
    .row:last-child { border-bottom: 0; }
    label {
      color: var(--muted);
      font-size: 13px;
    }
    input, select {
      min-width: 0;
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #0f1319;
      color: var(--text);
      padding: 10px;
      font: inherit;
    }
    input[type="range"] { padding: 0; }
    .inline {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .actions {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 8px;
    }
    .message {
      min-height: 22px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    @media (max-width: 840px) {
      main { grid-template-columns: 1fr; padding: 10px; }
      .readout { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
    @media (max-width: 460px) {
      header { padding: 0 10px; }
      .drive-pad { grid-template-rows: repeat(3, 68px); gap: 8px; }
      .actions, .inline { grid-template-columns: 1fr; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>AKA 控制台</h1>
    <div class="status"><span id="dot" class="dot"></span><span id="top-status">连接中</span></div>
  </header>

  <main>
    <section>
      <h2>底盘</h2>
      <div class="readout">
        <div class="metric"><span>左轮</span><b id="left-speed">0</b></div>
        <div class="metric"><span>右轮</span><b id="right-speed">0</b></div>
        <div class="metric"><span>模式</span><b id="mode">...</b></div>
        <div class="metric"><span>动作</span><b id="last-action">...</b></div>
      </div>
      <div class="drive-pad">
        <button class="up primary" data-action="forward">▲</button>
        <button class="left primary" data-action="left">◀</button>
        <button class="stop" data-action="stop">■</button>
        <button class="right primary" data-action="right">▶</button>
        <button class="down primary" data-action="backward">▼</button>
      </div>
      <div class="row">
        <label for="speed">速度</label>
        <input id="speed" type="range" min="0" max="100" value="50">
      </div>
      <div class="row">
        <label>直控</label>
        <div class="inline">
          <input id="manual-left" type="number" min="-100" max="100" value="0">
          <input id="manual-right" type="number" min="-100" max="100" value="0">
          <button onclick="manualDrive()">发送</button>
        </div>
      </div>
      <div class="message" id="drive-message"></div>
    </section>

    <div class="stack">
      <section>
        <h2>机械臂</h2>
        <div class="actions">
          <button onclick="arm('grab_pos')">预备</button>
          <button onclick="arm('grab')">抓取</button>
          <button onclick="arm('release')">释放</button>
          <button onclick="arm('release_pos')">释放位</button>
          <button onclick="arm('show')">展示</button>
          <button onclick="armAngle()">设角</button>
        </div>
        <div class="row">
          <label for="servo-id">舵机</label>
          <select id="servo-id">
            <option value="0">0</option>
            <option value="1">1</option>
            <option value="2">2</option>
          </select>
        </div>
        <div class="row">
          <label for="angle">角度</label>
          <input id="angle" type="number" min="0" max="270" value="90">
        </div>
        <div class="row">
          <label for="time-ms">时间</label>
          <input id="time-ms" type="number" min="0" max="10000" value="500">
        </div>
        <div class="actions">
          <button onclick="armTorque('torque_on')">上电</button>
          <button onclick="armTorque('torque_off')">卸力</button>
          <button onclick="reconnect()">重连</button>
        </div>
        <div class="message" id="arm-message"></div>
      </section>

      <section>
        <h2>硬件</h2>
        <div class="row"><label>电机</label><div id="motor-status">...</div></div>
        <div class="row"><label>机械臂</label><div id="arm-status">...</div></div>
        <div class="message" id="hardware-message"></div>
      </section>
    </div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const speed = () => Number($("speed").value);

    async function post(url, body) {
      const response = await fetch(url, {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify(body || {})
      });
      return response.json();
    }

    async function sendDrive(action) {
      const data = await post("/api/drive", { action, speed: speed() });
      $("drive-message").textContent = data.message;
      renderStatus(data.status);
    }

    async function manualDrive() {
      const left = Number($("manual-left").value);
      const right = Number($("manual-right").value);
      const data = await post("/api/drive", { left, right });
      $("drive-message").textContent = data.message;
      renderStatus(data.status);
    }

    async function arm(action) {
      const data = await post("/api/arm/action", { action });
      $("arm-message").textContent = data.message;
      renderStatus(data.status);
    }

    async function armAngle() {
      const data = await post("/api/arm/action", {
        action: "set_angle",
        servo_id: Number($("servo-id").value),
        angle: Number($("angle").value),
        time_ms: Number($("time-ms").value)
      });
      $("arm-message").textContent = data.message;
      renderStatus(data.status);
    }

    async function armTorque(action) {
      const data = await post("/api/arm/action", {
        action,
        servo_id: Number($("servo-id").value)
      });
      $("arm-message").textContent = data.message;
      renderStatus(data.status);
    }

    async function reconnect() {
      const data = await post("/api/reconnect", {});
      $("hardware-message").textContent = data.message;
      renderStatus(data.status);
    }

    async function refresh() {
      try {
        const response = await fetch("/api/status");
        const data = await response.json();
        renderStatus(data.status);
      } catch {
        $("top-status").textContent = "离线";
        $("dot").className = "dot";
      }
    }

    function renderStatus(status) {
      const healthy = status.mock || (status.motor_connected && status.arm_connected);
      $("dot").className = "dot " + (healthy ? "ok" : "warn");
      $("top-status").textContent = healthy ? "在线" : "硬件未就绪";
      $("left-speed").textContent = status.left_speed;
      $("right-speed").textContent = status.right_speed;
      $("mode").textContent = status.mock ? "mock" : "real";
      $("last-action").textContent = status.last_action;
      $("motor-status").textContent = status.motor_connected ? "已连接" : (status.motor_error || "未连接");
      $("arm-status").textContent = status.arm_connected ? "已连接" : (status.arm_error || "未连接");
    }

    document.querySelectorAll("[data-action]").forEach((button) => {
      const action = button.dataset.action;
      const down = () => sendDrive(action);
      const up = () => {
        if (action !== "stop") sendDrive("stop");
      };
      button.addEventListener("pointerdown", down);
      button.addEventListener("pointerup", up);
      button.addEventListener("pointercancel", up);
      button.addEventListener("pointerleave", up);
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>"#;
