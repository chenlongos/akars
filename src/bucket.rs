//! 彩色桶识别 + 「抓起来 → 找桶 → 放进去」的状态机。**默认目标是蓝色**
//! (`[bucket] bucket_color`,预设表见 `color_preset`)。
//!
//! 移植自 /tmp/aka-rk3588（同类小车的 C++ 实现）：
//!   - 检测器：`test_cmds.cpp:458-704`（`_rgb2hsv` / `_is_red` / `_find_red_boxes`）
//!   - 状态机：`tennis.cpp:329-441`（FIND_BUCKET / APPROACH_BUCKET）+ `:340-361`（DEPOSIT）
//!
//! 四处与参考**故意不同**，原因写在各处的注释里：
//!   1. 检测跑在**色度网格**上（320×240），不是全分辨率 RGB —— 零拷贝路径下用户态
//!      根本没有像素，只有 ioctl 4 拷回来的 I420/I422。颜色本来就只在色度里，
//!      而面积比是尺度无关的，所以 0.90 / 0.70 这些阈值原值就可用。
//!   2. `BUCKET_SEARCH_SPD` 用 24（本仓库 `IDLE_SPEED`）而不是参考的 12 ——
//!      12 映射后 PWM 约 20%，低于 `robot.rs` 注释里实测的"下单但车不动"线。
//!   3. 趋近用**独立的连续比例差速**，不接 `robot.rs` 的 `servo_step` —— 参考就是连续
//!      比例；我们的球道因为连续修正会摆头才改成脉冲+静置（见 `robot.rs:98-128`）。
//!      桶是静止大目标、末端还有 350ms 硬刹托底，所以这里回归连续控制，且完全隔离在
//!      本模块内，不会渗进追球逻辑。但接线极性补偿是共享的：左右方向同样过
//!      `[chassis] turn_reverse`，和追球/丢球搜索/找桶旋转同源，不然轮子通道反接的
//!      底盘在追桶时会左右反向。
//!   4. 舵机角度不搬（两边标定不同），只搬时序。
//!   5. **目标色可配**:参考写死红色,我们用蓝色。颜色判据(色相窗口 + S/V 下限 + 早退
//!      用的色度平面)全部收在 `color_preset` 这张预设表里,配置文件只写颜色名
//!      (`[bucket] bucket_color = "blue"`);其余代码颜色无关,加一种颜色只加一行。

use crate::arm::Arm;
use crate::camera::PlanarFormat;
use crate::config;
use crate::motor::Motor;
use std::time::{Duration, Instant};

// ── 趋近参数的**内置默认值**（与参考同名同值）─────────────────────────────
//
// 下列常量全部可以通过配置文件 `[bucket]` 段覆盖（`config::BucketConfig`）。
// 这里只留默认值，作为单一真值来源：`config.rs` 的默认值直接取这些常量，
// 所以配置文件缺项时行为与硬编码时代完全一致，改参数则不用重编译。

/// 桶面积比 ≥ 此值 → 够近，可以放球。
pub(crate) const BUCKET_AREA_DEPOSIT: f32 = 0.90;
/// 桶面积比 ≥ 此值 → 进入制动区，降到蠕行速度（避免撞桶）。
pub(crate) const BUCKET_AREA_BRAKE: f32 = 0.70;
pub(crate) const BUCKET_APPROACH_SPD: i32 = 25;
/// 制动区（面积 ≥ 0.70）的蠕行速度。
///
/// **不能照抄参考的 5**。参考写 5，但它的 `min_speed` 同样是 15，所以 5 映射后只有
/// 15% PWM —— 而本车实测"推得动整车"的下限是 26%（速度 20，见 `MIN_MOVING_SPEED`）。
/// 5 只有下限的六成，车原地不动；不动则面积不涨，于是永远卡在 0.70~0.89 出不去。
/// 参考那边多半也没真跑进过制动区。
///
/// 取 22：高于下限且留一点余量，又明显低于趋近档 25，两级才还有意义。
pub(crate) const BUCKET_BRAKE_SPD: i32 = 22;
/// 实测能推得动整车的最低速度档。低于它 PWM 过不了静摩擦，车"看着在下令、不动"
/// （来源：`robot.rs` 接近速度分级那段的实测结论，值 20 → 26% PWM）。
/// 这条是**护栏不是参数**：不顺配置文件下发，两级速度都按它钳制（见 `approach_speeds`）。
const MIN_MOVING_SPEED: i32 = 20;

/// 两级趋近速度的加载期归一化。
///
/// 踩过这个坑：参考的制动档 5 映射后只有 15% PWM，车在制动区里根本不动，而面积又不会
/// 再涨，于是永远卡在 0.70~0.89 —— 场上表现为"靠近桶就站住"。
/// 这几个值原本用编译期断言锁死（谁调回去都过不了编译）；参数外置到配置文件之后编译期
/// 看不到取值，所以改成在这里钳制：两档都抬到能推动整车的下限，制动档不超过趋近档。
fn approach_speeds(cfg: &config::BucketConfig) -> (i32, i32) {
    let approach = cfg.approach_speed.clamp(MIN_MOVING_SPEED, 100);
    let brake = cfg.brake_speed.clamp(MIN_MOVING_SPEED, approach);
    (approach, brake)
}

/// 转向增益：bias = K × 偏差 / 半宽，夹在 ±MAX_BIAS。
pub(crate) const BUCKET_K_TURN: f32 = 20.0;
pub(crate) const BUCKET_MAX_BIAS: i32 = 8;
/// 中心死区（全分辨率像素）。注意它比追球的死区紧得多：桶是大目标，不需要亚像素级对准。
pub(crate) const BUCKET_CENTER_DEAD_ZONE: i32 = 15;
/// 找不到桶时原地右转的速度。**故意不用参考的 12**，见文件头注释。
pub(crate) const BUCKET_SEARCH_SPD: i32 = 24;
/// 趋近中连续多少帧看不到桶 → 退回搜索。
pub(crate) const BUCKET_LOST_FRAMES: i32 = 10;
/// 连续多少帧看到桶 → 确认（防抖）。
pub(crate) const BUCKET_CONFIRM_CNT: i32 = 3;
/// 放球前的硬刹时长。参考在停车前整段 `motor.brake()`，我们分帧重发。
pub(crate) const BUCKET_BRAKE_MS: u64 = 350;
/// 松爪后等舵机走完，再让机械臂归位。
pub(crate) const BUCKET_DEPOSIT_MS: u64 = 500;
/// 归位后等一小会儿再回追球，别让底盘在手臂还在摆的时候就起步。
pub(crate) const BUCKET_HOME_MS: u64 = 100;
/// 放完球后的**倒退时长**（ms）。车头此刻几乎贴着桶，不退就直接“原地旋转找球”，
/// 车身一扫就扫到桶壁。
///
/// 这里只求“退开一点”，不要求停准 —— 所以按帧推进的定时相就够用，不走抓球那个
/// 跑满墙钟的后退脉冲（那个是为了退准）。0 = 不退（旧行为）。速度不给新参数，
/// 直接用 `[chassis] backward_speed`：和“抓球时太近先退”是同一个物理动作。
pub(crate) const BUCKET_DEPOSIT_BACKOFF_MS: u64 = 300;
/// 连通域的最小面积（**全分辨率像素数**，色度网格上按采样率折算）。
pub(crate) const BUCKET_MIN_AREA_FULLRES: i32 = 1000;

/// 调参开关：AKARS_BUCKET_LOG=1 时每帧打印桶状态机的决策（默认关）。
fn bucket_log() -> bool {
    std::env::var("AKARS_BUCKET_LOG").is_ok()
}

/// 全局转向极性：`[chassis] turn_reverse` 置 true 时镜像左右，与追球修正/
/// 丢球搜索共用同一份接线补偿。抓取盲转不吃本开关（看 `grab_turn_direction`）。
fn turn_polarity() -> i32 {
    if config::get().chassis.turn_reverse {
        -1
    } else {
        1
    }
}

// ── 检测器 ────────────────────────────────────────────────────────────────

/// 一个目标色连通域的包围盒。**`x, y` 是左上角**（沿用参考 `BucketBox` 的约定）——
/// 不要套用 `detector::Box2d` 的中心约定，两边含义相反。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketBox {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub area: i32,
}

/// 检测结果。坐标已换算回**全分辨率 640×480 坐标系**，这样下游的控制常数
/// （死区 15px、半宽 320、K_TURN）与参考逐字一致，不用重新标定。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BucketResult {
    /// 最大连通域的采样数 / 色度网格面积。与"全分辨率目标色像素占比"等价：
    /// 分子分母同时被采样率除，比值不变。
    pub area_ratio: f32,
    pub cx: i32,
    pub cy: i32,
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
}

/// 色度网格的布局：告诉检测器该去哪里读 U/V、每个采样点覆盖多大的亮度块。
#[derive(Clone, Copy, Debug)]
struct ChromaLayout {
    cw: i32,
    ch: i32,
    /// 每个采样点横向/纵向覆盖的全分辨率像素数（I420 是 2×2，I422 是 2×1）。
    sx: i32,
    sy: i32,
    u_off: usize,
    v_off: usize,
}

impl ChromaLayout {
    fn for_format(planar: PlanarFormat, w: i32, h: i32) -> Self {
        let luma = w as usize * h as usize;
        match planar {
            // Y 满分辨率 + U/V 各 1/4（横纵都减半）。
            PlanarFormat::I420 => Self {
                cw: w / 2,
                ch: h / 2,
                sx: 2,
                sy: 2,
                u_off: luma,
                v_off: luma + luma / 4,
            },
            // Y 满分辨率 + U/V 横向减半、纵向不减。
            PlanarFormat::I422 => Self {
                cw: w / 2,
                ch: h,
                sx: 2,
                sy: 1,
                u_off: luma,
                v_off: luma + luma / 2,
            },
        }
    }

    fn sample_count(&self) -> usize {
        (self.cw * self.ch) as usize
    }

    /// 全分辨率像素数 → 色度网格采样数（向下取整，宁可严一点）。
    fn min_area_chroma(&self, fullres: i32) -> i32 {
        (fullres / (self.sx * self.sy)).max(1)
    }
}

/// 在一帧平面 YUV 里找最大的目标色连通域；找不到返回 `None`。
///
/// `min_area_fullres` 用**全分辨率像素数**表达（与参考的 `min_area=1000` 同量纲），
/// 内部按采样率折算到色度网格。
pub fn detect_bucket(
    frame: &[u8],
    width: i32,
    height: i32,
    planar: PlanarFormat,
    min_area_fullres: i32,
) -> Option<BucketResult> {
    if width <= 0 || height <= 0 {
        return None;
    }
    let layout = ChromaLayout::for_format(planar, width, height);
    let needed = layout.v_off + layout.sample_count();
    if frame.len() < needed {
        return None;
    }

    let mask = build_color_mask(frame, width, &layout);
    let boxes = find_boxes(
        &mask,
        layout.cw,
        layout.ch,
        layout.min_area_chroma(min_area_fullres),
    );
    let best = boxes.iter().max_by_key(|b| b.area)?;

    // 色度网格坐标 → 全分辨率坐标（横向恒定 ×2；纵向 I420 是 ×2、I422 是 ×1）。
    let x = best.x * layout.sx;
    let y = best.y * layout.sy;
    let w = best.w * layout.sx;
    let h = best.h * layout.sy;
    Some(BucketResult {
        area_ratio: best.area as f32 / layout.sample_count() as f32,
        cx: x + w / 2,
        cy: y + h / 2,
        x,
        y,
        w,
        h,
    })
}

/// 诊断用：把画面里"有颜色"（够饱和、够亮）的采样点的 H/S/V 均值报出来。
///
/// 用来对着真桶标定颜色：`detect_bucket` 判不出来时，这里能告诉你桶的实际色相
/// 落在哪 —— 比如"均值 H=25"就说明该把 `bucket_color` 换成 `red`，"均值 H=215"
/// 却报 not found 说明 `blue` 的窗口漏了它。**它故意不走早退那道判据**：那道是给
/// "当前目标色"用的，而标定恰恰要知道"当前目标色之外还有什么颜色"。
///
/// 用法是把桶怼近相机再跑 `capture --bucket`：均值会被画面里所有饱和色一起拉动，
/// 桶占满画面时才等于桶的颜色。它只用来读数定阈值，不参与控制。
///
/// 注意均值是跨 0/360 卷绕取的：蓝色（240）离接缝远，量蓝色很准；
/// 红色（0/360 正好是接缝）会被拉偏，只能当粗参考。
pub fn bucket_color_stats(
    frame: &[u8],
    width: i32,
    height: i32,
    planar: PlanarFormat,
) -> Option<(i32, i32, i32, usize)> {
    if width <= 0 || height <= 0 {
        return None;
    }
    let layout = ChromaLayout::for_format(planar, width, height);
    if frame.len() < layout.v_off + layout.sample_count() {
        return None;
    }
    let w = width as usize;
    // 和检测器用同一条 S/V 下限(取自目标色预设),只是**不卡色相** —— 这正是标定要看的
    // 东西:被阈值卡掉的那些像素到底是什么色相。
    let preset = target_color();
    let (mut hs, mut ss, mut vs, mut n) = (0i64, 0i64, 0i64, 0usize);
    for cy in 0..layout.ch as usize {
        for cx in 0..layout.cw as usize {
            let (h, s, v) = sample_hsv(frame, w, &layout, cx, cy);
            if s < preset.min_sat || v < preset.min_val {
                continue;
            }
            hs += h as i64;
            ss += s as i64;
            vs += v as i64;
            n += 1;
        }
    }
    if n == 0 {
        return None;
    }
    Some((
        (hs / n as i64) as i32,
        (ss / n as i64) as i32,
        (vs / n as i64) as i32,
        n,
    ))
}

/// 取一个色度采样点，重建出它的 (H, S, V)。
/// 亮度取该采样点覆盖的亮度块均值（I420: 2×2，I422: 2×1）—— 检测器和标定都走这里，
/// 保证"哪里算目标色"和"看到的色相是多少"用的是同一套几何。
fn sample_hsv(
    frame: &[u8],
    width: usize,
    layout: &ChromaLayout,
    cx: usize,
    cy: usize,
) -> (i32, i32, i32) {
    let idx = cy * layout.cw as usize + cx;
    let u = frame[layout.u_off + idx];
    let v = frame[layout.v_off + idx];

    let (mut sum, mut n) = (0u32, 0u32);
    for dy in 0..layout.sy as usize {
        let row = cy * layout.sy as usize + dy;
        for dx in 0..layout.sx as usize {
            sum += frame[row * width + cx * layout.sx as usize + dx] as u32;
            n += 1;
        }
    }
    let (r, g, b) = yuv_to_rgb((sum / n) as u8, u, v);
    rgb2hsv(r, g, b)
}

/// 逐采样点判目标色，生成 0/1 掩膜。
fn build_color_mask(frame: &[u8], width: i32, layout: &ChromaLayout) -> Vec<u8> {
    let w = width as usize;
    // 早退判据和颜色判据必须来自同一份预设:各读一次配置既慢,也可能在运行期 init
    // 之后两份对不上,变成"早退按蓝、判据按红"。
    let preset = target_color();
    let mut mask = vec![0u8; layout.sample_count()];
    for cy in 0..layout.ch as usize {
        for cx in 0..layout.cw as usize {
            let idx = cy * layout.cw as usize + cx;
            if preset
                .guard
                .excludes(frame[layout.u_off + idx], frame[layout.v_off + idx])
            {
                continue;
            }
            let (h, s, v) = sample_hsv(frame, w, layout, cx, cy);
            if preset.matches(h, s, v) {
                mask[idx] = 1;
            }
        }
    }
    mask
}

/// BT.601 定点 YUV→RGB，系数与 `image_bridge.rs:317-355` 保持一致
/// （同一条色彩链路，不然两处对"红"的判断会被不同的矩阵悄悄错开）。
fn yuv_to_rgb(y: u8, u: u8, v: u8) -> (u8, u8, u8) {
    let y = y as i32;
    let u = u as i32 - 128;
    let v = v as i32 - 128;
    let r = y + ((91881 * v) >> 16);
    let g = y - ((22554 * u + 46802 * v) >> 16);
    let b = y + ((116130 * u) >> 16);
    (clamp_u8(r), clamp_u8(g), clamp_u8(b))
}

fn clamp_u8(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// RGB → HSV，**逐字照搬参考**（`test_cmds.cpp:470-488`），包括两个怪癖：
/// 红区的取模 `(G-B) % (delta*6)`，以及负色相 `+360` 回绕后**先转 int 再返回**。
/// 它不是教科书写法，但阈值是针对它的输出调出来的 —— 改了它，`color_preset` 里的窗口就挪了。
fn rgb2hsv(r: u8, g: u8, b: u8) -> (i32, i32, i32) {
    let (r, g, b) = (r as i32, g as i32, b as i32);
    let cmax = r.max(g).max(b);
    let cmin = r.min(g).min(b);
    let delta = cmax - cmin;

    let v = cmax;
    let s = if cmax == 0 { 0 } else { delta * 255 / cmax };
    if delta == 0 {
        return (0, s, v);
    }

    let h = if cmax == r {
        60.0 * ((g - b) % (delta * 6)) as f32 / delta as f32
    } else if cmax == g {
        60.0 * ((b - r) as f32 / delta as f32 + 2.0)
    } else {
        60.0 * ((r - g) as f32 / delta as f32 + 4.0)
    };
    let h = if h < 0.0 { h + 360.0 } else { h };
    (h as i32, s, v)
}

// ── 目标色预设 ──────────────────────────────────────────────────────────
//
// 色相/饱和度这些数值**只出现在这张表里**:配置文件只写颜色名
// (`[bucket] bucket_color = "blue"`),`config::BucketColor` 到这里查判据。
//
// 本仓 `rgb2hsv` 约定:红=0/360、黄=60、绿=120、青=180、纯蓝=240、品红=300。
// 三种颜色的 S/V 下限都沿用参考的 S≥80、V≥50。

/// 一种目标色的完整判据。
#[derive(Clone, Copy, Debug)]
pub struct ColorPreset {
    /// 颜色名,与 TOML 取值一致(日志用)。
    pub name: &'static str,
    /// 窗口一端:不跨接缝时是下界;跨接缝(`wrap`)时是贴着 360 那一段的起点。
    pub hue_min: i32,
    /// 窗口另一端:不跨接缝时是上界;跨接缝时是贴着 0 那一段的终点。
    pub hue_max: i32,
    /// 窗口是否横跨 0/360 接缝(红)。
    pub wrap: bool,
    /// 饱和度下限(0~255)。
    pub min_sat: i32,
    /// 亮度下限(0~255)。
    pub min_val: i32,
    /// 无损早退:哪个色度平面低到阈值以下就一定不是这个颜色。
    guard: ChromaGuard,
}

impl ColorPreset {
    /// `(h, s, v)` 是否落在目标色窗口内。
    fn matches(&self, h: i32, s: i32, v: i32) -> bool {
        if s < self.min_sat || v < self.min_val {
            return false;
        }
        if self.wrap {
            h >= self.hue_min || h <= self.hue_max
        } else {
            (self.hue_min..=self.hue_max).contains(&h)
        }
    }

    /// 色相窗口的可读形式(日志用):跨接缝时写成两段。
    pub fn hue_window_label(&self) -> String {
        if self.wrap {
            format!("H≥{} 或 H≤{}", self.hue_min, self.hue_max)
        } else {
            format!("H∈[{},{}]", self.hue_min, self.hue_max)
        }
    }
}

/// 单平面色度早退判据:那个平面的值低于阈值就一定不是目标色。
///
/// 它是**无损**的(写错只会丢早退、变慢,不会漏检),所以按色相的主通道挑一项就够:
/// 蓝色时 `cmax == B`,而 `Cb - 128 = 0.5B - 0.169R - 0.331G ≥ 0` 恒成立(最坏情况
/// R=G=B 是灰、S=0,本来就过不了判据),所以 Cb 明显低于 128 的像素(绿/红/黄背景)
/// 一定不是蓝;120 是给定点换算留的舍入余量。红看 Cr(V 平面),对称。
/// 绿/青的 U、V 会同时越过 128(H 越靠青 U 越高),没有单平面判据,只能走全路径。
#[derive(Clone, Copy, Debug)]
enum ChromaGuard {
    /// U(Cb) 低于此值 → 不是目标色。
    LowU(u8),
    /// V(Cr) 低于此值 → 不是目标色。
    LowV(u8),
    /// 没有单平面判据。
    None,
}

impl ChromaGuard {
    fn excludes(self, u: u8, v: u8) -> bool {
        match self {
            ChromaGuard::LowU(limit) => u < limit,
            ChromaGuard::LowV(limit) => v < limit,
            ChromaGuard::None => false,
        }
    }
}

/// 颜色名 → 判据。加一种颜色只加一个分支、一行测试。
fn color_preset(color: config::BucketColor) -> ColorPreset {
    match color {
        // 蓝窗**别按"240 两边各放 20"来取**。这个约定下 R=0 时 `H = 60×(4 - G/B)`,
        // 只要蓝里带一点绿,色相就往下漂 —— 真实的蓝桶(深蓝 R=0/G=90/B=170)落在 208,
        // 浅蓝 (100,160,220) 落在 210,紫蓝 (80,60,200) 才到 248;取 [220,260] 会把最
        // 常见的几种蓝全漏掉。所以用 OpenCV 惯例的蓝色区间:0-180 制的 H∈[100,130],
        // 换算到本约定就是 [200,260] —— 盖住上述所有蓝,同时把青(180)挡在外面。
        config::BucketColor::Blue => ColorPreset {
            name: "blue",
            hue_min: 200,
            hue_max: 260,
            wrap: false,
            min_sat: 80,
            min_val: 50,
            guard: ChromaGuard::LowU(120),
        },
        // 红色正好横跨 0/360 接缝:直窗口 [340,20] 会一个字都匹配不上,必须回绕。
        config::BucketColor::Red => ColorPreset {
            name: "red",
            hue_min: 340,
            hue_max: 20,
            wrap: true,
            min_sat: 80,
            min_val: 50,
            guard: ChromaGuard::LowV(120),
        },
        // 绿窗居中于 120:上界 160 把青(180)挡在外面,下界 80 与网球的黄绿(55~70)
        // 留出间隔 —— 手里夹着的那颗球不会被当成桶。
        config::BucketColor::Green => ColorPreset {
            name: "green",
            hue_min: 80,
            hue_max: 160,
            wrap: false,
            min_sat: 80,
            min_val: 50,
            guard: ChromaGuard::None,
        },
    }
}

/// 当前配置的目标色判据。`capture --bucket` 用它打印"按什么判"。
pub fn target_color() -> ColorPreset {
    color_preset(config::get().bucket.bucket_color)
}

/// 两遍扫描 + 并查集的 4-连通域标记，照搬参考 `test_cmds.cpp:497-576`。
/// 输入是色度网格上的 0/1 掩膜，返回面积 ≥ `min_area`（采样数）的连通域。
fn find_boxes(mask: &[u8], cw: i32, ch: i32, min_area: i32) -> Vec<BucketBox> {
    let (cw, ch) = (cw as usize, ch as usize);
    let n = cw * ch;
    if mask.len() < n || n == 0 {
        return Vec::new();
    }

    let mut label = vec![0i32; n];
    // parent[0] = 0 占位：标签 0 表示背景。
    let mut parent: Vec<i32> = vec![0];
    let mut next_label = 1i32;

    // 第一遍：只看上、左两个邻居（4-连通）。
    for y in 0..ch {
        for x in 0..cw {
            let idx = y * cw + x;
            if mask[idx] == 0 {
                continue;
            }
            let up = if y > 0 { label[(y - 1) * cw + x] } else { 0 };
            let left = if x > 0 { label[y * cw + x - 1] } else { 0 };
            match (up, left) {
                (0, 0) => {
                    label[idx] = next_label;
                    parent.push(next_label);
                    next_label += 1;
                }
                (u, 0) => label[idx] = u,
                (0, l) => label[idx] = l,
                (u, l) => {
                    label[idx] = l;
                    let (ra, rb) = (
                        uf_find(&mut parent, u as usize),
                        uf_find(&mut parent, l as usize),
                    );
                    if ra != rb {
                        parent[ra] = rb as i32;
                    }
                }
            }
        }
    }

    // 第二遍：路径压缩后统计每个根的包围盒与面积。
    let mut x0 = vec![cw as i32; next_label as usize];
    let mut y0 = vec![ch as i32; next_label as usize];
    let mut x1 = vec![-1i32; next_label as usize];
    let mut y1 = vec![-1i32; next_label as usize];
    let mut cnt = vec![0i32; next_label as usize];
    for y in 0..ch {
        for x in 0..cw {
            let idx = y * cw + x;
            if label[idx] == 0 {
                continue;
            }
            let root = uf_find(&mut parent, label[idx] as usize);
            label[idx] = root as i32;
            cnt[root] += 1;
            let (xi, yi) = (x as i32, y as i32);
            if xi < x0[root] {
                x0[root] = xi;
            }
            if yi < y0[root] {
                y0[root] = yi;
            }
            if xi > x1[root] {
                x1[root] = xi;
            }
            if yi > y1[root] {
                y1[root] = yi;
            }
        }
    }

    let mut boxes = Vec::new();
    for l in 1..next_label as usize {
        // 第二遍之后所有 label 都已指向根，非根节点没有自己的像素。
        if uf_find(&mut parent, l) != l || cnt[l] < min_area {
            continue;
        }
        boxes.push(BucketBox {
            x: x0[l],
            y: y0[l],
            w: x1[l] - x0[l] + 1,
            h: y1[l] - y0[l] + 1,
            area: cnt[l],
        });
    }
    boxes
}

/// 并查集 find，路径减半（避免递归，同参考的 while 写法）。
fn uf_find(parent: &mut [i32], mut x: usize) -> usize {
    while parent[x] as usize != x {
        parent[x] = parent[parent[x] as usize];
        x = parent[x] as usize;
    }
    x
}

// ── 趋近控制 ──────────────────────────────────────────────────────────────

/// 趋近桶时的两轮速度。照搬参考 `tennis.cpp:424-434`：比例转向 + 中心死区 +
/// 两级速度。`cx` 与 `half_w` 都在全分辨率坐标系里。`polarity` 来自
/// `[chassis] turn_reverse`：轮子通道与物理左右反接的底盘用它把差速镜像，
/// 使修正朝物理上正确的一侧；死区/幅值/速度档与极性无关。
fn bucket_approach_command(area_ratio: f32, cx: i32, half_w: i32, polarity: i32) -> (i32, i32) {
    let cfg = config::get().bucket;
    let (approach_spd, brake_spd) = approach_speeds(&cfg);
    let half_w = half_w.max(1);
    let off = cx - half_w;
    let bias = if off.abs() <= cfg.center_dead_zone {
        0
    } else {
        (cfg.k_turn * off as f32 / half_w as f32) as i32
    };
    let bias = bias.clamp(-cfg.max_bias, cfg.max_bias) * polarity;
    let spd = if area_ratio >= cfg.area_brake {
        brake_spd
    } else {
        approach_spd
    };
    ((spd + bias).clamp(-100, 100), (spd - bias).clamp(-100, 100))
}

// ── 状态机 ────────────────────────────────────────────────────────────────

/// 放球后的倒退时长 → 目的地相。`ms = 0` 表示不退，直接收工（旧行为）。
fn deposit_backoff_phase(ms: u64, now: Instant) -> Option<BucketPhase> {
    (ms > 0).then(|| BucketPhase::Backoff {
        until: now + Duration::from_millis(ms),
    })
}

/// 一帧的只读视图，打包成**一个**参数。
///
/// 拆成四个散参数（pixels / width / height / planar）时，任何一处传错都不会报错 ——
/// 只会拿错的几何去算，静默出一堆垃圾。打包之后类型系统替我们盯着这件事。
#[derive(Clone, Copy, Debug)]
pub struct FrameView<'a> {
    pub pixels: &'a [u8],
    pub width: i32,
    pub height: i32,
    pub planar: PlanarFormat,
}

/// 桶流程是否还在进行中。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BucketOutcome {
    Busy,
    /// 放球完成，调用方负责复位球侧状态并回到追球。
    Finished,
}

/// 桶流程的阶段。与 `robot.rs` 的 `GrabPhase` 同构：进入某段时发一次动作，
/// 之后按时间戳推进，主循环照常取帧打日志。
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BucketPhase {
    /// 未启用（默认）。
    Idle,
    /// 旋转搜索目标色桶；看到后连续确认若干帧。
    Find,
    /// 已确认，差速对准并趋近。
    Approaching,
    /// 够近，硬刹 350ms（参考在停车前整段 brake）。
    Braking { until: Instant },
    /// 已松爪，等舵机走完再把机械臂归位。
    DepositRelease { until: Instant },
    /// 机械臂归位中，之后放球尾部结束。
    DepositHome { until: Instant },
    /// 归位后倒退一小段，把车头从桶边挪开，然后交还追球。
    Backoff { until: Instant },
}

/// 桶流程的全部状态。必须保持 `Copy` —— `RobotState` 依赖它。
#[derive(Clone, Copy, Debug)]
pub struct BucketTrack {
    phase: BucketPhase,
    /// 连续看到桶的帧数（防抖）。
    confirm: i32,
    /// 连续看不到桶的帧数（趋近中用它决定退回搜索）。
    lost: i32,
}

impl Default for BucketTrack {
    fn default() -> Self {
        Self {
            phase: BucketPhase::Idle,
            confirm: 0,
            lost: 0,
        }
    }
}

impl BucketTrack {
    /// 桶流程是不是正在进行（主循环据此决定走不走拷贝路径）。
    pub fn is_active(&self) -> bool {
        !matches!(self.phase, BucketPhase::Idle)
    }

    #[cfg(test)]
    pub fn phase(&self) -> BucketPhase {
        self.phase
    }

    #[cfg(test)]
    pub fn confirm_count(&self) -> i32 {
        self.confirm
    }

    #[cfg(test)]
    pub fn lost_count(&self) -> i32 {
        self.lost
    }

    /// 抓到球之后进入：开始找桶。调用方此时夹爪闭合、球在手上。
    pub fn start(&mut self) {
        self.phase = BucketPhase::Find;
        self.confirm = 0;
        self.lost = 0;
    }

    /// 收尾：清干净状态并把控制权交回追球。
    fn finish(&mut self) -> BucketOutcome {
        self.phase = BucketPhase::Idle;
        self.confirm = 0;
        self.lost = 0;
        if bucket_log() {
            eprintln!("[bucket] done -> back to chase");
        }
        BucketOutcome::Finished
    }

    /// 推进一帧。`frame` 是拷贝路径(ioctl 4)拿到的平面 YUV 原始字节 ——
    /// 相机 I/O 留在调用方，这里只吃字节，所以单测能直接喂合成帧。
    pub fn step(
        &mut self,
        frame: FrameView<'_>,
        motor: &mut Motor,
        arm: &mut Arm,
        now: Instant,
    ) -> BucketOutcome {
        let (width, height, planar) = (frame.width, frame.height, frame.planar);
        // 本帧的桶参数快照（全局配置是 Copy，逐帧取一次即可）。
        let cfg = config::get().bucket;
        match self.phase {
            BucketPhase::Idle => BucketOutcome::Finished,

            BucketPhase::Braking { until } => {
                if now < until {
                    motor.brake();
                    return BucketOutcome::Busy;
                }
                // 停稳 → 松爪，球落进桶里。
                motor.standby();
                arm.release();
                self.phase = BucketPhase::DepositRelease {
                    until: now + Duration::from_millis(cfg.deposit_ms),
                };
                if bucket_log() {
                    eprintln!("[bucket] deposit: braking done, released ball");
                }
                BucketOutcome::Busy
            }

            BucketPhase::DepositRelease { until } => {
                motor.standby();
                if now < until {
                    return BucketOutcome::Busy;
                }
                arm.grab_pos();
                self.phase = BucketPhase::DepositHome {
                    until: now + Duration::from_millis(cfg.home_ms),
                };
                BucketOutcome::Busy
            }

            BucketPhase::DepositHome { until } => {
                motor.standby();
                if now < until {
                    return BucketOutcome::Busy;
                }
                // 球已落进桶,车头还贴着桶壁:先倒一小段再交还追球。不然下一步
                // “没看到球 → 原地旋转搜索”就在桶鼻子上打转,车身扫过去就蹭上了。
                match deposit_backoff_phase(cfg.deposit_backoff_ms, now) {
                    Some(phase) => self.phase = phase,
                    None => return self.finish(),
                }
                BucketOutcome::Busy
            }

            BucketPhase::Backoff { until } => {
                // 只求离开桶:整段保持倒车速度,到点刹车交还追球。按帧定时,
                // 差一帧的位移无所谓 —— 这里不需要抓球后退脉冲那种“退准”。
                if now < until {
                    motor.backward(config::get().chassis.backward_speed);
                    return BucketOutcome::Busy;
                }
                motor.brake();
                if bucket_log() {
                    eprintln!("[bucket] deposit: backed off, handing back to chase");
                }
                self.finish()
            }

            BucketPhase::Find => {
                match detect_bucket(frame.pixels, width, height, planar, cfg.min_area_fullres) {
                    Some(r) => {
                        self.confirm += 1;
                        self.lost = 0;
                        // 确认期内站定不动，别边转边确认。
                        motor.standby();
                        if self.confirm >= cfg.confirm_frames {
                            self.phase = BucketPhase::Approaching;
                            self.confirm = 0;
                            if bucket_log() {
                                eprintln!(
                                    "[bucket] confirmed, area={:.3} -> approach",
                                    r.area_ratio
                                );
                            }
                        } else if bucket_log() {
                            eprintln!(
                                "[bucket] find: seen {}/{} area={:.3}",
                                self.confirm, cfg.confirm_frames, r.area_ratio
                            );
                        }
                    }
                    None => {
                        self.confirm = 0;
                        self.lost += 1;
                        // 参考固定右转；不按"最后看到的一侧"回摆，因为桶可能从未出现过。
                        // 轮速过极性镜像后，日志里的"右转"始终指物理右转，与丢球搜索一致。
                        let p = turn_polarity();
                        motor.drive(p * cfg.search_speed, -p * cfg.search_speed);
                        if bucket_log() {
                            eprintln!("[bucket] find: searching right, lost={}", self.lost);
                        }
                    }
                }
                BucketOutcome::Busy
            }

            BucketPhase::Approaching => {
                match detect_bucket(frame.pixels, width, height, planar, cfg.min_area_fullres) {
                    None => {
                        self.lost += 1;
                        if self.lost > cfg.lost_frames {
                            self.phase = BucketPhase::Find;
                            self.confirm = 0;
                            motor.standby();
                        } else {
                            // 短暂丢失：站住等它回来，别接着往前冲。
                            motor.standby();
                        }
                        if bucket_log() {
                            eprintln!("[bucket] approach: lost {}/{}", self.lost, cfg.lost_frames);
                        }
                    }
                    Some(r) => {
                        self.lost = 0;
                        if r.area_ratio >= cfg.area_deposit {
                            self.phase = BucketPhase::Braking {
                                until: now + Duration::from_millis(cfg.brake_ms),
                            };
                            motor.brake();
                            if bucket_log() {
                                eprintln!(
                                    "[bucket] area={:.3} -> brake {}ms",
                                    r.area_ratio, cfg.brake_ms
                                );
                            }
                        } else {
                            let (l, rr) = bucket_approach_command(
                                r.area_ratio,
                                r.cx,
                                width / 2,
                                turn_polarity(),
                            );
                            motor.drive(l, rr);
                            if bucket_log() {
                                eprintln!(
                                    "[bucket] approach area={:.3} cx={} off={} L={} R={}",
                                    r.area_ratio,
                                    r.cx,
                                    r.cx - width / 2,
                                    l,
                                    rr
                                );
                            }
                        }
                    }
                }
                BucketOutcome::Busy
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arm::Arm;
    use crate::motor::Motor;

    const W: i32 = 640;
    const H: i32 = 480;

    /// BT.601 正向（测试侧），与 `yuv_to_rgb` 的定点逆变换配对使用。
    fn rgb_to_yuv601(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
        let (r, g, b) = (r as f32, g as f32, b as f32);
        let y = 0.299 * r + 0.587 * g + 0.114 * b;
        let cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b;
        let cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b;
        (
            y.round().clamp(0.0, 255.0) as u8,
            cb.round().clamp(0.0, 255.0) as u8,
            cr.round().clamp(0.0, 255.0) as u8,
        )
    }

    /// 测试用的"桶"颜色：纯蓝（H=240，落在 blue 窗正中）。
    /// 换目标色时改这里和 `color_preset` 的预设表，其余测试跟着走。
    const TEST_BUCKET_RGB: (u8, u8, u8) = (0, 0, 255);

    /// 造一帧平面 YUV：灰底 + 若干纯目标色矩形（全分辨率坐标，必须偶数对齐，
    /// 这样每个色度采样点覆盖的亮度块要么全是目标色要么全灰，边界不含糊）。
    fn synthetic(
        w: usize,
        h: usize,
        planar: PlanarFormat,
        rects: &[(usize, usize, usize, usize)],
    ) -> Vec<u8> {
        let y_size = w * h;
        let c_size = match planar {
            PlanarFormat::I420 => w * h / 4,
            PlanarFormat::I422 => w * h / 2,
        };
        let mut f = vec![128u8; y_size + c_size * 2];
        let (ry, ru, rv) = rgb_to_yuv601(TEST_BUCKET_RGB.0, TEST_BUCKET_RGB.1, TEST_BUCKET_RGB.2);
        for &(x, y, rw, rh) in rects {
            assert!(
                x % 2 == 0 && y % 2 == 0 && rw % 2 == 0 && rh % 2 == 0,
                "测试矩形必须偶数对齐"
            );
            assert!(
                x + rw <= w && y + rh <= h,
                "测试矩形越界:{x},{y},{rw},{rh} 不在 {w}×{h} 内"
            );
            for row in y..y + rh {
                f[row * w + x..row * w + x + rw].fill(ry);
            }
            let cw = w / 2;
            let (v_off, row_scale) = match planar {
                PlanarFormat::I420 => (y_size + c_size, 2usize),
                PlanarFormat::I422 => (y_size + c_size, 1usize),
            };
            for row in y..y + rh {
                if row_scale == 2 && row % 2 != 0 {
                    continue;
                }
                let crow = row / row_scale;
                for col in (x..x + rw).step_by(2) {
                    let idx = crow * cw + col / 2;
                    f[y_size + idx] = ru;
                    f[v_off + idx] = rv;
                }
            }
        }
        f
    }

    // ── 颜色判据 ──────────────────────────────────────────────────────────

    #[test]
    fn converts_rgb_to_hsv_like_the_reference() {
        assert_eq!(rgb2hsv(255, 0, 0), (0, 255, 255), "纯红:H=0");
        assert_eq!(rgb2hsv(0, 255, 0), (120, 255, 255), "纯绿:H=120");
        assert_eq!(rgb2hsv(0, 0, 255), (240, 255, 255), "纯蓝:H=240");
        // 灰色:delta=0,S=0。
        assert_eq!(rgb2hsv(100, 100, 100), (0, 0, 100));
        // G<B 的红(品红一侧):靠取模 + `+360` 回绕落到 300~360。
        let (h, s, v) = rgb2hsv(200, 50, 150);
        assert!((300..360).contains(&h), "期望落在回绕段,实得 H={h}");
        assert!(s > 0 && v == 200);
    }

    /// 蓝窗边界必须是闭的，且不许把色相环上的邻居放进来。
    #[test]
    fn blue_window_boundaries_match() {
        // 窗口 [200, 260]，门槛 S≥80、V≥50。
        let blue = color_preset(config::BucketColor::Blue);
        assert!(blue.matches(blue.hue_min, 80, 50), "闭区间左端");
        assert!(blue.matches(blue.hue_max, 80, 50), "闭区间右端");
        assert!(!blue.matches(blue.hue_min - 1, 255, 255), "左端外一点");
        assert!(!blue.matches(blue.hue_max + 1, 255, 255), "右端外一点");
        assert!(!blue.matches(240, 79, 255), "S 差一点");
        assert!(!blue.matches(240, 255, 49), "V 差一点");
        // 色相环上的非蓝邻居一个都不许漏进来。青(180)是左边界最近的那个。
        for h in [0, 60, 120, 180, 190, 270, 300, 340] {
            assert!(!blue.matches(h, 255, 255), "H={h} 不该被判成桶");
        }
    }

    /// 红窗横跨 0/360 接缝，必须写成回绕形式:直窗口 `[340,20]` 会一个字都匹配不上。
    #[test]
    fn red_window_wraps_across_the_hue_seam() {
        let red = color_preset(config::BucketColor::Red);
        assert!(red.wrap);
        for h in [340, 350, 359, 0, 10, 20] {
            assert!(red.matches(h, 255, 255), "红窗该收下 H={h}");
        }
        for h in [30, 60, 120, 200, 240, 300, 330] {
            assert!(!red.matches(h, 255, 255), "红窗不该收 H={h}");
        }
    }

    /// 无论选哪个颜色，网球（黄绿 H≈55~70）都不许被判成桶 —— 否则追桶循环会把
    /// 手里夹着的那颗球当成目标。
    #[test]
    fn no_preset_takes_the_tennis_ball_for_a_bucket() {
        for color in [
            config::BucketColor::Blue,
            config::BucketColor::Red,
            config::BucketColor::Green,
        ] {
            let preset = color_preset(color);
            for h in 50..=75 {
                assert!(
                    !preset.matches(h, 255, 255),
                    "{} 把网球 H={h} 当桶了",
                    preset.name
                );
            }
        }
    }

    /// 早退只能让进算更快，不许挡住真正的目标色。
    #[test]
    fn chroma_guard_never_hides_the_target_colour() {
        let blue = color_preset(config::BucketColor::Blue);
        let (_, u, _) = rgb_to_yuv601(0, 90, 170);
        assert!(!blue.guard.excludes(u, 128), "深蓝桶被 U 早退挡掉了");
        let red = color_preset(config::BucketColor::Red);
        let (_, _, v) = rgb_to_yuv601(200, 30, 30);
        assert!(!red.guard.excludes(128, v), "红桶被 V 早退挡掉了");
        // 绿色没有单平面判据，任何色度值都不许早退。
        let green = color_preset(config::BucketColor::Green);
        assert!(!green.guard.excludes(255, 255));
    }

    /// 窗口必须盖住**真实的蓝**，不是只盖住"纯蓝"。这个约定下 R=0 时
    /// `H = 60×(4 - G/B)`，蓝里带绿就往下漂 —— 顺手把这块盯住，
    /// 免得以后有人把窗口"对称地"调回 240±20。
    #[test]
    fn hue_window_covers_realistic_blue_objects() {
        // 几种真实蓝物体在该 RGB 下的色相(见 rgb2hsv 的算法)。
        let realistic_blues = [
            ((0u8, 0u8, 255u8), 240), // 理想纯蓝
            ((0, 90, 170), 208),      // 常见深蓝塑料桶
            ((0, 60, 120), 210),      // 更暗的蓝
            ((30, 80, 200), 222),     // 标准中蓝
            ((100, 160, 220), 210),   // 浅蓝/天蓝
            ((80, 60, 200), 248),     // 偏紫的蓝
            ((20, 35, 90), 227),      // 藏青
        ];
        for ((r, g, b), expect) in realistic_blues {
            let (h, s, v) = rgb2hsv(r, g, b);
            assert_eq!(h, expect, "({r},{g},{b}) 的色相遇算计有变");
            assert!(
                color_preset(config::BucketColor::Blue).matches(h, s, v),
                "({r},{g},{b}) H={h} 应被判成蓝"
            );
        }
    }

    #[test]
    fn converts_yuv_back_to_rgb() {
        let (r, g, b) = yuv_to_rgb(76, 85, 255);
        assert!(
            r > 240 && g < 20 && b < 20,
            "纯红应还原成红,实得 {r},{g},{b}"
        );
        let (r, g, b) = yuv_to_rgb(128, 128, 128);
        assert!(
            (r as i32 - 128).abs() <= 2
                && (g as i32 - 128).abs() <= 2
                && (b as i32 - 128).abs() <= 2,
            "灰应还原成灰,实得 {r},{g},{b}"
        );
    }

    // ── 检测器 ────────────────────────────────────────────────────────────

    #[test]
    fn finds_a_blue_rect_and_reports_full_resolution_coords() {
        let frame = synthetic(640, 480, PlanarFormat::I420, &[(200, 150, 200, 160)]);
        let r = detect_bucket(&frame, W, H, PlanarFormat::I420, BUCKET_MIN_AREA_FULLRES)
            .expect("应找到蓝块");
        // 面积比与全分辨率口径一致:200*160 / (640*480)
        let expect = (200.0 * 160.0) / (640.0 * 480.0);
        assert!(
            (r.area_ratio - expect).abs() < 1e-4,
            "实得 {}",
            r.area_ratio
        );
        // 坐标必须换算回全分辨率,否则死区 15px / 半宽 320 这些常数全错。
        assert_eq!((r.x, r.w), (200, 200));
        assert_eq!((r.y, r.h), (150, 160));
        assert_eq!((r.cx, r.cy), (300, 230));
    }

    #[test]
    fn same_rect_gives_same_ratio_in_i422() {
        // 大核回退路径是 I422。采样率不同(2×1 vs 2×2),但面积比是尺度无关的,
        // 所以 0.90/0.70 这些阈值在两个格式下通用。
        let i420 = synthetic(640, 480, PlanarFormat::I420, &[(200, 150, 200, 160)]);
        let i422 = synthetic(640, 480, PlanarFormat::I422, &[(200, 150, 200, 160)]);
        let a = detect_bucket(&i420, W, H, PlanarFormat::I420, BUCKET_MIN_AREA_FULLRES).unwrap();
        let b = detect_bucket(&i422, W, H, PlanarFormat::I422, BUCKET_MIN_AREA_FULLRES).unwrap();
        assert!((a.area_ratio - b.area_ratio).abs() < 1e-4);
        assert_eq!((a.cx, a.cy), (b.cx, b.cy));
        assert_eq!((a.x, a.y, a.w, a.h), (b.x, b.y, b.w, b.h));
    }

    #[test]
    fn picks_the_largest_blob() {
        let frame = synthetic(
            640,
            480,
            PlanarFormat::I420,
            &[(0, 0, 100, 100), (400, 260, 200, 200)],
        );
        let r = detect_bucket(&frame, W, H, PlanarFormat::I420, BUCKET_MIN_AREA_FULLRES).unwrap();
        assert_eq!(r.cx, 500, "应取大的那块(400..600)");
        assert_eq!(r.w, 200);
    }

    #[test]
    fn connectivity_is_four_connected() {
        // 对角相接 → 两块;边相接 → 一块。色度采样点是 2×2 全分辨率像素,
        // 所以色度上相邻 = 全分辨率上隔 2 像素。
        let diag = synthetic(
            640,
            480,
            PlanarFormat::I420,
            &[(20, 20, 2, 2), (22, 22, 2, 2)],
        );
        let r = detect_bucket(&diag, W, H, PlanarFormat::I420, 1).unwrap();
        assert_eq!(r.area_ratio, 1.0 / (320.0 * 240.0), "只应命中一个采样点");
        // 直接打 find_boxes 看连通域个数更直接。
        let layout = ChromaLayout::for_format(PlanarFormat::I420, W, H);
        assert_eq!(
            find_boxes(
                &build_color_mask(&diag, W, &layout),
                layout.cw,
                layout.ch,
                1
            )
            .len(),
            2
        );

        let edge = synthetic(
            640,
            480,
            PlanarFormat::I420,
            &[(20, 20, 2, 2), (22, 20, 2, 2)],
        );
        assert_eq!(
            find_boxes(
                &build_color_mask(&edge, W, &layout),
                layout.cw,
                layout.ch,
                1
            )
            .len(),
            1
        );
    }

    #[test]
    fn min_area_scales_with_the_chroma_grid() {
        // 1000 全分辨率像素 → 250 个色度采样点。
        let small = synthetic(640, 480, PlanarFormat::I420, &[(100, 100, 30, 30)]); // 900px → 225 采样
        assert!(detect_bucket(&small, W, H, PlanarFormat::I420, BUCKET_MIN_AREA_FULLRES).is_none());
        let big = synthetic(640, 480, PlanarFormat::I420, &[(100, 100, 80, 50)]); // 4000px → 1000 采样
        assert!(detect_bucket(&big, W, H, PlanarFormat::I420, BUCKET_MIN_AREA_FULLRES).is_some());
    }

    #[test]
    fn rejects_short_and_gray_frames() {
        let short = vec![128u8; 1000];
        assert!(detect_bucket(&short, W, H, PlanarFormat::I420, 1).is_none());
        // 全灰:没有任何目标色。
        let gray = vec![128u8; 640 * 480 * 3 / 2];
        assert!(detect_bucket(&gray, W, H, PlanarFormat::I420, 1).is_none());
    }

    #[test]
    fn bucket_color_stats_reads_the_hue_off_a_full_blue_frame() {
        let blue = synthetic(640, 480, PlanarFormat::I420, &[(0, 0, 640, 480)]);
        let (h, s, v, n) =
            bucket_color_stats(&blue, W, H, PlanarFormat::I420).expect("应有有色像素");
        // 纯蓝经 YUV 往返后落在 239~240(定点往返有 ±1 的色相抖动),窗口正中。
        assert!((238..=241).contains(&h), "纯蓝 H≈240,实得 {h}");
        assert!(s >= 250 && v >= 250, "纯蓝应又饱和又亮,实得 S={s} V={v}");
        assert_eq!(n, 320 * 240, "整幅都是目标色,采样数应等于整个色度网格");
    }

    #[test]
    fn bucket_color_stats_ignores_gray() {
        // 全灰 S=0 —— 不该被当成"有色像素"报出来。
        let gray = vec![128u8; 640 * 480 * 3 / 2];
        assert!(bucket_color_stats(&gray, W, H, PlanarFormat::I420).is_none());
    }

    /// 这个才是标定场景:桶其实是橙的(H≈25),检测器看不见它(窗口是蓝 220~260),
    /// 但 `bucket_color_stats` 要能把"它其实是 25 号色"报出来 —— 这就是不开整车
    /// 也能发现"颜色搞错了 / 阈值该挪"的依据。
    #[test]
    fn bucket_color_stats_reports_a_hue_that_detection_rejects() {
        let (oy, ou, ov) = rgb_to_yuv601(255, 106, 0);
        let mut frame = vec![128u8; 640 * 480 * 3 / 2];
        let (w, h) = (640usize, 480usize);
        frame[..w * h].fill(oy);
        for i in 0..(w / 2) * (h / 2) {
            frame[w * h + i] = ou;
            frame[w * h + w * h / 4 + i] = ov;
        }

        assert!(
            detect_bucket(&frame, W, H, PlanarFormat::I420, 1).is_none(),
            "H≈25 落在蓝窗之外,检测器应该看不见"
        );
        let (h, s, v, n) =
            bucket_color_stats(&frame, W, H, PlanarFormat::I420).expect("标定应看得见");
        assert!((20..30).contains(&h), "应报出真实色相约 25,实得 H={h}");
        assert!(s >= 80 && v >= 50);
        assert_eq!(n, 320 * 240);
    }

    /// 暴力 flood fill,用来对拍并查集那份实现。
    fn brute_boxes(mask: &[u8], cw: usize, ch: usize) -> Vec<BucketBox> {
        let mut seen = vec![false; cw * ch];
        let mut out = Vec::new();
        for start in 0..cw * ch {
            if mask[start] == 0 || seen[start] {
                continue;
            }
            seen[start] = true;
            let mut stack = vec![start];
            let (mut x0, mut y0, mut x1, mut y1, mut area) =
                (cw as i32, ch as i32, -1i32, -1i32, 0i32);
            while let Some(i) = stack.pop() {
                let (x, y) = ((i % cw) as i32, (i / cw) as i32);
                area += 1;
                x0 = x0.min(x);
                y0 = y0.min(y);
                x1 = x1.max(x);
                y1 = y1.max(y);
                if x > 0 && mask[i - 1] != 0 && !seen[i - 1] {
                    seen[i - 1] = true;
                    stack.push(i - 1);
                }
                if (x as usize) + 1 < cw && mask[i + 1] != 0 && !seen[i + 1] {
                    seen[i + 1] = true;
                    stack.push(i + 1);
                }
                if y > 0 && mask[i - cw] != 0 && !seen[i - cw] {
                    seen[i - cw] = true;
                    stack.push(i - cw);
                }
                if (y as usize) + 1 < ch && mask[i + cw] != 0 && !seen[i + cw] {
                    seen[i + cw] = true;
                    stack.push(i + cw);
                }
            }
            out.push(BucketBox {
                x: x0,
                y: y0,
                w: x1 - x0 + 1,
                h: y1 - y0 + 1,
                area,
            });
        }
        out
    }

    #[test]
    fn union_find_matches_brute_force_on_random_masks() {
        // 并查集那两遍扫描是整份移植里最容易写错的地方,用随机掩膜对拍。
        let (cw, ch) = (37usize, 29usize);
        let mut seed = 0x1234_5678u32;
        for round in 0..200 {
            let mut mask = vec![0u8; cw * ch];
            for m in mask.iter_mut() {
                seed = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                *m = ((seed >> 16) & 1) as u8;
            }
            let mut got = find_boxes(&mask, cw as i32, ch as i32, 1);
            let mut want = brute_boxes(&mask, cw, ch);
            got.sort_by_key(|b| (b.y, b.x, b.w, b.h, b.area));
            want.sort_by_key(|b| (b.y, b.x, b.w, b.h, b.area));
            assert_eq!(got, want, "第 {round} 轮掩膜:连通域不一致");
        }
    }

    // ── 趋近控制 ──────────────────────────────────────────────────────────

    #[test]
    fn approach_holds_a_dead_zone_and_clamps_the_bias() {
        let half = W / 2; // 320
        // 死区内不动舵:off = cx - 320
        assert_eq!(bucket_approach_command(0.1, half, half, 1), (25, 25));
        assert_eq!(bucket_approach_command(0.1, half + 15, half, 1), (25, 25));
        // 出死区:比例转向
        assert_eq!(bucket_approach_command(0.1, half + 16, half, 1), (26, 24));
        // 偏到底:bias 夹在 ±8
        assert_eq!(bucket_approach_command(0.1, W, half, 1), (33, 17));
        assert_eq!(bucket_approach_command(0.1, 0, half, 1), (17, 33));
    }

    /// 两级趋近速度的护栏:低于"推得动整车"下限的取值被抬到下限,制动档不超过
    /// 趋近档 —— 否则制动区里车原地不动,面积不再涨,永远卡在 area_brake~area_deposit。
    #[test]
    fn approach_speeds_clamp_to_the_moving_threshold() {
        let mut cfg = config::BucketConfig::default();
        // 参考的制动档 5:抬到下限,但仍不高于趋近档。
        cfg.approach_speed = 25;
        cfg.brake_speed = 5;
        assert_eq!(approach_speeds(&cfg), (25, MIN_MOVING_SPEED));

        // 连趋近档都低于下限:两档一起抬到下限。
        cfg.approach_speed = 10;
        cfg.brake_speed = 22;
        assert_eq!(approach_speeds(&cfg), (MIN_MOVING_SPEED, MIN_MOVING_SPEED));

        // 制动档反而比趋近档快:压回趋近档(两级退化成一级,至少不会越靠近越快)。
        cfg.approach_speed = 30;
        cfg.brake_speed = 40;
        assert_eq!(approach_speeds(&cfg), (30, 30));

        // 默认值原样通过 —— 护栏不应悄悄改掉标定值。
        let d = config::BucketConfig::default();
        assert_eq!((d.approach_speed, d.brake_speed), approach_speeds(&d));
    }

    /// `[chassis] turn_reverse` 的极性只镜像差速修正，死区/幅值/速度档不变，
    /// 与追球/丢球搜索共用同一份接线补偿。
    #[test]
    fn approach_mirrors_only_the_bias_with_turn_polarity() {
        let half = W / 2; // 320
        // 死区内：直行不受极性影响。
        assert_eq!(bucket_approach_command(0.1, half + 15, half, -1), (25, 25));
        // 出死区：两轮目标镜像，修正幅值不变。
        assert_eq!(bucket_approach_command(0.1, half + 16, half, -1), (24, 26));
        assert_eq!(bucket_approach_command(0.1, W, half, -1), (17, 33));
        assert_eq!(bucket_approach_command(0.1, 0, half, -1), (33, 17));
        // 制动区同样只镜像方向，不拆速度档。
        assert_eq!(
            bucket_approach_command(0.80, half + 160, half, -1),
            (14, 30)
        );
    }

    #[test]
    fn approach_creeps_once_in_the_brake_zone() {
        let half = W / 2;
        assert_eq!(bucket_approach_command(0.69, half, half, 1), (25, 25));
        assert_eq!(
            bucket_approach_command(0.70, half, half, 1),
            (22, 22),
            "进入制动区降到蠕行档"
        );
        assert_eq!(bucket_approach_command(0.90, half, half, 1), (22, 22));
        // 制动区 + 大偏差:仍是一个向前的小弧,不是原地转。
        // (参考的 spd=5 在这里会让内轮反转成 13/-3 —— 而两轮都低于能动下限,
        //  整车根本不动,正是"靠近桶就站住"的成因。)
        // bias 在 off=160 就顶到 ±8 了(20×160/320 = 10 → 夹到 8)。
        assert_eq!(bucket_approach_command(0.80, half + 100, half, 1), (28, 16));
        assert_eq!(bucket_approach_command(0.80, half + 160, half, 1), (30, 14));
        assert_eq!(
            bucket_approach_command(0.80, half + 160, half, 1),
            bucket_approach_command(0.80, W, half, 1)
        );
        // 关键不变量:制动力矩全给到偏差那一侧时,外侧轮仍在能动档以上 ——
        // 车会偏着走,但绝不死住。(内侧轮可能低于下限而拖住,那是允许的。)
        let (l, r) = bucket_approach_command(0.80, half + 160, half, 1);
        assert!(
            l.max(r) >= MIN_MOVING_SPEED,
            "两轮都低于能动下限 = 整车死住"
        );
    }

    // ── 状态机 ────────────────────────────────────────────────────────────

    fn bucket_frame() -> Vec<u8> {
        // 够大的一块目标色,面积比 ≈ 0.92 > BUCKET_AREA_DEPOSIT。
        synthetic(640, 480, PlanarFormat::I420, &[(0, 0, 620, 460)])
    }

    fn small_bucket_frame() -> Vec<u8> {
        // 面积比 ≈ 0.10,看得见但还很远。
        synthetic(640, 480, PlanarFormat::I420, &[(280, 200, 80, 80)])
    }

    fn no_bucket_frame() -> Vec<u8> {
        vec![128u8; 640 * 480 * 3 / 2]
    }

    struct Rig {
        track: BucketTrack,
        motor: Motor,
        arm: Arm,
    }

    impl Rig {
        fn new() -> Self {
            Self {
                track: BucketTrack::default(),
                motor: Motor::mock(),
                arm: Arm::mock(),
            }
        }

        fn step(&mut self, frame: &[u8], now: Instant) -> BucketOutcome {
            self.track.step(
                FrameView {
                    pixels: frame,
                    width: 640,
                    height: 480,
                    planar: PlanarFormat::I420,
                },
                &mut self.motor,
                &mut self.arm,
                now,
            )
        }
    }

    #[test]
    fn idle_until_started() {
        let mut rig = Rig::new();
        assert!(!rig.track.is_active());
        assert_eq!(
            rig.step(&bucket_frame(), Instant::now()),
            BucketOutcome::Finished
        );
        rig.track.start();
        assert!(rig.track.is_active());
        assert_eq!(rig.track.phase(), BucketPhase::Find);
    }

    #[test]
    fn find_needs_three_consecutive_sightings() {
        let mut rig = Rig::new();
        let now = Instant::now();
        rig.track.start();
        let frame = small_bucket_frame();

        rig.step(&frame, now);
        assert_eq!(rig.track.phase(), BucketPhase::Find, "1 帧不够");
        rig.step(&frame, now);
        assert_eq!(rig.track.phase(), BucketPhase::Find, "2 帧不够");
        rig.step(&frame, now);
        assert_eq!(rig.track.phase(), BucketPhase::Approaching, "第 3 帧才确认");

        // 中间丢一帧 → 计数清零,重新数 3 帧。
        let mut rig = Rig::new();
        rig.track.start();
        rig.step(&frame, now);
        rig.step(&no_bucket_frame(), now);
        assert_eq!(rig.track.confirm_count(), 0);
        rig.step(&frame, now);
        rig.step(&frame, now);
        assert_eq!(
            rig.track.phase(),
            BucketPhase::Find,
            "丢过一帧后要重新数满 3 帧"
        );
    }

    #[test]
    fn approach_holds_then_gives_up_after_the_lost_grace() {
        let mut rig = Rig::new();
        let now = Instant::now();
        rig.track.start();
        let frame = small_bucket_frame();
        for _ in 0..BUCKET_CONFIRM_CNT {
            rig.step(&frame, now);
        }
        assert_eq!(rig.track.phase(), BucketPhase::Approaching);

        // 宽限期内:站住等,不回退。
        for _ in 0..BUCKET_LOST_FRAMES {
            rig.step(&no_bucket_frame(), now);
        }
        assert_eq!(
            rig.track.phase(),
            BucketPhase::Approaching,
            "10 帧内不该放弃"
        );
        // 第 11 帧 → 退回搜索。
        rig.step(&no_bucket_frame(), now);
        assert_eq!(rig.track.phase(), BucketPhase::Find);

        // 重新看到桶即清零丢失计数。
        let mut rig2 = Rig::new();
        rig2.track.start();
        for _ in 0..BUCKET_CONFIRM_CNT {
            rig2.step(&frame, now);
        }
        rig2.step(&no_bucket_frame(), now);
        assert_eq!(rig2.track.lost_count(), 1);
        rig2.step(&frame, now);
        assert_eq!(rig2.track.lost_count(), 0);
    }

    #[test]
    fn deposit_sequence_advances_on_timestamps() {
        let mut rig = Rig::new();
        let now = Instant::now();
        rig.track.start();
        let frame = small_bucket_frame();
        for _ in 0..BUCKET_CONFIRM_CNT {
            rig.step(&frame, now);
        }

        // 够近 → 硬刹。
        rig.step(&bucket_frame(), now);
        let BucketPhase::Braking { until } = rig.track.phase() else {
            panic!("期望 Braking,实得 {:?}", rig.track.phase());
        };
        assert_eq!(until - now, Duration::from_millis(BUCKET_BRAKE_MS));
        // 刹车期间不动地方。
        rig.step(&bucket_frame(), now + Duration::from_millis(1));
        assert!(matches!(rig.track.phase(), BucketPhase::Braking { .. }));

        // 刹完 → 松爪。
        let t1 = until;
        rig.step(&bucket_frame(), t1);
        let BucketPhase::DepositRelease { until: t2 } = rig.track.phase() else {
            panic!("期望 DepositRelease,实得 {:?}", rig.track.phase());
        };
        assert_eq!(t2 - t1, Duration::from_millis(BUCKET_DEPOSIT_MS));

        // 松爪等够 → 归位。
        rig.step(&bucket_frame(), t2);
        let BucketPhase::DepositHome { until: t3 } = rig.track.phase() else {
            panic!("期望 DepositHome,实得 {:?}", rig.track.phase());
        };
        assert_eq!(t3 - t2, Duration::from_millis(BUCKET_HOME_MS));

        // 归位等够 → 倒退一小段(按帧定时,和其它段同一套写法)。
        rig.step(&bucket_frame(), t3);
        let BucketPhase::Backoff { until: t4 } = rig.track.phase() else {
            panic!("期望 Backoff,实得 {:?}", rig.track.phase());
        };
        assert_eq!(t4 - t3, Duration::from_millis(BUCKET_DEPOSIT_BACKOFF_MS));
        // 倒退期间还没收工(控制权还在桶流程手里)。
        assert_eq!(
            rig.step(&bucket_frame(), t4 - Duration::from_millis(1)),
            BucketOutcome::Busy
        );

        // 倒退到点 → 收工,状态清干净。
        assert_eq!(rig.step(&bucket_frame(), t4), BucketOutcome::Finished);
        assert!(!rig.track.is_active());
        assert_eq!(rig.track.confirm_count(), 0);
        assert_eq!(rig.track.lost_count(), 0);
    }

    /// 倒退只是“放球尾部”的一段:`deposit_backoff_ms = 0` 时不排这一段、直接收工。
    #[test]
    fn deposit_backoff_phase_is_skipped_when_zero() {
        let now = Instant::now();
        assert_eq!(deposit_backoff_phase(0, now), None);
        assert_eq!(
            deposit_backoff_phase(300, now),
            Some(BucketPhase::Backoff {
                until: now + Duration::from_millis(300)
            })
        );
        // 缺省是真退,不是 0。
        assert!(config::BucketConfig::default().deposit_backoff_ms > 0);
    }

    #[test]
    fn deposit_runs_to_completion_after_the_bucket_leaves_view() {
        // 放球途中就算桶离开视野也必须走完,不能退回去搜索 —— 球已经在桶口了。
        let mut rig = Rig::new();
        let now = Instant::now();
        rig.track.start();
        let frame = small_bucket_frame();
        for _ in 0..BUCKET_CONFIRM_CNT {
            rig.step(&frame, now);
        }
        rig.step(&bucket_frame(), now);
        let mut t = now;
        while rig.track.is_active() {
            t += Duration::from_millis(50);
            rig.step(&no_bucket_frame(), t);
        }
        assert_eq!(rig.track.phase(), BucketPhase::Idle);
    }
}
