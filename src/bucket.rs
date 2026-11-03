//! 彩色桶识别 + 「抓起来 → 找桶 → 放进去」的状态机。**当前目标是蓝色**。
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
//!   5. **颜色不同**：参考写死红色，我们用蓝色。颜色只由 `BUCKET_HUE_MIN/MAX` 和
//!      `chroma_excludes_target` 这一对决定，其余代码颜色无关、换色不用改。

use crate::arm::Arm;
use crate::camera::PlanarFormat;
use crate::config;
use crate::motor::Motor;
use std::time::{Duration, Instant};

// ── 趋近参数（与参考同名同值） ────────────────────────────────────────────
/// 桶面积比 ≥ 此值 → 够近，可以放球。
const BUCKET_AREA_DEPOSIT: f32 = 0.90;
/// 桶面积比 ≥ 此值 → 进入制动区，降到蠕行速度（避免撞桶）。
const BUCKET_AREA_BRAKE: f32 = 0.70;
const BUCKET_APPROACH_SPD: i32 = 25;
/// 制动区（面积 ≥ 0.70）的蠕行速度。
///
/// **不能照抄参考的 5**。参考写 5，但它的 `min_speed` 同样是 15，所以 5 映射后只有
/// 15% PWM —— 而本车实测"推得动整车"的下限是 26%（速度 20，见 `MIN_MOVING_SPEED`）。
/// 5 只有下限的六成，车原地不动；不动则面积不涨，于是永远卡在 0.70~0.89 出不去。
/// 参考那边多半也没真跑进过制动区。
///
/// 取 22：高于下限且留一点余量，又明显低于趋近档 25，两级才还有意义。
const BUCKET_BRAKE_SPD: i32 = 22;
/// 实测能推得动整车的最低速度档。低于它 PWM 过不了静摩擦，车"看着在下令、不动"
/// （来源：`robot.rs` 接近速度分级那段的实测结论，值 20 → 26% PWM）。
const MIN_MOVING_SPEED: i32 = 20;

/// 编译期护栏：两级趋近速度都必须高于"下单但车不动"的下限。
///
/// 踩过这个坑：参考的制动档 5 映射后只有 15% PWM，车在制动区里根本不动，而面积又不会
/// 再涨，于是永远卡在 0.70~0.89 —— 场上表现为"靠近桶就站住"。约束写死在这里，以后谁
/// 把这两个值调回去都过不了编译，不用等跑测试。
const _: () = {
    assert!(BUCKET_BRAKE_SPD >= MIN_MOVING_SPEED);
    assert!(BUCKET_APPROACH_SPD >= MIN_MOVING_SPEED);
    // 但仍要比趋近档慢，否则分两级没意义。
    assert!(BUCKET_BRAKE_SPD < BUCKET_APPROACH_SPD);
};
/// 转向增益：bias = K × 偏差 / 半宽，夹在 ±MAX_BIAS。
const BUCKET_K_TURN: f32 = 20.0;
const BUCKET_MAX_BIAS: i32 = 8;
/// 中心死区（全分辨率像素）。注意它比追球的死区紧得多：桶是大目标，不需要亚像素级对准。
const BUCKET_CENTER_DEAD_ZONE: i32 = 15;
/// 找不到桶时原地右转的速度。**故意不用参考的 12**，见文件头注释。
const BUCKET_SEARCH_SPD: i32 = 24;
/// 趋近中连续多少帧看不到桶 → 退回搜索。
const BUCKET_LOST_FRAMES: i32 = 10;
/// 连续多少帧看到桶 → 确认（防抖）。
const BUCKET_CONFIRM_CNT: i32 = 3;
/// 放球前的硬刹时长。参考在停车前整段 `motor.brake()`，我们分帧重发。
const BUCKET_BRAKE_MS: u64 = 350;
/// 松爪后等舵机走完，再让机械臂归位。
const BUCKET_DEPOSIT_MS: u64 = 500;
/// 归位后等一小会儿再回追球，别让底盘在手臂还在摆的时候就起步。
const BUCKET_HOME_MS: u64 = 100;
/// 连通域的最小面积（**全分辨率像素数**，色度网格上按采样率折算）。
/// 公开是因为 `capture --bucket` 标定时要用同一口径。
pub const BUCKET_MIN_AREA_FULLRES: i32 = 1000;

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
/// 落在哪 —— 比如"均值 H=25"就说明窗口该往橙红挪，"均值 H=215"说明 `BUCKET_HUE_MIN`
/// 放窄了。**它故意不走 `chroma_excludes_target` 那道早退**：那道是给"当前目标色"
/// 用的，而标定恰恰要知道"当前目标色之外还有什么颜色"。
///
/// 用法是把桶怼近相机再跑 `capture --bucket`：均值会被画面里所有饱和色一起拉动，
/// 桶占满画面时才等于桶的颜色。它只用来读数定阈值，不参与控制。
///
/// 注意均值是跨 0/360 卷绕取的：目标色（蓝，240）离接缝远，量蓝色很准；
/// 若改回红色（0/360 正好是接缝）这里会被拉偏，只能当粗参考。
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
    let (mut hs, mut ss, mut vs, mut n) = (0i64, 0i64, 0i64, 0usize);
    for cy in 0..layout.ch as usize {
        for cx in 0..layout.cw as usize {
            let (h, s, v) = sample_hsv(frame, w, &layout, cx, cy);
            // 和 is_target_hue 用同一条饱和度/亮度下限，只是**不卡色相** ——
            // 这正是标定要看的东西:阈值卡掉的那些像素到底是什么色相。
            if s < 80 || v < 50 {
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

/// 该采样点的 Cb 是否低到不可能是目标色（蓝）。
///
/// 无损早退：蓝色（H∈[180,300)）对应 `rgb2hsv` 里 `cmax == B` 那一支，而
/// cmax == B 时 `Cb - 128 = 0.5B - 0.169R - 0.331G ≥ 0` 恒成立（最坏情况 R=G=B，
/// 那是灰、S=0，本来就过不了判据）。所以 Cb 明显低于 128 的像素（绿/红/黄背景）
/// 一定不是蓝，不必走完 RGB→HSV。120 是给定点换算留的舍入余量。
///
/// **改颜色时这条要跟着改**：红色看 Cr（V 平面）、蓝色看 Cb（U 平面）。
/// 判据改错不会漏检（只是丢掉早退、变慢），但会白跑全路径。
fn chroma_excludes_target(frame: &[u8], layout: &ChromaLayout, cx: usize, cy: usize) -> bool {
    let idx = cy * layout.cw as usize + cx;
    frame[layout.u_off + idx] < 120
}

/// 逐采样点判目标色，生成 0/1 掩膜。
fn build_color_mask(frame: &[u8], width: i32, layout: &ChromaLayout) -> Vec<u8> {
    let w = width as usize;
    let mut mask = vec![0u8; layout.sample_count()];
    for cy in 0..layout.ch as usize {
        for cx in 0..layout.cw as usize {
            if chroma_excludes_target(frame, layout, cx, cy) {
                continue;
            }
            let (h, s, v) = sample_hsv(frame, w, layout, cx, cy);
            if is_target_hue(h, s, v) {
                mask[cy * layout.cw as usize + cx] = 1;
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
/// 它不是教科书写法，但阈值是针对它的输出调出来的 —— 改了它，`is_target_hue` 的边界就挪了。
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

/// 目标颜色的色相窗口（闭区间）。在这个 `rgb2hsv` 约定下：
/// 红=0/360、黄=60、绿=120、青=180、纯蓝=240、品红=300。
///
/// **别按"240 两边各放 20"来取**。这个约定下 R=0 时 `H = 60×(4 - G/B)`，
/// 只要蓝里带一点绿，色相就往下漂 —— 真实的蓝桶（深蓝 R=0/G=90/B=170）落在 208，
/// 浅蓝 (100,160,220) 落在 210，紫蓝 (80,60,200) 才到 248。取 [220,260] 会把
/// 最常见的几种蓝全漏掉。
///
/// 所以用 OpenCV 惯例的蓝色区间：0-180 制的 H∈[100,130]，换算到本约定就是 [200,260]。
/// 它覆盖上述所有蓝，同时把青(180)挡在外面。
///
/// 对着真桶用 `akars capture --bucket` 读实际 H（会打 `mean H=..`）再定窗口。
pub const BUCKET_HUE_MIN: i32 = 200;
pub const BUCKET_HUE_MAX: i32 = 260;

/// 目标色判据：饱和度/亮度下限 + 色相落在窗口内。下限沿用参考（S≥80, V≥50）。
///
/// 注意窗口形式是"直窗口"，因为蓝色（240）离 0/360 接缝足够远。
/// **若改回红色，必须写成回绕形式**（`h <= 20 || h >= 340`）——红色正好横跨接缝，
/// 用直窗口 [340, 20] 会一个字都匹配不上。
///
/// 网球（H 约 55~70）安全落在蓝窗之外 —— "手里夹着的那颗球"不可能被当成桶。
fn is_target_hue(h: i32, s: i32, v: i32) -> bool {
    if s < 80 || v < 50 {
        return false;
    }
    (BUCKET_HUE_MIN..=BUCKET_HUE_MAX).contains(&h)
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
    let half_w = half_w.max(1);
    let off = cx - half_w;
    let bias = if off.abs() <= BUCKET_CENTER_DEAD_ZONE {
        0
    } else {
        (BUCKET_K_TURN * off as f32 / half_w as f32) as i32
    };
    let bias = bias.clamp(-BUCKET_MAX_BIAS, BUCKET_MAX_BIAS) * polarity;
    let spd = if area_ratio >= BUCKET_AREA_BRAKE {
        BUCKET_BRAKE_SPD
    } else {
        BUCKET_APPROACH_SPD
    };
    ((spd + bias).clamp(-100, 100), (spd - bias).clamp(-100, 100))
}

// ── 状态机 ────────────────────────────────────────────────────────────────

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
    /// 机械臂归位中，之后放行回追球。
    DepositHome { until: Instant },
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
                    until: now + Duration::from_millis(BUCKET_DEPOSIT_MS),
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
                    until: now + Duration::from_millis(BUCKET_HOME_MS),
                };
                BucketOutcome::Busy
            }

            BucketPhase::DepositHome { until } => {
                motor.standby();
                if now < until {
                    return BucketOutcome::Busy;
                }
                self.phase = BucketPhase::Idle;
                self.confirm = 0;
                self.lost = 0;
                if bucket_log() {
                    eprintln!("[bucket] done -> back to chase");
                }
                BucketOutcome::Finished
            }

            BucketPhase::Find => {
                match detect_bucket(frame.pixels, width, height, planar, BUCKET_MIN_AREA_FULLRES) {
                    Some(r) => {
                        self.confirm += 1;
                        self.lost = 0;
                        // 确认期内站定不动，别边转边确认。
                        motor.standby();
                        if self.confirm >= BUCKET_CONFIRM_CNT {
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
                                self.confirm, BUCKET_CONFIRM_CNT, r.area_ratio
                            );
                        }
                    }
                    None => {
                        self.confirm = 0;
                        self.lost += 1;
                        // 参考固定右转；不按"最后看到的一侧"回摆，因为桶可能从未出现过。
                        // 轮速过极性镜像后，日志里的"右转"始终指物理右转，与丢球搜索一致。
                        let p = turn_polarity();
                        motor.drive(p * BUCKET_SEARCH_SPD, -p * BUCKET_SEARCH_SPD);
                        if bucket_log() {
                            eprintln!("[bucket] find: searching right, lost={}", self.lost);
                        }
                    }
                }
                BucketOutcome::Busy
            }

            BucketPhase::Approaching => {
                match detect_bucket(frame.pixels, width, height, planar, BUCKET_MIN_AREA_FULLRES) {
                    None => {
                        self.lost += 1;
                        if self.lost > BUCKET_LOST_FRAMES {
                            self.phase = BucketPhase::Find;
                            self.confirm = 0;
                            motor.standby();
                        } else {
                            // 短暂丢失：站住等它回来，别接着往前冲。
                            motor.standby();
                        }
                        if bucket_log() {
                            eprintln!(
                                "[bucket] approach: lost {}/{}",
                                self.lost, BUCKET_LOST_FRAMES
                            );
                        }
                    }
                    Some(r) => {
                        self.lost = 0;
                        if r.area_ratio >= BUCKET_AREA_DEPOSIT {
                            self.phase = BucketPhase::Braking {
                                until: now + Duration::from_millis(BUCKET_BRAKE_MS),
                            };
                            motor.brake();
                            if bucket_log() {
                                eprintln!(
                                    "[bucket] area={:.3} -> brake {}ms",
                                    r.area_ratio, BUCKET_BRAKE_MS
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

    /// 测试用的"桶"颜色：纯蓝（H=240，落在 `BUCKET_HUE_MIN..=MAX` 正中）。
    /// 换目标色时改这里和 `is_target_hue` 的窗口，其余测试跟着走。
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

    #[test]
    fn target_hue_boundaries_match_the_window() {
        // 窗口 [200, 260]，门槛 S≥80、V≥50。
        assert!(is_target_hue(BUCKET_HUE_MIN, 80, 50), "闭区间左端");
        assert!(is_target_hue(BUCKET_HUE_MAX, 80, 50), "闭区间右端");
        assert!(!is_target_hue(BUCKET_HUE_MIN - 1, 255, 255), "左端外一点");
        assert!(!is_target_hue(BUCKET_HUE_MAX + 1, 255, 255), "右端外一点");
        assert!(!is_target_hue(240, 79, 255), "S 差一点");
        assert!(!is_target_hue(240, 255, 49), "V 差一点");
        // 色相环上的非蓝邻居一个都不许漏进来。青(180)是左边界最近的那个。
        for h in [0, 60, 120, 180, 190, 270, 300, 340] {
            assert!(!is_target_hue(h, 255, 255), "H={h} 不该被判成桶");
        }
        // 网球是黄绿(H≈55~70)—— 手里夹着的那颗球永远不会被判成桶。
        assert!(!is_target_hue(60, 255, 255));
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
            assert!(is_target_hue(h, s, v), "({r},{g},{b}) H={h} 应被判成蓝");
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

        // 归位等够 → 收工,状态清干净。
        assert_eq!(rig.step(&bucket_frame(), t3), BucketOutcome::Finished);
        assert!(!rig.track.is_active());
        assert_eq!(rig.track.confirm_count(), 0);
        assert_eq!(rig.track.lost_count(), 0);
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
