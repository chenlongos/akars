# akars

Rust rewrite entry point for AKA-00 hardware control on SG2002.

Implemented scope:

- USB camera capture from `/dev/cvi-usb-camera0` using the StarryOS ioctl protocol.
- MJPEG decode and letterbox preprocessing through a pure Rust image pipeline.
- YOLOv8 CVI runtime inference and postprocess from `aka-sg2002/detect.*`.
- UART motor protocol from `aka-rk3588/motor/uart_motor_driver.*`.
- ZP10D arm UART protocol from `aka-rk3588/arm`.
- Tennis chasing and grab state machine from `aka-sg2002/tennis.cpp`.

`AKA-00/demo` and shell scripts are intentionally not rewritten here.

Host builds compile the pure Rust pieces and use a TPU stub. SG2002 builds link
only the TPU SDK runtime libraries for `riscv64gc-unknown-linux-musl`.

Build for SG2002:

```bash
cd akars
scripts/setup.sh
scripts/build.sh
```

`scripts/setup.sh` installs the Rust target, initializes the TPU SDK submodule,
downloads the Xuantie V3.4.0 musl toolchain, verifies its SHA-256, and extracts
it to `toolchains/xuantie-v3.4.0`. The TPU SDK lives at
`toolchains/tpu-sdk-sg200x` as a git submodule.

For an offline setup, provide the toolchain archive explicitly:

```bash
scripts/setup.sh --archive /path/to/Xuantie-900-gcc-linux-6.6.36-musl64-x86_64-V3.4.0-20260323.tar.gz
```

Override local paths when needed:

```bash
AKARS_TOOLCHAIN_DIR=/path/to/xuantie-v3.4.0 \
AKARS_TPU_SDK_DIR=/path/to/tpu-sdk-sg200x \
scripts/build.sh
```

Build facts:

- Target: `riscv64gc-unknown-linux-musl`. The AKA-00 board runs a musl rootfs,
  not glibc and not bare-metal `-none-elf`.
- Rust uses the prebuilt target `std`; nightly and `-Zbuild-std` are not needed.
- `.cargo/config.toml` disables static CRT and requests the board loader:
  `/lib/ld-musl-riscv64v0p7_xthead.so.1`.
- `scripts/linker.sh` uses the Xuantie V3.4.0 GCC driver directly; its GNU ld
  links current Rust output successfully.

The output binary is `target/riscv64gc-unknown-linux-musl/release/akars`. On the
device it needs `libcviruntime.so`, `libcvikernel.so`, and the matching
`libstdc++.so.6` reachable via `LD_LIBRARY_PATH`.

Upload the binary to a rootfs partition device, for example an SD card second
partition:

```bash
scripts/upload.sh /dev/sda2
```

This also copies the motion-parameter config (`akars.toml`, falling back to
`akars.example.toml`) to `/root/akars.toml`. An existing on-board config is kept
unless you pass `--force-config`, so tuned values survive re-uploads.

Run on device:

```bash
./akars /path/to/yolo_model.cvimodel \
  --camera /dev/cvi-usb-camera0 \
  --motor /dev/ttyS3 \
  --arm /dev/ttyS2
```

Useful options:

- `--frames N`: stop after N frames for smoke tests.
- `--conf X`: confidence threshold, default `0.5`.
- `--iou X`: NMS IOU threshold, default `0.5`.
- `--classes N`: class count, default `1`.
- `--config PATH`: motion-parameter config, default `/root/akars.toml`.

## Motion parameter tuning

The chassis (chasing, lane-keeping, turn pulses, approach-speed grading, search
and back-off), the grab state machine (distance thresholds, confirm frames, per-
phase durations), the arm (servo pose angles, pulse calibration, grasp timings),
the bucket chase (target colour, approach speeds, find/deposit timings) and the
motor mapping (`speed_scale`, `min_speed`, `ppr`, `pwm_freq`) are read
from a TOML file at startup instead of being compiled in. Edit the file and
restart the program — no rebuild needed.

- Default path is `/root/akars.toml` (next to the `/root/akars` binary);
  override with `--config PATH`.
- `akars.example.toml` in this directory lists every key with a one-line
  description and the values calibrated on a real car. They are a starting
  point, not neutral constants — servo angles, grab distances and the turn
  polarity depend on the individual arm/claw mounting, so re-check them per
  vehicle. Copy it to `/root/akars.toml` and edit.
- The whole file, any `[section]`, or any single key may be omitted; a missing
  value falls back to the compiled-in default declared in `src/config.rs`, which
  mirrors the `pub(crate) const` values in `src/robot.rs` and `src/arm.rs`. Those
  defaults are not repeated in the example file, so the numbers written there are
  purely the tuned ones. A missing or unparseable file only prints a warning and
  runs entirely on defaults.
- Inference thresholds (`--conf` / `--iou` / `--classes`) and device paths stay
  on the command line; only the motion parameters above are externalized.
- `[chassis] turn_reverse = true` mirrors the left/right steering polarity for every
  closed-loop differential on the chassis — the chase correction, the lost-ball
  search turn, the bucket-approach proportional steer and the bucket search spin. It is
  meant for boards whose two motor channels are wired to the opposite sides (a
  "left" command physically turns right). Chasing is normally closed-loop (it
  corrects toward the ball) but still inherits the mirror so all paths stay
  consistent. Use `false` for a normally wired car instead of rebuilding.
- `[grab] grab_turn_direction = "left" | "right"` sets the open-loop pre-grab turn
  side on its own, independent of `turn_reverse`. Because the claw sits off-center
  from the camera and that offset side varies with how the arm is mounted, different
  cars need to turn toward different sides to bring the ball in front of the claw,
  so this is an explicit per-car calibration value rather than something derived from
  the wiring polarity.
- **Open-loop chassis motions are wall-clock timed, never frame-stepped.** The
  pre-grab blind turn is `grab_turn_count` pulses of `grab_turn_ms`, separated and
  followed by `grab_turn_gap_ms`; the back-off is one reverse pulse of
  `grab_backoff_ms`. Both run to completion inside a single call and brake
  immediately at the end, instead of being advanced one frame at a time. They used
  to be stepped by the frame loop, which let the frame period (~60ms, and much
  longer whenever the camera hiccups) quantize the pulse: the same config would
  sometimes deliver half a pulse and sometimes sweep the whole motion in one go, so
  the claw landed in different places and the reverse pulse occasionally backed
  much further than intended. Braking at the end of every pulse also stops it from
  inheriting the previous one's residual spin. The motion per grab is now
  repeatable; retune `grab_turn_ms` / `grab_turn_count` if the claw starts stopping
  short, since the pulses no longer carry that coast over.
- After the back-off pulse the car stays still for a short **settle window** before
  the distance is re-judged, so the EMA has taken in fresh post-back-off frames;
  judging immediately would re-use the pre-back-off (too-close) reading and back off
  forever. That window never moves the chassis, so its frame-based timing cannot
  turn into displacement.
- `[bucket] chase_bucket = true | false` decides whether a successful grab hands over
  to the "chase the bucket -> deposit the ball" loop (`true`, the default) or the
  claw just releases in place and the car chases the next ball (`false`). It used to
  live in `[grab]`; a stale `[grab] chase_bucket` is now ignored, so move it into
  `[bucket]`. For quick on-car debugging the `AKARS_NO_BUCKET` environment variable
  force-disables the bucket loop without editing the config file.
- `[bucket]` holds everything the bucket chase itself needs, so a colour or
  distance retune never needs a rebuild: the target colour as a **name**
  (`bucket_color = "blue" | "red" | "green"`, blue by default), the blob
  `min_area_fullres`, the `confirm_frames` / `lost_frames` de-bounce counters, the
  `search_speed` spin, the two-stage approach (`approach_speed`, `brake_speed`,
  `area_brake`, `area_deposit`), the proportional steer (`k_turn`, `max_bias`,
  `center_dead_zone`) and the deposit timings (`brake_ms`, `deposit_ms`, `home_ms`,
  `deposit_backoff_ms`). `deposit_backoff_ms` is a short straight reverse run right
  after the arm returns home: the car stops practically against the bucket rim, and
  the very next thing it does is spin in place to search for a ball, which sweeps the
  chassis into the bucket — backing off first clears it. Unlike the pre-grab back-off
  (a wall-clock pulse that runs to completion inside one call, because it has to stop
  the claw at a precise distance) this one only has to get clear of the rim, so it is
  an ordinary frame-stepped timed phase at `[chassis] backward_speed`: a frame of
  slop does not matter and the camera loop keeps running. `0` disables it.
  The hue window, the saturation/value floors and the chroma-plane fast reject behind
  each colour name are **not** in the file: they live in the preset table in
  `src/bucket.rs`, because a hand-tuned H/S/V window is exactly how you
  lose a blue bucket (most real blues sit near H=210, not at 240). Add a colour
  there if you need one; yellow is deliberately absent since the tennis ball is
  yellow-green. Two guardrails that used to be compile-time assertions now run at
  load time: both approach speeds are raised to the minimum speed that actually
  moves the car (`>= 20`), and `brake_speed` is capped at `approach_speed` — the
  reference's creep speed of 5 maps to ~15% PWM, the car never moves in the brake
  band, the area stops growing and the car parks forever at `area_brake`.
  `akars capture --bucket` reads this file too and prints the colour it is judged by
  (including the mean H of the saturated pixels in view), so you can tell which name
  fits the real bucket.
- **A grab only counts if the claw actually closed on the ball.** After the full
  grab sequence — including the lift — `arm.grab()` reads the claw position back
  over the servo bus (ZP10S `#002PRAD!` -> `#002P1500!`) and compares it with the
  close target: a held ball blocks the jaws, so the servo stalls clearly above
  `[arm] id2_angle_close`; an empty close walks all the way to it. The read happens
  **after the lift**, not right after closing: closing on the ball is not the same
  as still holding it once the arm is up, and a ball that slips or gets shaken out
  mid-lift lets the (still commanded) jaws close the rest of the way. That is also
  the final holding state the bucket chase inherits. A **missed** grab never enters
  the bucket/deposit stage — the claw reopens, the arm returns to the ready pose and
  the car goes straight back to the chase -> grab loop, so a `chase_bucket = true`
  car can no longer drive off hunting a bucket with an empty claw. Tune the
  empty-close band with `[arm] grab_close_tolerance_deg` (degrees above the close
  angle still counted as empty) on the logged `claw closed to N deg` line; set
  `[arm] grab_verify = false` to skip the read entirely (or when the servo
  firmware does not answer `PRAD`). With no reply the verdict degrades to
  "unknown" and the old keep-going behavior is kept, so a car without position
  feedback keeps working exactly as before.
- The grab trigger compares an **EMA-filtered** ball area ratio (`[grab]
  area_filter_alpha`) against `grab_area` with a **hysteresis band** (`[grab]
  grab_hysteresis_frac`, a fraction of `grab_area`): jitter of the raw YOLO box
  around the threshold line no longer resets the confirmation. The dwell is timed
  in **milliseconds** (`[grab] grab_confirm_ms`), not frames, so the trigger
  distance no longer depends on camera framerate. While confirming, the chassis
  actively brakes (`CMD_BRAKE`) instead of coasting to a stop, and once engaged the
  straight segment stops creeping forward — the car holds the distance it crossed
  the threshold at. `grab_area_max` should stay above `grab_area`; after each
  back-off pulse the car holds still for a settle window and then the distance is
  re-measured from fresh frames: still too close backs off again (up to 2 retries),
  overshot aborts back to chasing, in-band proceeds to the blind turn.
- **Ball-out-of-frame too-close signal**: the ball bbox shows the near-clip
  signature — its bottom edge touches the frame bottom, or the rigid-sphere box
  comes out squashed (width − height > 20px; fully-framed boxes measure within
  a few px, clipped ones tens) — the area readout there is clipped and no
  longer a trustworthy distance. While the grab gate is still disengaged, that
  signal enters the same pulse-based backoff used for "too close" at grab
  start (one wall-clock timed reverse pulse, then a settle window and a
  re-measure); once engaged the
  approach is left alone — the dwell already brakes every frame and hands over
  to the grab sequence within ~200ms, whose too-close branch owns the measured
  back-off. Interrupting mid-dwell would cut every normal coast-in at the
  claw's ~11cm sweet spot and loop chase↔reverse forever.
- Migration: the old `[grab] grab_confirm_threshold` (consecutive-frame count) is
  gone; leftover keys on a card are ignored and the dwell falls back to
  `grab_confirm_ms` (default 300ms, ~= the old 5 frames at 17fps).

### Calibrating grab distances (area ratio -> physical distance)

`grab_area` and friends are ball-bbox fractions of the **raw camera frame**
(640x360 here; the 640x480 letterbox used for inference is mapped back before
the area is computed, so it never enters the denominator). For a ball of fixed
physical size the area follows an inverse-square law, `a ~= (k/z)^2`, with one
car-specific constant `k` that absorbs ball size, lens and mounting. Measure it
with `akars capture` (or the on-screen bbox) at 2-3 known ball-center distances,
fit `k = z*sqrt(a)`, then translate thresholds:

```text
car1 measured (k ~= 7.2-7.5 cm, valid 120cm down to ~11.5cm):
  0.004 -> ~120cm   0.015 -> ~60cm   0.15 -> ~19cm   0.20 -> ~16cm
  0.39 -> ~11.5cm (armtest claw sweet spot)   0.58 -> ~8.4cm (bottom clipped)
```

Pick the engage threshold from the desired claw gap: the car still coasts
several centimeters after forward stops (brake is a coast `CMD_BRAKE`, dwell
is ~200ms), so `engage distance ~= grab distance + coast distance`. On car1
engage 0.20 + ~5cm coast fires the grab right at the armtest sweet spot
(0.39 @ ~11.5cm ball-center distance, leaving roughly 3cm between bumper and
ball). Area values above that spot are unreliable as a distance readout: once
the ball is closer than ~11.5cm its bottom edge leaves the frame, the bbox gets
clipped and stops following `1/z^2` (k drops from ~7.3 to ~6.4 by 0.58). Keep
both the engage threshold and any closed-loop band below the clipping onset;
re-check `k` per vehicle and after mounting changes; `akars motortest` verifies
the chassis start/stop/reverse behavior the coast distance depends on, and
`akars armtest` finds the physical claw sweet spot the engage threshold should
land on after the coast.

Test TPU inference on a single image:

```bash
./akars detect /path/to/yolo_model.cvimodel input.jpg --out result.jpg
```

Runs the model on `input.jpg`, prints each detection (class, score, box), and
writes `result.jpg` with the boxes drawn. Same `--classes` / `--conf` / `--iou`
options as above; `--out` defaults to `detect_out.jpg`. Useful for verifying the
model and runtime independently of the camera and motors.

Web control:

```bash
./akars serve \
  --listen 0.0.0.0:8080 \
  --motor /dev/ttyS3 \
  --arm /dev/ttyS2
```

Open `http://<device-ip>:8080/` to control the base and arm from the built-in frontend.
Use `--mock` to run the same frontend/backend on a development machine without AKA hardware.
