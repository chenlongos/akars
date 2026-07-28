//! SG2002 pin-mux configuration via the kernel's `/dev/pinmux` device.
//!
//! On this board the non-console UARTs (UART1/2/3) are enabled in the device
//! tree but have **no** pinctrl, so their TX/RX pads default to another
//! function and the port is electrically silent even though `/dev/ttySN`
//! opens fine. The kernel exposes `/dev/pinmux`, which writes the SoC FMUX
//! registers (base `0x0300_1000`) from user space. Its text interface takes
//! one `"0xOFFSET VALUE"` line per pad; offsets are relative to the FMUX base.
//!
//! UART1 is muxed onto the JTAG pads (function select = 4):
//!   - `JTAG_CPU_TMS` @ FMUX 0x64 -> UART1_TX
//!   - `JTAG_CPU_TCK` @ FMUX 0x68 -> UART1_RX
//! Configuring this consumes the JTAG pins (debug JTAG becomes unavailable).

use std::io::{self, Write};

const PINMUX_DEVICE: &str = "/dev/pinmux";

// FMUX register offsets (relative to 0x0300_1000) and the UART function select.
// On the JTAG pads the UART1 TX/RX function is FSEL=6 (FSEL=4 there is the
// RTS/CTS flow-control function, not TX/RX).
const FMUX_JTAG_CPU_TMS: u32 = 0x64;
const FMUX_JTAG_CPU_TCK: u32 = 0x68;
const FSEL_UART1: u32 = 6;

// UART2 TX/RX on the IIC0_SCL / IIC0_SDA pads (FSEL=2).
const FMUX_IIC0_SCL: u32 = 0x70;
const FMUX_IIC0_SDA: u32 = 0x74;
const FSEL_UART2: u32 = 2;

/// Write one FMUX register through `/dev/pinmux` using its text interface.
///
/// The whole `"0xOFFSET VALUE"` line must reach the device in a single write:
/// the kernel parses each write independently, so `write!`/`write_fmt` (which
/// emits one syscall per format fragment) would deliver partial tokens and be
/// rejected with EINVAL. Format first, then one `write_all`.
fn write_fmux(offset: u32, value: u32) -> io::Result<()> {
    let mut dev = std::fs::OpenOptions::new()
        .write(true)
        .open(PINMUX_DEVICE)?;
    let line = format!("0x{offset:X} {value}");
    dev.write_all(line.as_bytes())?;
    Ok(())
}

/// Route the JTAG pads to UART1 TX/RX so `/dev/ttyS1` can actually drive the
/// ESP32 base controller. Safe to call once at startup before opening the port.
pub fn configure_uart1() -> io::Result<()> {
    write_fmux(FMUX_JTAG_CPU_TMS, FSEL_UART1)?;
    write_fmux(FMUX_JTAG_CPU_TCK, FSEL_UART1)?;
    Ok(())
}

/// Route the IIC0 pads to UART2 TX/RX so `/dev/ttyS2` can drive the arm servo
/// controller. The IIC0 bus becomes unavailable after this.
pub fn configure_uart2() -> io::Result<()> {
    write_fmux(FMUX_IIC0_SCL, FSEL_UART2)?;
    write_fmux(FMUX_IIC0_SDA, FSEL_UART2)?;
    Ok(())
}

/// Configure pin-mux for whichever UART backs `device`, based on the `ttySN`
/// alias. UART1 (`ttyS1`) and UART2 (`ttyS2`) are handled; other ports are
/// left untouched with a warning so a mistargeted path is visible.
pub fn configure_for_device(device: &str) {
    if device.ends_with("ttyS1") {
        match configure_uart1() {
            Ok(()) => eprintln!("[pinmux] UART1 pads routed (JTAG TMS/TCK -> UART1 TX/RX)"),
            Err(err) => eprintln!("[pinmux] failed to configure UART1 via {PINMUX_DEVICE}: {err}"),
        }
    } else if device.ends_with("ttyS2") {
        match configure_uart2() {
            Ok(()) => eprintln!("[pinmux] UART2 pads routed (IIC0 SCL/SDA -> UART2 TX/RX)"),
            Err(err) => eprintln!("[pinmux] failed to configure UART2 via {PINMUX_DEVICE}: {err}"),
        }
    } else {
        eprintln!("[pinmux] no pinmux profile for {device}; assuming pads already muxed");
    }
}
