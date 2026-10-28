#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
BINARY="$ROOT/target/riscv64gc-unknown-linux-musl/release/akars"
DEST="root/akars"
# 底盘/机械臂运动参数配置文件:优先用仓库里的 akars.toml,退回示例文件。
CONFIG_SRC="$ROOT/akars.toml"
[[ -f "$CONFIG_SRC" ]] || CONFIG_SRC="$ROOT/akars.example.toml"
CONFIG_DEST="root/akars.toml"
FORCE_CONFIG=0
ORIGINAL_ARGS=("$@")

usage() {
  cat <<'USAGE'
Usage: scripts/upload.sh [--force-config] DEVICE

Mount DEVICE at a temporary directory and copy the SG2002 akars binary to
/root/akars on that filesystem.

Also copies the motion-parameter config (akars.toml, falling back to
akars.example.toml) to /root/akars.toml. An existing on-board config is
preserved unless --force-config is given, so tuned values survive re-uploads.

Example:
  scripts/upload.sh /dev/sda2
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

warn() {
  echo "warning: $*" >&2
}

if [[ $# -eq 1 && ( "$1" == "-h" || "$1" == "--help" ) ]]; then
  usage
  exit 0
fi

# Accept an optional --force-config flag alongside the single DEVICE argument.
DEVICE_ARGS=()
for arg in "$@"; do
  case "$arg" in
    --force-config) FORCE_CONFIG=1 ;;
    *) DEVICE_ARGS+=("$arg") ;;
  esac
done

if [[ ${#DEVICE_ARGS[@]} -ne 1 ]]; then
  usage >&2
  exit 2
fi

DEVICE="${DEVICE_ARGS[0]}"

[[ -f "$BINARY" ]] || die "binary not found: $BINARY; run scripts/build.sh first"
[[ -e "$DEVICE" ]] || die "device not found: $DEVICE"

if [[ ${EUID:-$(id -u)} -ne 0 ]]; then
  command -v sudo >/dev/null 2>&1 || die "sudo is required to mount $DEVICE"
  exec sudo -- "$0" "${ORIGINAL_ARGS[@]}"
fi

MOUNT_DIR=""
MOUNTED=0

cleanup() {
  local status=$?
  if [[ "$MOUNTED" -eq 1 ]]; then
    umount "$MOUNT_DIR" || warn "failed to unmount $MOUNT_DIR during cleanup"
  fi
  if [[ -n "$MOUNT_DIR" ]]; then
    rmdir "$MOUNT_DIR" || warn "failed to remove temporary mount directory: $MOUNT_DIR"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

MOUNT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/akars-upload.XXXXXX")" || die "failed to create temporary mount directory"

mount "$DEVICE" "$MOUNT_DIR" || die "failed to mount $DEVICE at $MOUNT_DIR"
MOUNTED=1

install -D -m 0755 "$BINARY" "$MOUNT_DIR/$DEST" || die "failed to copy $BINARY to $MOUNT_DIR/$DEST"

# Ship the motion config without clobbering on-board tuning unless asked.
if [[ -f "$MOUNT_DIR/$CONFIG_DEST" && "$FORCE_CONFIG" -eq 0 ]]; then
  warn "keeping existing /$CONFIG_DEST on the board (use --force-config to overwrite)"
elif [[ -f "$CONFIG_SRC" ]]; then
  install -D -m 0644 "$CONFIG_SRC" "$MOUNT_DIR/$CONFIG_DEST" \
    || warn "failed to copy $CONFIG_SRC to $MOUNT_DIR/$CONFIG_DEST"
else
  warn "config source not found: $CONFIG_SRC"
fi

sync "$MOUNT_DIR" || sync || warn "sync failed"

umount "$MOUNT_DIR" || die "failed to unmount $MOUNT_DIR"
MOUNTED=0
rmdir "$MOUNT_DIR" || warn "failed to remove temporary mount directory: $MOUNT_DIR"
MOUNT_DIR=""
trap - EXIT INT TERM

echo "uploaded $BINARY to $DEVICE:/$DEST"
