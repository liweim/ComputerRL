#!/usr/bin/env bash
set -euo pipefail

QCOW_PATH="/data1/lwm/projects/ubuntu_osworld/Ubuntu.qcow2"
SOURCE_MAIN="/data1/lwm/projects/ComputerRL/desktop_env/server/main.py"
MOUNT_DIR="/mnt/qcow"
NBD_DEV="/dev/nbd0"

cleanup() {
  set +e
  if mountpoint -q "$MOUNT_DIR"; then
    sudo umount "$MOUNT_DIR"
  fi
  sudo qemu-nbd --disconnect "$NBD_DEV" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[1/8] Validate inputs..."
[ -f "$QCOW_PATH" ] || { echo "ERROR: QCOW not found: $QCOW_PATH"; exit 1; }
[ -f "$SOURCE_MAIN" ] || { echo "ERROR: Source file not found: $SOURCE_MAIN"; exit 1; }

if ! command -v sudo >/dev/null 2>&1; then
  echo "ERROR: sudo not found"
  exit 1
fi

if ! command -v qemu-nbd >/dev/null 2>&1; then
  echo "ERROR: qemu-nbd not found"
  exit 1
fi

echo "[2/8] Stop running osworld containers to release file lock..."
docker ps --filter ancestor=happysixd/osworld-docker --format '{{.ID}}' | xargs -r docker stop >/dev/null || true

echo "[3/8] Prepare nbd device..."
sudo modprobe nbd max_part=8
sudo qemu-nbd --disconnect "$NBD_DEV" >/dev/null 2>&1 || true


echo "[4/8] Attach qcow2..."
sudo qemu-nbd --connect "$NBD_DEV" "$QCOW_PATH"
sudo partprobe "$NBD_DEV" || true
sleep 1

mkdir -p "$MOUNT_DIR"

echo "[5/8] Mount root partition..."
MOUNTED_PART=""
for p in "${NBD_DEV}p3" "${NBD_DEV}p2" "${NBD_DEV}p1"; do
  if [ -b "$p" ]; then
    if sudo mount "$p" "$MOUNT_DIR" 2>/tmp/qcow_mount_err.log; then
      MOUNTED_PART="$p"
      break
    fi
  fi
done

if [ -z "$MOUNTED_PART" ]; then
  echo "ERROR: Failed to mount any partition from $NBD_DEV"
  echo "Mount errors:"
  cat /tmp/qcow_mount_err.log || true
  exit 1
fi

echo "Mounted: $MOUNTED_PART"

echo "[6/8] Locate target main.py in qcow..."
TARGET_MAIN=$(sudo find "$MOUNT_DIR" -type f -path '*/main.py' 2>/dev/null | head -n 1 || true)
if [ -z "${TARGET_MAIN}" ]; then
  echo "ERROR: Could not find */main.py inside qcow mounted at $MOUNT_DIR"
  exit 1
fi

echo "Target: $TARGET_MAIN"

echo "[7/8] Backup and replace..."
BACKUP_PATH="${TARGET_MAIN}.bak_$(date +%Y%m%d_%H%M%S)"
sudo cp "$TARGET_MAIN" "$BACKUP_PATH"
sudo cp "$SOURCE_MAIN" "$TARGET_MAIN"

SRC_SHA=$(sha256sum "$SOURCE_MAIN" | awk '{print $1}')
TGT_SHA=$(sudo sha256sum "$TARGET_MAIN" | awk '{print $1}')

echo "Source sha256: $SRC_SHA"
echo "Target sha256: $TGT_SHA"

if [ "$SRC_SHA" != "$TGT_SHA" ]; then
  echo "ERROR: Checksum mismatch after replace"
  exit 1
fi

echo "[8/8] Sync and done."
sync

echo "SUCCESS"
echo "- QCOW: $QCOW_PATH"
echo "- Mounted partition: $MOUNTED_PART"
echo "- Patched file: $TARGET_MAIN"
echo "- Backup file: $BACKUP_PATH"
