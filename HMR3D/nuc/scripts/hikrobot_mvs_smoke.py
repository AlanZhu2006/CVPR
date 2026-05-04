#!/usr/bin/env python3
from __future__ import annotations

import argparse
from ctypes import byref, memset, sizeof, POINTER, cast
from pathlib import Path
import sys
from typing import Any


def _add_mvs_python_paths() -> None:
    candidates = [
        Path("/opt/MVS/Samples/aarch64/Python/MvImport"),
        Path("/opt/MVS/Samples/arm64/Python/MvImport"),
        Path("/opt/MVS/Samples/64/Python/MvImport"),
    ]
    for path in candidates:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _decode_name(raw: Any) -> str:
    data = bytes(raw)
    return data.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


def main() -> int:
    parser = argparse.ArgumentParser(description="Enumerate HikRobot MVS cameras and grab one frame.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--timeout-ms", type=int, default=2000)
    args = parser.parse_args()

    _add_mvs_python_paths()
    try:
        from MvCameraControl_class import (  # type: ignore
            MV_ACCESS_Exclusive,
            MV_CC_DEVICE_INFO,
            MV_CC_DEVICE_INFO_LIST,
            MV_FRAME_OUT,
            MV_GIGE_DEVICE,
            MV_TRIGGER_MODE_OFF,
            MV_USB_DEVICE,
            MvCamera,
        )
    except Exception as exc:
        raise SystemExit(
            "Failed to import HikRobot MVS Python bindings. Install MVS first and source "
            "HMR3D/nuc/configs/hikrobot_mvs_env.sh. "
            f"Import error: {type(exc).__name__}: {exc}"
        ) from exc

    device_list = MV_CC_DEVICE_INFO_LIST()
    ret = MvCamera.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE, device_list)
    if ret != 0:
        raise SystemExit(f"MV_CC_EnumDevices failed: 0x{ret:x}")
    print(f"Found {device_list.nDeviceNum} HikRobot device(s)")
    if device_list.nDeviceNum <= 0:
        raise SystemExit("No MVS camera found")
    if args.index < 0 or args.index >= device_list.nDeviceNum:
        raise SystemExit(f"--index {args.index} is out of range")

    for i in range(device_list.nDeviceNum):
        info = cast(device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
        if info.nTLayerType == MV_USB_DEVICE:
            usb = info.SpecialInfo.stUsb3VInfo
            print(
                f"[{i}] USB3 model={_decode_name(usb.chModelName)} "
                f"serial={_decode_name(usb.chSerialNumber)}"
            )
        elif info.nTLayerType == MV_GIGE_DEVICE:
            gige = info.SpecialInfo.stGigEInfo
            print(
                f"[{i}] GigE model={_decode_name(gige.chModelName)} "
                f"serial={_decode_name(gige.chSerialNumber)}"
            )
        else:
            print(f"[{i}] layer={info.nTLayerType}")

    cam = MvCamera()
    info = cast(device_list.pDeviceInfo[args.index], POINTER(MV_CC_DEVICE_INFO)).contents
    ret = cam.MV_CC_CreateHandle(info)
    if ret != 0:
        raise SystemExit(f"MV_CC_CreateHandle failed: 0x{ret:x}")
    try:
        ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
        if ret != 0:
            raise SystemExit(f"MV_CC_OpenDevice failed: 0x{ret:x}")
        cam.MV_CC_SetEnumValue("TriggerMode", MV_TRIGGER_MODE_OFF)
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise SystemExit(f"MV_CC_StartGrabbing failed: 0x{ret:x}")

        frame = MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        ret = cam.MV_CC_GetImageBuffer(frame, int(args.timeout_ms))
        if ret != 0:
            raise SystemExit(f"MV_CC_GetImageBuffer failed: 0x{ret:x}")
        try:
            info = frame.stFrameInfo
            print(
                "Grabbed frame: "
                f"{info.nWidth}x{info.nHeight}, len={info.nFrameLen}, "
                f"pixel_type=0x{int(info.enPixelType):x}, frame_num={info.nFrameNum}"
            )
        finally:
            cam.MV_CC_FreeImageBuffer(frame)
        cam.MV_CC_StopGrabbing()
    finally:
        cam.MV_CC_CloseDevice()
        cam.MV_CC_DestroyHandle()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
