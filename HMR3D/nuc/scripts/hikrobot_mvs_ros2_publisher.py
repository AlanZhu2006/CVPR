#!/usr/bin/env python3
from __future__ import annotations

import argparse
from ctypes import byref, c_ubyte, memset, sizeof, POINTER, cast
from pathlib import Path
import sys
from typing import Any


def _add_mvs_python_paths() -> None:
    for path in (
        Path("/opt/MVS/Samples/aarch64/Python/MvImport"),
        Path("/opt/MVS/Samples/arm64/Python/MvImport"),
        Path("/opt/MVS/Samples/64/Python/MvImport"),
    ):
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))


def _set_float(cam: Any, name: str, value: float | None) -> None:
    if value is None:
        return
    ret = cam.MV_CC_SetFloatValue(name, float(value))
    if ret != 0:
        print(f"Warning: failed to set {name}={value}: 0x{ret:x}", flush=True)


def _set_int(cam: Any, name: str, value: int | None) -> None:
    if value is None or int(value) <= 0:
        return
    ret = cam.MV_CC_SetIntValue(name, int(value))
    if ret != 0:
        print(f"Warning: failed to set {name}={value}: 0x{ret:x}", flush=True)


def _set_enum_off(cam: Any, name: str) -> None:
    if hasattr(cam, "MV_CC_SetEnumValueByString"):
        ret = cam.MV_CC_SetEnumValueByString(name, "Off")
        if ret == 0:
            return
    ret = cam.MV_CC_SetEnumValue(name, 0)
    if ret != 0:
        print(f"Warning: failed to set {name}=Off: 0x{ret:x}", flush=True)


class HikRobotCamera:
    def __init__(self, index: int, timeout_ms: int):
        _add_mvs_python_paths()
        try:
            from MvCameraControl_class import (  # type: ignore
                MV_ACCESS_Exclusive,
                MV_CC_DEVICE_INFO,
                MV_CC_DEVICE_INFO_LIST,
                MV_CC_PIXEL_CONVERT_PARAM,
                MV_FRAME_OUT,
                MV_GIGE_DEVICE,
                MV_TRIGGER_MODE_OFF,
                MV_USB_DEVICE,
                MvCamera,
                PixelType_Gvsp_RGB8_Packed,
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to import HikRobot MVS Python bindings. Install MVS and source "
                "HMR3D/nuc/configs/hikrobot_mvs_env.sh first."
            ) from exc

        self._MV_ACCESS_Exclusive = MV_ACCESS_Exclusive
        self._MV_CC_DEVICE_INFO = MV_CC_DEVICE_INFO
        self._MV_CC_DEVICE_INFO_LIST = MV_CC_DEVICE_INFO_LIST
        self._MV_CC_PIXEL_CONVERT_PARAM = MV_CC_PIXEL_CONVERT_PARAM
        self._MV_FRAME_OUT = MV_FRAME_OUT
        self._MV_GIGE_DEVICE = MV_GIGE_DEVICE
        self._MV_TRIGGER_MODE_OFF = MV_TRIGGER_MODE_OFF
        self._MV_USB_DEVICE = MV_USB_DEVICE
        self._MvCamera = MvCamera
        self._PixelType_Gvsp_RGB8_Packed = PixelType_Gvsp_RGB8_Packed
        self.timeout_ms = int(timeout_ms)
        self.index = int(index)
        self.cam = None

    def open(
        self,
        exposure_us: float | None,
        gain: float | None,
        fps: float | None,
        width: int | None = None,
        height: int | None = None,
    ) -> None:
        device_list = self._MV_CC_DEVICE_INFO_LIST()
        ret = self._MvCamera.MV_CC_EnumDevices(self._MV_GIGE_DEVICE | self._MV_USB_DEVICE, device_list)
        if ret != 0:
            raise RuntimeError(f"MV_CC_EnumDevices failed: 0x{ret:x}")
        if device_list.nDeviceNum <= 0:
            raise RuntimeError("No HikRobot MVS camera found")
        if self.index < 0 or self.index >= device_list.nDeviceNum:
            raise RuntimeError(f"Camera index {self.index} out of range; found {device_list.nDeviceNum}")

        cam = self._MvCamera()
        info = cast(device_list.pDeviceInfo[self.index], POINTER(self._MV_CC_DEVICE_INFO)).contents
        ret = cam.MV_CC_CreateHandle(info)
        if ret != 0:
            raise RuntimeError(f"MV_CC_CreateHandle failed: 0x{ret:x}")
        ret = cam.MV_CC_OpenDevice(self._MV_ACCESS_Exclusive, 0)
        if ret != 0:
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"MV_CC_OpenDevice failed: 0x{ret:x}")
        cam.MV_CC_SetEnumValue("TriggerMode", self._MV_TRIGGER_MODE_OFF)
        if width is not None or height is not None:
            _set_int(cam, "OffsetX", 0)
            _set_int(cam, "OffsetY", 0)
            _set_int(cam, "Width", width)
            _set_int(cam, "Height", height)
        if exposure_us is not None:
            _set_enum_off(cam, "ExposureAuto")
        if gain is not None:
            _set_enum_off(cam, "GainAuto")
        _set_float(cam, "ExposureTime", exposure_us)
        _set_float(cam, "Gain", gain)
        if fps is not None:
            cam.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True)
            _set_float(cam, "AcquisitionFrameRate", fps)
        ret = cam.MV_CC_StartGrabbing()
        if ret != 0:
            cam.MV_CC_CloseDevice()
            cam.MV_CC_DestroyHandle()
            raise RuntimeError(f"MV_CC_StartGrabbing failed: 0x{ret:x}")
        self.cam = cam

    def read_rgb(self) -> tuple[bytes, int, int]:
        if self.cam is None:
            raise RuntimeError("Camera is not open")
        frame = self._MV_FRAME_OUT()
        memset(byref(frame), 0, sizeof(frame))
        ret = self.cam.MV_CC_GetImageBuffer(frame, self.timeout_ms)
        if ret != 0:
            raise RuntimeError(f"MV_CC_GetImageBuffer failed: 0x{ret:x}")
        try:
            info = frame.stFrameInfo
            width = int(info.nWidth)
            height = int(info.nHeight)
            dst_size = width * height * 3
            dst = (c_ubyte * dst_size)()
            convert = self._MV_CC_PIXEL_CONVERT_PARAM()
            memset(byref(convert), 0, sizeof(convert))
            convert.nWidth = width
            convert.nHeight = height
            convert.pSrcData = frame.pBufAddr
            convert.nSrcDataLen = int(info.nFrameLen)
            convert.enSrcPixelType = info.enPixelType
            convert.enDstPixelType = self._PixelType_Gvsp_RGB8_Packed
            convert.pDstBuffer = dst
            convert.nDstBufferSize = dst_size
            ret = self.cam.MV_CC_ConvertPixelType(convert)
            if ret != 0:
                raise RuntimeError(
                    f"MV_CC_ConvertPixelType to rgb8 failed: 0x{ret:x}, "
                    f"source_pixel_type=0x{int(info.enPixelType):x}"
                )
            return bytes(dst), width, height
        finally:
            self.cam.MV_CC_FreeImageBuffer(frame)

    def close(self) -> None:
        if self.cam is None:
            return
        try:
            self.cam.MV_CC_StopGrabbing()
            self.cam.MV_CC_CloseDevice()
        finally:
            self.cam.MV_CC_DestroyHandle()
            self.cam = None


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish HikRobot MVS camera frames as ROS2 Image.")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--image-topic", default="/hikrobot/image_raw")
    parser.add_argument("--camera-info-topic", default="/hikrobot/camera_info")
    parser.add_argument("--frame-id", default="hikrobot_camera")
    parser.add_argument("--timeout-ms", type=int, default=2000)
    parser.add_argument("--exposure-us", type=float, default=None)
    parser.add_argument("--gain", type=float, default=None)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=0)
    parser.add_argument("--height", type=int, default=0)
    parser.add_argument("--fx", type=float, default=0.0)
    parser.add_argument("--fy", type=float, default=0.0)
    parser.add_argument("--cx", type=float, default=0.0)
    parser.add_argument("--cy", type=float, default=0.0)
    args = parser.parse_args()

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import CameraInfo, Image

    rclpy.init()
    node = Node("hikrobot_mvs_publisher")
    image_pub = node.create_publisher(Image, args.image_topic, 4)
    info_pub = node.create_publisher(CameraInfo, args.camera_info_topic, 4)
    camera = HikRobotCamera(args.index, args.timeout_ms)
    camera.open(args.exposure_us, args.gain, args.fps, args.width or None, args.height or None)
    print(f"Publishing HikRobot camera to {args.image_topic}", flush=True)
    try:
        while rclpy.ok():
            data, width, height = camera.read_rgb()
            stamp = node.get_clock().now().to_msg()
            image = Image()
            image.header.stamp = stamp
            image.header.frame_id = args.frame_id
            image.height = height
            image.width = width
            image.encoding = "rgb8"
            image.is_bigendian = False
            image.step = width * 3
            image.data = data
            image_pub.publish(image)

            info = CameraInfo()
            info.header = image.header
            info.height = height
            info.width = width
            fx = float(args.fx or width)
            fy = float(args.fy or width)
            cx = float(args.cx or width / 2.0)
            cy = float(args.cy or height / 2.0)
            info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            info_pub.publish(info)
            rclpy.spin_once(node, timeout_sec=0.0)
    finally:
        camera.close()
        node.destroy_node()
        rclpy.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
