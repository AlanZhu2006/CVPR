import pyrealsense2 as rs


def safe_get_info(device, info_kind):
    """Return a device info string if the field exists."""
    if device.supports(info_kind):
        return device.get_info(info_kind)
    return "n/a"


def main() -> None:
    """List connected RealSense devices and serial numbers."""
    ctx = rs.context()
    devices = list(ctx.devices)

    if not devices:
        print("No RealSense devices detected.")
        print("Connect the camera, then rerun this script to get serial numbers for frame_nano_rig.yaml.")
        return

    print(f"Detected {len(devices)} RealSense device(s):")
    for index, device in enumerate(devices, start=1):
        name = safe_get_info(device, rs.camera_info.name)
        serial = safe_get_info(device, rs.camera_info.serial_number)
        product_line = safe_get_info(device, rs.camera_info.product_line)
        firmware = safe_get_info(device, rs.camera_info.firmware_version)
        print(f"[{index}] {name}")
        print(f"    serial: {serial}")
        print(f"    product_line: {product_line}")
        print(f"    firmware: {firmware}")


if __name__ == "__main__":
    main()
