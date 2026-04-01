from realsense import *
import time
import os
import cv2
import numpy as np
import argparse
import yaml

def parse_args():
    ap = argparse.ArgumentParser(description="Eye-to-hand capture with ChArUco.")
    ap.add_argument("-c", "--config", type=str, required=True, help="Path to config.yaml")
    return ap.parse_args()

def load_config(path: str) -> dict:
    """
    Load YAML config and apply defaults for optional sections.

    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg

def main():
    args = parse_args()
    cfg = load_config(args.config)

    color_path = cfg['data_path']
    depth_path = cfg['depth_path']
    
    os.makedirs(color_path,exist_ok=True)
    os.makedirs(depth_path,exist_ok=True)

    device_serial = get_devices()
    resolution = (cfg['width'],cfg['height'])

    rs = RealSense(device_serial[0], resolution, resolution)
    
    print(device_serial)
    rs.start()

    rgb_intrinsics, rgb_coeffs, depth_intrinsics, depth_coeffs = rs.get_intrinsics_raw()
    depth_scale = rs.get_depth_scale()

    # drop the first few frames to allow the camera to warm up
    _, _ = rs.shoot()  
    time.sleep(2)

    while True:
        rgb_image, depth_image = rs.shoot()
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        depth_image = depth_image * depth_scale

        cv2.imshow("color",rgb_image)

        key=cv2.waitKey(1)

        if (key&0xff==ord('s')):
            cv2.imwrite(os.path.join(color_path,'2.png'),rgb_image)
            np.save(os.path.join(depth_path,'2.npy'), depth_image)
            print('img saved!')

        elif(key&0xff==ord('q')):
            break

if __name__ == "__main__":
    main()