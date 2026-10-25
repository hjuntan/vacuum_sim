"""Minimal Python client: RPC, map, wheels, goto, state and camera streams.

pip install pyzmq numpy pillow; python client.py [--host localhost]
"""
import argparse
import io
import json
import math
import numpy as np
import time
import zmq

from client import Client
from controller import Controller
from PIL import Image
from planner import Planner

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port-base", type=int, default=5555)
    args = ap.parse_args()

    client = Client(host=args.host, port_base=args.port_base)

    # List the available scenes
    scenes, _ = client.call("list_scenes")
    if not scenes:
        print("Error: No scenes available from the simulator.")
        return
    
    print("--- Available Scenes ---")
    print(" ID |      Name       ")
    for idx, s in enumerate(scenes):
        print(f" {s['id']} | {s['name']} ")

    default_scene_id = scenes[0]["id"]
    scene_choice = input(f"\nSelect a scene index (1-{len(scenes)}) or type the Scene ID [default: {default_scene_id}]: ").strip()

    selected_scene_id = default_scene_id
    if scene_choice:
        if scene_choice.isdigit() and 1 <= int(scene_choice) <= len(scenes):
            selected_scene_id = scenes[int(scene_choice) - 1].get('id', scenes[int(scene_choice) - 1]) if isinstance(scenes[int(scene_choice) - 1], dict) else scenes[int(scene_choice) - 1]
        else:
            # Check if input matches any scene ID directly
            matching = [s for s in scenes if (s.get('id') if isinstance(s, dict) else s) == scene_choice]
            if matching:
                selected_scene_id = scene_choice
            else:
                print(f"⚠️ Invalid choice '{scene_choice}'. Defaulting to scene '{selected_scene_id}'.")

    default_seed = 42
    seed_input = input(f"Enter a seed for the scene (default: {default_seed}): ").strip()

    selected_seed = default_seed
    if seed_input:
        if seed_input.isdigit():
            selected_seed = int(seed_input)
        else:
            print(f"Defaulting to seed {default_seed}.")

    print(f"\nLoading scene '{selected_scene_id}' with seed {selected_seed}...")
    scene, _ = client.call("load_scene", id=selected_scene_id, seed=selected_seed)
    print(f"Start Pose: {scene['start']}")

    # Get map, robot and camera data
    map_info, (png,) = client.call("get_map")
    robot_info, _ = client.call("get_robot_info")
    camera_info, _ = client.call("get_camera_info")

    print(f"Map loaded: {map_info['width']}x{map_info['height']} at resolution {map_info['resolution']}m/pixel")

    # Initialize the controller and planner
    planner = Planner(map_info, png)
    controller = Controller(client, planner, robot_info, camera_info)

    # Get available goals
    goals = scene["goals"]
    if not goals:
        goals, _ = client.call("get_goals")
    print("Goals:", [g["name"] for g in goals])

    if not goals:
        print("Error: No goals found in scene.")
        exit(1)

    formatted_goals = [(idx, target["name"], target["pose"]) for idx, target in enumerate(goals)]

    controller.run_nearest_neighbor_missions(formatted_goals)

    print("\n####################")
    print("All goals processed!")
    print("####################")

if __name__ == "__main__":
    main()

# # Wheels: rad/s per wheel, resend at 20 Hz (250 ms watchdog). Turn in place at 1 rad/s for 1.5 s.
# turn_rate = 1.0
# wheel_speed = turn_rate * robot["wheel_base"] / 2 / robot["wheel_radius"]
# for _ in range(30):
#     cmd.send_json({"type": "wheels", "left": -wheel_speed, "right": wheel_speed})
#     time.sleep(0.05)
# robot_state = json.loads(latest(state)[0])
# print("after turning:", robot_state["pose"], robot_state["wheels"])

# # goto 0.3 m straight ahead, then follow the state stream for 3 s.
# pose = robot_state["pose"]
# x = pose["x"] + 0.3 * math.cos(pose["yaw"])
# y = pose["y"] + 0.3 * math.sin(pose["yaw"])
# cmd.send_json({"type": "goto", "id": 1, "x": x, "y": y})
# end = time.monotonic() + 3.0
# while time.monotonic() < end:
#     robot_state = json.loads(latest(state)[0])
# print("goto:", robot_state["goto"], "pose:", robot_state["pose"], "bumper:", robot_state["bumper"])

# # Camera: header, JPEG, uint16 depth (mm).
# header, jpg, raw = latest(camera)
# header = json.loads(header)
# rgb = np.array(Image.open(io.BytesIO(jpg)))
# depth = np.frombuffer(raw, "<u2").reshape(header["height"], header["width"]) * cam["depth_scale"]

# # Back-project the centre pixel: pixel -> camera frame -> robot frame -> world frame.
# u, v = header["width"] // 2, header["height"] // 2
# z = depth[v, u]
# x_camera = (u - cam["cx"]) * z / cam["fx"]
# y_camera = (v - cam["cy"]) * z / cam["fy"]
# p_camera = np.array([x_camera, y_camera, z, 1.0])
# p_robot = np.array(cam["T_base_camera"]) @ p_camera
# pose = header["pose"]
# cos_yaw, sin_yaw = math.cos(pose["yaw"]), math.sin(pose["yaw"])
# x_world = pose["x"] + cos_yaw * p_robot[0] - sin_yaw * p_robot[1]
# y_world = pose["y"] + sin_yaw * p_robot[0] + cos_yaw * p_robot[1]
# p_world = (x_world, y_world, p_robot[2])
# print(f"camera seq {header['seq']}: rgb {rgb.shape}, centre depth {z:.3f} m -> world {np.round(p_world, 3)}")
