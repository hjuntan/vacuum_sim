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
    print("Available scenes:", [s["id"] for s in scenes])

    # Load scene 01 with a fixed seed
    scene, _ = client.call("load_scene", id="01", seed=42)
    start_pose = scene["start"]
    print(f"Start Pose: {start_pose}")

    # Get map & robot data
    map_info, (png,) = client.call("get_map")
    robot_info, _ = client.call("get_robot_info")

    # Initialize the controller and planner
    controller = Controller(client, robot_info)
    planner = Planner(map_info, png)

    # Get available goals
    goals = scene["goals"]
    if not goals:
        goals, _ = client.call("get_goals")
    print("Goals:", [g["name"] for g in goals])

    if not goals:
        print("Error: No goals found in scene.")
        exit(1)

    target_goal = goals[2]
    goal_pose = target_goal["pose"]
    print(f"Testing path to Goal '{target_goal['name']}' at pose: {goal_pose}")

    # Get map and initialize planner
    map_info, (png,) = client.call("get_map")
    print(f"Map loaded: {map_info['width']}x{map_info['height']} at resolution {map_info['resolution']}m/pixel")

    for idx, target in enumerate(goals):
        # Format: [x, y] or [x, y, yaw]
        goal_name = target["name"]
        goal_pose = target["pose"]  

        target_x, target_y = goal_pose[0], goal_pose[1]
        target_yaw = goal_pose[2] if len(goal_pose) > 2 else None

        print(f"Planning path to goal '{goal_name}' at x={target_x}, y={target_y}")

        # Plan A* path from current robot pose
        robot_state = client.latest_state()
        current_pose = robot_state["pose"]
        path = planner.plan(current_pose, goal_pose)

        if not path:
            print(f"A* failed to find a path to '{goal_name}'. Skipping...")
            continue

        print(f"A* path found with {len(path)} waypoints. Navigating...")

        # Subsample waypoints for smooth stepping (e.g., every 5th waypoint)
        waypoints = path[::5]
        if path[-1] not in waypoints:
            waypoints.append(path[-1])

        planned_success = controller.navigate_path(idx, goal_name, waypoints, goal_pose)

        if planned_success:
            print(f"\nSuccessfully arrived at '{goal_name}'! 🎉 Calling mark_goal...")
            try:
                mark_res, _ = client.call("mark_goal", name=goal_name)
                print(f"Mark goal response: {mark_res}\n")
            except RuntimeError as e:
                print(f"Mark goal note: {e}\n")

    print("####################")
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
