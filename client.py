"""Minimal Python client: RPC, map, wheels, goto, state and camera streams.

pip install pyzmq numpy pillow; python client.py [--host localhost]
"""
import argparse
import io
import json
import math
import time

import numpy as np
from planner import Planner
import zmq
from PIL import Image

ap = argparse.ArgumentParser()
ap.add_argument("--host", default="localhost")
ap.add_argument("--port-base", type=int, default=5555)
args = ap.parse_args()


def url(k):
    return f"tcp://{args.host}:{args.port_base + k}"

# Setups for the 4 sockets: RPC, command, state stream, camera stream.
ctx = zmq.Context()
rpc = ctx.socket(zmq.REQ)
rpc.connect(url(0))
cmd = ctx.socket(zmq.PUSH)
cmd.connect(url(1))
state = ctx.socket(zmq.SUB)
state.connect(url(2))
state.setsockopt(zmq.SUBSCRIBE, b"")
camera = ctx.socket(zmq.SUB)
camera.setsockopt(zmq.RCVHWM, 2)
camera.connect(url(3))
camera.setsockopt(zmq.SUBSCRIBE, b"")


def call(method, **params):
    rpc.send_json({"method": method, "params": params})
    frames = rpc.recv_multipart()
    rep = json.loads(frames[0])
    if not rep["ok"]:
        raise RuntimeError(rep["error"])
    return rep["result"], frames[1:]


def latest(sock):
    """Most recent message: drop anything queued (streams are 'latest wins')."""
    msg = sock.recv_multipart()
    while sock.poll(0):
        msg = sock.recv_multipart()
    return msg

# List the available scenes
scenes, _ = call("list_scenes")
print("Available scenes:", [s["id"] for s in scenes])

# Choose scene: 01 for now
scene, _ = call("load_scene", id="01", seed=42)
print("Loaded scene '01'. Start pose:", scene["start"])

# Get available goals
goals = scene["goals"]
if not goals:
    goals, _ = call("get_goals")
print("Goals:", [g["name"] for g in goals])

if not goals:
    print("No goals found in this scene.")
    exit(1)

# Get robot dimensions for calculating wheel speeds during rotation
robot_info, _ = call("get_robot_info")
wheel_base = robot_info["wheel_base"]
wheel_radius = robot_info["wheel_radius"]

# Calculate wheel speed for turning in place (e.g., 1.5 rad/s turn rate)
turn_rate = 1.5
turn_wheel_speed = turn_rate * wheel_base / 2 / wheel_radius

# Get map and initialize planner
map_info, (png,) = call("get_map")
planner = Planner(map_info, png)

for idx, target in enumerate(goals):
    # Format: [x, y] or [x, y, yaw]
    goal_name = target["name"]
    goal_pose = target["pose"]  

    target_x, target_y = goal_pose[0], goal_pose[1]
    target_yaw = goal_pose[2] if len(goal_pose) > 2 else None

    print(f"Planning path to goal '{goal_name}' at x={target_x}, y={target_y}")

    # Plan A* path from current robot pose
    robot_state = json.loads(latest(state)[0])
    current_pose = robot_state["pose"]
    path = planner.plan(current_pose, goal_pose)

    if not path:
        print(f"A* failed to find a path to '{goal_name}'. Skipping...")
        continue

    print(f"A* path found with {len(path)} waypoints. Navigating...")

    # Subsample waypoints for smooth stepping (e.g., every 5th waypoint)
    waypoints = path[::50]
    if path[-1] not in waypoints:
        waypoints.append(path[-1])

    goal_failed = False

    for wp_idx, (wp_x, wp_y) in enumerate(waypoints):
        goto_id = (idx + 1) * 100 + wp_idx + 1  # Unique ID for each waypoint
        is_last_waypoint = (wp_idx == len(waypoints) - 1)

        # Append final target yaw only on the final waypoint if provided
        wyaw = goal_pose[2] if (is_last_waypoint and len(goal_pose) > 2) else None

        payload = {"type": "goto", "id": goto_id, "x": wp_x, "y": wp_y}
        if wyaw is not None:
            payload["yaw"] = wyaw

        cmd.send_json(payload)

        # Wait for the simulator to acknowledge the new goto_id
        acknowledged = False
        start_wait = time.monotonic()
        while not acknowledged:
            robot_state = json.loads(latest(state)[0])
            goto_info = robot_state["goto"]
            if goto_info["id"] == goto_id:
                acknowledged = True
            elif time.monotonic() - start_wait > 2.0:
                cmd.send_json(payload)
                start_wait = time.monotonic()
            time.sleep(0.01)

        # Monitor state stream until the waypoint is reached or bumped
        has_reached_wp = False
        while not has_reached_wp:
            robot_state = json.loads(latest(state)[0])
            goto_info = robot_state["goto"]
            status = goto_info["status"]
            pose = robot_state["pose"]

            print(f"🔄 Pose: x={pose['x']:.2f}, y={pose['y']:.2f} | Goto status: {status}", end="\r")

            if goto_info.get("id") != goto_id:
                time.sleep(0.02)
                continue  # Ignore other goto commands

            if status == "reached":
                print(f"\nReached waypoint {wp_idx + 1}/{len(waypoints)} for '{goal_name}'.")
                has_reached_wp = True
            elif status == "bumped":
                print(f"\nRobot bumped an obstacle while heading to waypoint {wp_idx + 1}/{len(waypoints)}! Executing recovery rotation...")
                
                # Rotate in place for 1.5 seconds to back away / clear the bumper sensor
                recovery_end = time.monotonic() + 1.5
                while time.monotonic() < recovery_end:
                    # Send counter-rotating wheel commands at ~20 Hz to respect watchdog
                    cmd.send_json({"type": "wheels", "left": -turn_wheel_speed, "right": turn_wheel_speed})
                    time.sleep(0.05)
                
                # Stop wheels explicitly
                cmd.send_json({"type": "wheels", "left": 0.0, "right": 0.0})
                print("Recovery rotation complete. Moving to next waypoint...")
                goal_failed = False  # Move on to the next waypoint after recovery
                break

            time.sleep(0.02)  # Poll matching the state stream rate (~50 Hz)

        if goal_failed:
            print(f"Failed to reach waypoint {wp_idx + 1}/{len(waypoints)} for '{goal_name}'. Skipping to next goal...")
            break

    if not goal_failed:
        print(f"\nSuccessfully arrived at '{goal_name}'! 🎉 Calling mark_goal...")
        try:
            mark_res, _ = call("mark_goal", name=goal_name)
            print(f"Mark goal response: {mark_res}\n")
        except RuntimeError as e:
            print(f"Mark goal note: {e}\n")
    
    # print("Goal processing complete for ", goal_name, "!\n")

print("####################")
print("All goals processed!")
print("####################")

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
