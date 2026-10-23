"""Manual test script for the Planner A* pathfinding module.

pip install pyzmq numpy pillow matplotlib; python test_planner.py
"""
import json
import zmq
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from io import BytesIO
from planner import Planner

ctx = zmq.Context()
rpc = ctx.socket(zmq.REQ)
rpc.connect("tcp://localhost:5555")

def call(method, **params):
    rpc.send_json({"method": method, "params": params})
    frames = rpc.recv_multipart()
    rep = json.loads(frames[0])
    if not rep["ok"]:
        raise RuntimeError(rep["error"])
    return rep["result"], frames[1:]

# 1. Load scene 01 with a fixed seed
scene, _ = call("load_scene", id="01", seed=42)
start_pose = scene["start"]
print(f"Start Pose: {start_pose}")

# 2. Get available goals
goals = scene["goals"]
if not goals:
    goals, _ = call("get_goals")
print("Goals:", [g["name"] for g in goals])

if not goals:
    print("Error: No goals found in scene.")
    exit(1)

target_goal = goals[2]
goal_pose = target_goal["pose"]
print(f"Testing path to Goal '{target_goal['name']}' at pose: {goal_pose}")

# 3. Get map and initialize planner
map_info, (png,) = call("get_map")
planner = Planner(map_info, png)
print(f"Map loaded: {map_info['width']}x{map_info['height']} at resolution {map_info['resolution']}m/pixel")

# 4. Run A* planner
path = planner.plan(start_pose, goal_pose)

if not path:
    print("❌ A* Planner test FAILED: No path found!")
    exit(1)

print(f"✅ A* Planner test PASSED! Generated path with {len(path)} waypoints.")
print(f"First 5 waypoints: {path[:5]}") 

# 5. Visualize and save path on top of the map image for inspection
grid_img = Image.open(BytesIO(png)).convert("RGB")

plt.figure(figsize=(8, 8))
plt.imshow(grid_img, cmap="gray", origin="upper")

# Convert path world coordinates back to pixel coordinates for plotting
pixel_path = [planner.world_to_map(wx, wy) for wx, wy in path]
py, px = zip(*pixel_path)
plt.plot(px, py, color="red", linewidth=2, label="A* Path")

# Plot start and goal
start_r, start_c = planner.world_to_map(start_pose["x"], start_pose["y"])
goal_r, goal_c = planner.world_to_map(goal_pose[0], goal_pose[1])
plt.scatter([start_c], [start_r], color="green", s=100, label="Start", zorder=5)
plt.scatter([goal_c], [goal_r], color="blue", s=100, label="Goal", zorder=5)

plt.legend()
plt.title(f"A* Path Test: Scene 01 -> {target_goal['name']}")
plt.savefig("path_debug.png", bbox_inches="tight")
print("Saved visual debugging image to 'path_debug.png'.")
plt.close()
