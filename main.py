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
from recorder import Recorder

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

    # Initialize and start camera recorder (Task 2)
    recorder = Recorder(client)
    recorder.start()

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

    recorder.stop()

    print("\n####################")
    print("All goals processed!")
    print("####################")

if __name__ == "__main__":
    main()
