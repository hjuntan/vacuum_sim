import time
import math
import numpy as np

class Controller:
    def __init__(self, client, robot_info, camera_info=None):
        self.client = client
        self.camera_info = camera_info
        turn_rate = 1.5
        self.turn_wheel_speed = turn_rate * robot_info["wheel_base"] / 2 / robot_info["wheel_radius"]

    def is_path_clear(self, target_distance):
        if not self.camera_info:
            print("Warning: No camera info provided. Assuming path is clear.")
            return True
        try:
            header, jpg, raw = self.client.latest_camera()
            depth = np.frombuffer(raw, "<u2").reshape(header["height"], header["width"]) * self.camera_info["depth_scale"]

            h, w = depth.shape
            center_region = depth[int(h*0.3):int(h*0.7), int(w*0.3):int(w*0.7)]
            valid = center_region[center_region > 0.05]
            if valid.size > 0 and np.min(valid) < min(target_distance, 2.0):
                return False
        except Exception as e:
            pass
        return True

    def execute_goal_mission(self, goal_idx, goal_name, goal_pose, planner, current_pose=None):
        target_x, target_y = goal_pose[0], goal_pose[1]

        print(f"Planning path to goal '{goal_name}' at x={target_x}, y={target_y}")

        dist_to_goal = math.hypot(target_x - current_pose["x"], target_y - current_pose["y"]) if current_pose else None

        # Dual Gatekeeper check
        map_clear = planner.is_line_safe(current_pose, goal_pose) if current_pose else False
        camera_clear = self.is_path_clear(dist_to_goal)

        direct_success = False
        if map_clear and camera_clear:
            direct_success = self.navigate_direct_path(goal_idx, goal_name, goal_pose)

        if direct_success:
            return True  # Indicate that the goal was successfully processed via direct navigation
        
        # A* fallback if direct navigation fails or is not clear
        print(f"Using A* waypoints...")
        return self.navigate_A_star_path(planner, goal_idx, goal_name, goal_pose)


    def navigate_direct_path(self, goal_idx, goal_name, goal_pose):
        target_x, target_y = goal_pose[0], goal_pose[1]
        target_yaw = goal_pose[2] if len(goal_pose) > 2 else None

        goto_id = (goal_idx + 1) * 1000
        self.client.send_goto(goto_id, target_x, target_y, yaw=target_yaw)

        acknowledged = False
        start_wait = time.monotonic()
        while not acknowledged:
            robot_state = self.client.latest_state()
            goto_info = robot_state["goto"]
            if goto_info["id"] == goto_id:
                acknowledged = True
            elif time.monotonic() - start_wait > 2.0:
                self.client.send_goto(goto_id, target_x, target_y, yaw=target_yaw)
                start_wait = time.monotonic()
            time.sleep(0.01)

        direct_success = False
        while True:
            robot_state = self.client.latest_state()
            goto_info = robot_state["goto"]
            status = goto_info["status"]
            pose = robot_state["pose"]

            print(f"🔄 Direct path | Pose: x={pose['x']:.2f}, y={pose['y']:.2f} | Status: {status}", end="\r")

            if goto_info.get("id") == goto_id:
                if status == "reached":
                    print(f"\nDirect path succeeded for '{goal_name}'!")
                    direct_success = True
                    break
                elif status == "bumped":
                    print(f"\nDirect path bumped an obstacle! Executing recovery rotation...")
                    self.execute_recovery_rotation()
                    print("Recovery rotation complete. Switching to A* planner...")
                    break
            time.sleep(0.02)


        if direct_success:
            self.mark_goal(goal_name)
        
        return direct_success

    def navigate_A_star_path(self, planner, goal_idx, goal_name, goal_pose):
        robot_state = self.client.latest_state()
        current_pose = robot_state["pose"]
        path = planner.plan(current_pose, goal_pose)

        if not path:
            print(f"❌ A* fallback failed to find a path to '{goal_name}'. Skipping...")
            return False

        print(f"A* fallback path found with {len(path)} waypoints. Navigating...")
        waypoints = path[::5]
        if path[-1] not in waypoints:
            waypoints.append(path[-1])

        navigation_success = self.navigate_waypoints(goal_idx, goal_name, waypoints, goal_pose)
        
        if navigation_success:
            self.mark_goal(goal_name)

        return navigation_success

    """Navigation and monitoring methods for waypoints and recovery rotation are defined below."""
    def navigate_waypoints(self, goal_idx, goal_name, waypoints, goal_pose):
        for wp_idx, (wp_x, wp_y) in enumerate(waypoints):
            goto_id = (goal_idx + 1) * 100 + wp_idx + 1  # Unique ID for each waypoint
            is_last_waypoint = (wp_idx == len(waypoints) - 1)

            # Append final target yaw only on the final waypoint if provided
            wyaw = goal_pose[2] if (is_last_waypoint and len(goal_pose) > 2) else None

            self.client.send_goto(goto_id, wp_x, wp_y, yaw=wyaw)

            # Wait for the simulator to acknowledge the new goto_id
            acknowledged = False
            start_wait = time.monotonic()
            while not acknowledged:
                robot_state = self.client.latest_state()
                goto_info = robot_state["goto"]
                if goto_info["id"] == goto_id:
                    acknowledged = True
                elif time.monotonic() - start_wait > 2.0:
                    self.client.send_goto(goto_id, wp_x, wp_y, yaw=wyaw)
                    start_wait = time.monotonic()
                time.sleep(0.01)

            # Monitor the waypoint until reached or bumped
            if not self.monitor_waypoint(goto_id, goal_name, wp_idx, len(waypoints)):
                print(f"❌ Navigation to waypoint {wp_idx + 1}/{len(waypoints)} failed due to bump. Aborting path navigation.")
                return False  # Indicate that the navigation failed due to a bump
            
        return True

    def monitor_waypoint(self, goto_id, goal_name, wp_idx, num_waypoints):
        """Monitor state stream until the waypoint is reached or bumped"""
        has_reached_wp = False
        while not has_reached_wp:
            robot_state = self.client.latest_state()
            goto_info = robot_state["goto"]
            status = goto_info["status"]
            pose = robot_state["pose"]

            print(f"🔄 Pose: x={pose['x']:.2f}, y={pose['y']:.2f} | Goto status: {status}", end="\r")

            if goto_info.get("id") != goto_id:
                time.sleep(0.02)
                continue  # Ignore other goto commands

            if status == "reached":
                print(f"\nReached waypoint {wp_idx + 1}/{num_waypoints} for '{goal_name}'.")
                has_reached_wp = True
            elif status == "bumped":
                print(f"\nRobot bumped an obstacle while heading to waypoint {wp_idx + 1}/{num_waypoints}! Executing recovery rotation...")
                
                self.execute_recovery_rotation()
                return False  # Indicate that the navigation failed due to a bump

            time.sleep(0.02)  # Poll matching the state stream rate (~50 Hz)

        return True  # Indicate that the waypoint was successfully reached

    def execute_recovery_rotation(self):
        """Rotate in place to clear the bumper sensor[cite: 1]."""
        recovery_end = time.monotonic() + 1.5
        while time.monotonic() < recovery_end:
            self.client.send_wheel(-self.turn_wheel_speed, self.turn_wheel_speed)
            time.sleep(0.05)
        self.client.send_wheel(0.0, 0.0)

    def mark_goal(self, goal_name):
        """Mark goal as reached via RPC[cite: 1]."""
        print(f"🎉 Successfully arrived at '{goal_name}'! Calling mark_goal...")
        try:
            mark_res, _ = self.client.call("mark_goal", name=goal_name)
            print(f"Mark goal response: {mark_res}\n")
            return True
        except RuntimeError as e:
            print(f"Mark goal note: {e}\n")
            return False