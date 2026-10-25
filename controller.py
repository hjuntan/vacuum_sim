import time
import json
import math
import numpy as np

class Controller:
    def __init__(self, client, planner, robot_info, camera_info=None):
        self.client = client
        self.camera_info = camera_info
        self.planner = planner
        self.robot_info = robot_info
        turn_rate = 0.8
        self.turn_wheel_speed = turn_rate * robot_info["wheel_base"] / 2 / robot_info["wheel_radius"]

    def execute_goal_mission(self, goal_idx, goal_name, goal_pose, current_pose=None):
        target_x, target_y = goal_pose[0], goal_pose[1]

        print(f"Planning path to goal '{goal_name}' at x={target_x}, y={target_y}")

        # Orient the robot to face the goal before performing the camera check
        target_yaw = math.atan2(target_y - current_pose["y"], target_x - current_pose["x"])
        print(f"Orienting robot to face goal at yaw {target_yaw:.2f}...")
        self.rotate_to(target_yaw)
        time.sleep(0.2)  # Brief pause to let camera stream settle

        dist_to_goal = math.hypot(target_x - current_pose["x"], target_y - current_pose["y"]) if current_pose else None

        # Check if the direct path is clear using current map data
        map_clear = self.planner.is_line_safe(current_pose, goal_pose) if current_pose else False

        direct_success = False
        if map_clear:
            direct_success = self.navigate_direct_path(goal_idx, goal_name, goal_pose)

        if direct_success:
            return True  # Indicate that the goal was successfully processed via direct navigation
        
        # A* fallback if direct navigation fails or is not clear
        print(f"Using A* waypoints...")


        return self.navigate_A_star_path(goal_idx, goal_name, goal_pose)


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

            self.planner.visualize(robot_pose=pose)

            print(f"🔄 Direct path | Pose: x={pose['x']:.2f}, y={pose['y']:.2f}, yaw={pose['yaw']:.2f} | Status: {status}", end="\r")

            if goto_info.get("id") == goto_id:
                if status == "reached":
                    print(f"\nDirect path succeeded for '{goal_name}'!")
                    direct_success = True
                    break
                elif status == "bumped":
                    print(f"\n💥 Robot bumped an obstacle! Marking and recovering...")
                    # Precisely mark the obstacle using the robot pose at the moment of impact
                    self.planner.mark_obstacle_from_bumper(pose)
                    self.execute_recovery()
                    print("Recovery rotation complete. Switching to A* planner...")
                    break
            time.sleep(0.02)


        if direct_success:
            self.mark_goal(goal_name)
        
        return direct_success

    def navigate_A_star_path(self, goal_idx, goal_name, goal_pose):
        max_attempts = 5
        for attempt in range(max_attempts):
            # self.update_map_from_camera()

            robot_state = self.client.latest_state()
            current_pose = robot_state["pose"]
            path = self.planner.plan(current_pose, goal_pose)

            if not path:
                print(f"❌ A* failed to find a path to '{goal_name}'. Waiting and retrying...")
                time.sleep(0.5)
                continue

            print(f"A* path found with {len(path)} waypoints. Navigating...")
            waypoints = self.planner.optimize_path(current_pose, path)
            if path[-1] not in waypoints:
                waypoints.append(path[-1])

            navigation_success = self.navigate_waypoints(goal_idx, goal_name, waypoints, goal_pose)
            
            if navigation_success:
                self.mark_goal(goal_name)
                return True
            else:
                print("🔄 Bump detected during navigation. Re-running A* search with updated map...")
                time.sleep(0.2)

        print(f"❌ Failed to reach goal '{goal_name}' after {max_attempts} attempts.")
        return False

    """Navigation and monitoring methods for waypoints and recovery rotation are defined below."""
    def navigate_waypoints(self, goal_idx, goal_name, waypoints, goal_pose):
        track_width = self.robot_info["radius"] * 2.0        # Robot width matching simulator specs
        wheel_radius = self.robot_info["wheel_radius"]      # Standard simulator wheel radius
        max_linear_speed = self.robot_info["max_linear_speed"]   # Max forward speed (m/s)
        max_angular_speed = self.robot_info["max_angular_speed"]   # Max turning speed (rad/s)

        for idx, (wx, wy) in enumerate(waypoints):
            print(f"Waypoint {idx + 1}/{len(waypoints)}: heading to ({wx:.2f}, {wy:.2f})")
            
            while True:
                robot_state = self.client.latest_state()
                pose = robot_state["pose"]
                x, y, yaw = pose["x"], pose["y"], pose["yaw"]

                # Check bumper collision during movement
                if robot_state.get("bumped", False):
                    print(f"\n💥 Bumper collision detected during smooth wheel navigation!")
                    self.planner.mark_obstacle_from_bumper(pose)
                    self.execute_recovery()
                    return False

                # Dynamically update map from camera observations on the fly
                # self.update_map_from_camera()

                # Calculate distance and angle error to current waypoint
                dx = wx - x
                dy = wy - y
                dist = math.hypot(dx, dy)

                # Waypoint reached tolerance (15cm threshold)
                if dist < 0.15:
                    break

                target_angle = math.atan2(dy, dx)
                angle_error = target_angle - yaw
                # Normalize angle error to [-pi, pi]
                angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

                # Smooth velocity profiling: slow down linear speed during sharp turns
                linear_speed = max_linear_speed * max(0.15, math.cos(angle_error))
                if abs(angle_error) > 0.8:  # Sharp turn threshold
                    linear_speed = 0.05

                # Proportional steering controller for angular velocity
                angular_speed = 2.5 * angle_error
                angular_speed = max(-max_angular_speed, min(max_angular_speed, angular_speed))

                # Convert linear & angular velocities to left/right wheel speeds (rad/s)
                v_center = linear_speed / wheel_radius
                omega = angular_speed

                v_l = v_center - (omega * track_width / (2.0 * wheel_radius))
                v_r = v_center + (omega * track_width / (2.0 * wheel_radius))

                # Stream continuous smooth wheel commands
                self.client.send_wheels(v_l, v_r)
                
                # Live visualization
                self.planner.visualize(robot_pose=pose, path=waypoints)
                time.sleep(0.05)

        # Stop robot after completing final waypoint
        self.client.send_wheels(0.0, 0.0)
        print(f"✅ Successfully navigated all waypoints for '{goal_name}' with smooth motion!")
        return True

    def monitor_waypoint(self, goto_id, goal_name, wp_idx, num_waypoints):
        """Monitor state stream until the waypoint is reached or bumped"""
        has_reached_wp = False
        while not has_reached_wp:
            robot_state = self.client.latest_state()
            goto_info = robot_state["goto"]
            status = goto_info["status"]
            pose = robot_state["pose"]

            self.planner.visualize(robot_pose=pose)

            print(f"🔄 Pose: x={pose['x']:.2f}, y={pose['y']:.2f}, yaw={pose['yaw']:.2f} | Goto status: {status}", end="\r")

            if goto_info.get("id") != goto_id:
                time.sleep(0.02)
                continue  # Ignore other goto commands

            if status == "reached":
                print(f"\nReached waypoint {wp_idx + 1}/{num_waypoints} for '{goal_name}'.")
                has_reached_wp = True
            elif status == "bumped":
                print(f"\n💥 Robot bumped an obstacle! Marking and recovering...")
                # Precisely mark the obstacle using the robot pose at the moment of impact
                self.planner.mark_obstacle_from_bumper(pose)
                self.execute_recovery()
                return False  # Indicate that the navigation failed due to a bump

            time.sleep(0.02)  # Poll matching the state stream rate (~50 Hz)

        return True  # Indicate that the waypoint was successfully reached

    def update_map_from_camera(self):
        if not self.camera_info:
            return
        try:
            header, _, raw = self.client.latest_camera()
            header = json.loads(header)
            depth = np.frombuffer(raw, "<u2").reshape(header["height"], header["width"]) * self.camera_info["depth_scale"]
            
            robot_state = self.client.latest_state()
            self.planner.mark_obstacle_from_camera(depth, self.camera_info, robot_state["pose"])
        except Exception as e:
            print(f"⚠️ Failed to update map from camera: {e}")

    def execute_recovery(self):
        """Rotate in place to clear the bumper sensor."""
        wheel_radius = self.robot_info["radius"]
        wheel_base = self.robot_info["wheel_base"]
        
        target_reverse_distance = 0.17 + 0.10
        linear_speed = 0.25 # m/s 
        rev_wheel_speed = linear_speed / wheel_radius

        reverse_time = target_reverse_distance / linear_speed
        reverse_end = time.monotonic() + reverse_time
        while time.monotonic() < reverse_end:
            self.client.send_wheels(-rev_wheel_speed, -rev_wheel_speed)
            time.sleep(0.05)
        
        # Stop briefly after backing up
        self.client.send_wheels(0.0, 0.0)
        time.sleep(0.1)

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

    def rotate_to(self, target_yaw, tolerance=0.01, timeout=5.0):
        """Rotate the robot to a specific yaw angle (in radians) with a given tolerance."""
        start_time = time.monotonic()
        while time.monotonic() - start_time < timeout:
            robot_state = self.client.latest_state()
            current_yaw = robot_state["pose"]["yaw"]
            angle_diff = self.normalize_angle(target_yaw - current_yaw)

            if abs(angle_diff) <= tolerance:
                self.client.send_wheels(0.0, 0.0)
                return True  # Successfully rotated to target yaw

            turn_direction = -1.0 if angle_diff > 0 else 1.0
            self.client.send_wheels(turn_direction * self.turn_wheel_speed, -turn_direction * self.turn_wheel_speed)
            time.sleep(0.05)

        self.client.send_wheels(0.0, 0.0)
        return False  # Failed to rotate within the timeout

    def mark_goal(self, goal_name):
        """Mark goal as reached via RPC."""
        print(f"🎉 Successfully arrived at '{goal_name}'! Calling mark_goal...")
        try:
            mark_res, _ = self.client.call("mark_goal", name=goal_name)
            print(f"Mark goal response: {mark_res}\n")
            return True
        except RuntimeError as e:
            print(f"Mark goal note: {e}\n")
            return False