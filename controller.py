import time

class Controller:
    def __init__(self, client, robot_info):
        self.client = client
        turn_rate = 1.5
        self.turn_wheel_speed = turn_rate * robot_info["wheel_base"] / 2 / robot_info["wheel_radius"]

    def navigate_path(self, goal_idx, goal_name, waypoints, goal_pose):
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
                
                # Rotate in place for 1.5 seconds to back away / clear the bumper sensor
                recovery_end = time.monotonic() + 1.5
                while time.monotonic() < recovery_end:
                    # Send counter-rotating wheel commands at ~20 Hz to respect watchdog
                    self.client.send_wheel(-self.turn_wheel_speed, self.turn_wheel_speed)
                    time.sleep(0.05)
                
                # Stop wheels explicitly
                self.client.send_wheel(0.0, 0.0)
                print("Recovery rotation complete. Moving to next waypoint...")
                return False  # Indicate that the navigation failed due to a bump

            time.sleep(0.02)  # Poll matching the state stream rate (~50 Hz)

        return True  # Indicate that the waypoint was successfully reached