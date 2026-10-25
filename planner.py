"""Occupancy grid planner for the vacuum robot."""

import heapq
import math
import numpy as np

from io import BytesIO
from PIL import Image
from scipy.ndimage import binary_dilation

class Planner:
    def __init__(self, map_info, png, robot_diameter=0.34, leeway=0.075):
        self.resolution = map_info["resolution"]

        # Row 0 is the top of the image (largest y); rows grow towards −y, columns towards +x.
        # 0 = free, 255 = occupied
        self.origin = map_info["origin"]
        
        grid = np.array(Image.open(BytesIO(png)))
        self.width, self.height = map_info["width"], map_info["height"]

        robot_radius = robot_diameter / 2.0
        self.radius = robot_radius + leeway

        # Create a circular structuring element (footprint mask) for C-Space dilation
        radius_px = math.ceil(self.radius / self.resolution)
        self.radius_px = radius_px
        y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
        structuring_element = x**2 + y**2 <= (radius_px)**2

        # PRECOMPUTE: Dilate obstacles once using C-optimized SciPy binary dilation.
        # This permanently expands walls outward by the robot's full safety footprint.
        obstacles = grid > 127
        inflated_obstacles = binary_dilation(obstacles, structure=structuring_element)

        self.base_grid = np.zeros_like(grid)
        self.base_grid[inflated_obstacles] = 255
        self.grid = self.base_grid.copy()

    def plan(self, start, goal):
        """Plan a path from start to goal using A*."""
        start_row, start_col = self.world_to_map(start["x"], start["y"])
        goal_row, goal_col = self.world_to_map(goal[0], goal[1])

        # If start or goal is inside an obstacle, return empty path
        if not self.is_safe_to_occupy(start_row, start_col) or not self.is_safe_to_occupy(goal_row, goal_col):
            return []

        # h(n) = Euclidean distance heuristic
        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start_row, start_col))
        came_from = {} # For backtracking

        # g(n) = cost from start to current node
        g_score = {(start_row, start_col): 0}

        # 8-connected grid neighbors (orthogonal and diagonal)
        neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

        while open_set:
            _, current_row, current_col = heapq.heappop(open_set)

            if (current_row, current_col) == (goal_row, goal_col):
                # Reconstruct path
                path = []
                while (current_row, current_col) in came_from:
                    path.append(self.map_to_world(current_row, current_col))
                    current_row, current_col = came_from[(current_row, current_col)]
                path.reverse()
                return path

            for dr, dc in neighbors:
                neighbor_row, neighbor_col = current_row + dr, current_col + dc

                if neighbor_row < 0 or neighbor_row >= self.height or neighbor_col < 0 or neighbor_col >= self.width:
                    continue  # Skip out-of-bounds neighbors

                if not self.is_safe_to_occupy(neighbor_row, neighbor_col):
                    continue  # Skip occupied neighbors

                move_cost = math.hypot(dr, dc)
                tentative_g = g_score[(current_row, current_col)] + move_cost # g(n)

                if tentative_g < g_score.get((neighbor_row, neighbor_col), float("inf")):
                    came_from[(neighbor_row, neighbor_col)] = (current_row, current_col)
                    g_score[(neighbor_row, neighbor_col)] = tentative_g

                    # f(n) = g(n) + h(n)
                    f_score = tentative_g + heuristic((neighbor_row, neighbor_col), (goal_row, goal_col))
                    heapq.heappush(open_set, (f_score, neighbor_row, neighbor_col))

        return []  # No path found

    def is_safe_to_occupy(self, row, col):
        """Check if the robot can safely occupy the cell at (row, col) considering its footprint."""
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        return self.grid[row, col] <= 127

    def is_line_safe(self, start_pose, goal_pose, step_size=0.05):
        """Check if the straight line from start_pose to goal_pose is free of obstacles."""
        start_row, start_col = self.world_to_map(start_pose["x"], start_pose["y"])
        goal_row, goal_col = self.world_to_map(goal_pose[0], goal_pose[1])

        distance = math.hypot(goal_pose[0] - start_pose["x"], goal_pose[1] - start_pose["y"])
        num_steps = max(1, int(distance / step_size))

        for i in range(num_steps + 1):
            t = i / num_steps
            r = int(start_row + t * (goal_row - start_row))
            c = int(start_col + t * (goal_col - start_col))
            if not self.is_safe_to_occupy(r, c):
                return False
        return True

    def mark_obstacle_from_bumper(self, robot_pose):
        """Precisely mark an obstacle at the robot's perimeter when a bumper collision occurs."""
        x, y, yaw = robot_pose["x"], robot_pose["y"], robot_pose["yaw"]
        
        # Robot radius is 0.17m (0.34m diameter / 2)
        robot_radius = 0.17
        
        # The contact point is exactly at the robot's outer edge along its current yaw heading
        obs_x = x + robot_radius * math.cos(yaw)
        obs_y = y + robot_radius * math.sin(yaw)
        
        r, c = self.world_to_map(obs_x, obs_y)
        if 0 <= r < self.height and 0 <= c < self.width: # note: self.width
            # Inflate with robot radius footprint so future paths steer clear of it
            for dr in range(-self.radius_px, self.radius_px + 1):
                for dc in range(-self.radius_px, self.radius_px + 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        if math.hypot(dr * self.resolution, dc * self.resolution) <= self.radius:
                            self.grid[nr, nc] = 255
                            
        print(f"🛑 Bumper hit! Precisely marked obstacle at world coords ({obs_x:.2f}, {obs_y:.2f})")
    
    def map_to_world(self, row, col):
        """Convert map coordinates (pixels) to world coordinates (m)."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (self.height - 1 - row + 0.5) * self.resolution
        return x, y

    def world_to_map(self, x, y):
        """Convert world coordinates (m) to map coordinates (pixels)."""
        col = round((x - self.origin[0]) / self.resolution)
        row = round(self.height - 1 - ((y - self.origin[1]) / self.resolution))
        return row, col

    def optimize_path(self, start_pose, path, spacing=0.5):
        """Prune redundant waypoints using line-of-sight for sharp turns, 
        then resample straight segments so waypoints are spaced by `spacing` (e.g., 50cm)."""
        if not path:
            return []

        # Step 1: Greedy line-of-sight pruning to keep sharp inflection/turning points
        full_path = [(start_pose["x"], start_pose["y"])] + path
        optimized = [full_path[0]]

        i = 0
        while i < len(full_path) - 1:
            furthest_safe = i + 1
            for j in range(len(full_path) - 1, i + 1, -1):
                if self.is_line_safe_points(full_path[i], full_path[j]):
                    furthest_safe = j
                    break
            optimized.append(full_path[furthest_safe])
            i = furthest_safe

        # Step 2: Resample the corner-to-corner path at fixed 50cm intervals
        resampled = []
        if len(optimized) > 0:
            resampled.append(optimized[0])

        for k in range(len(optimized) - 1):
            p1 = optimized[k]
            p2 = optimized[k + 1]

            dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
            if dist == 0:
                continue

            # Calculate how many 50cm steps fit into this segment
            num_steps = max(1, math.ceil(dist / spacing))

            for step in range(1, num_steps + 1):
                t = step / num_steps
                x = p1[0] + t * (p2[0] - p1[0])
                y = p1[1] + t * (p2[1] - p1[1])
                
                # Avoid duplicate points near the end of segments
                if not resampled or math.hypot(x - resampled[-1][0], y - resampled[-1][1]) > 0.05:
                    resampled.append((x, y))

        # Return resampled path excluding the starting position
        return resampled[1:] if resampled else []

    def is_line_safe_points(self, p1, p2, step_size=0.05):
        """Helper to check if a straight line between two world coordinates is obstacle-free."""
        r1, c1 = self.world_to_map(p1[0], p1[1])
        r2, c2 = self.world_to_map(p2[0], p2[1])

        distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        num_steps = max(1, int(distance / step_size))

        for i in range(num_steps + 1):
            t = i / num_steps
            r = int(r1 + t * (r2 - r1))
            c = int(c1 + t * (c2 - c1))
            
            if not (0 <= r < self.height and 0 <= c < self.width):
                return False
            if self.grid[r, c] > 127:  # Obstacle detected
                return False
        return True

    def visualize(self, robot_pose=None, path=None):
        """Render the live occupancy grid map and robot position using OpenCV."""
        import cv2
        # Copy grid and convert to BGR color for overlays
        img = self.grid.copy().astype(np.uint8)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # Draw A* path if available (Blue line)
        if path and len(path) > 1:
            for i in range(len(path) - 1):
                r1, c1 = self.world_to_map(path[i][0], path[i][1])
                r2, c2 = self.world_to_map(path[i+1][0], path[i+1][1])
                cv2.line(img_bgr, (c1, r1), (c2, r2), (255, 0, 0), 2)

        # Draw robot position if available (Red filled circle)
        if robot_pose:
            rr, cc = self.world_to_map(robot_pose["x"], robot_pose["y"])
            radius_px = max(int(self.radius / self.resolution), 3)
            cv2.circle(img_bgr, (cc, rr), radius_px, (0, 0, 255), -1)

        # Display window
        cv2.imshow("Live Occupancy Grid Map", img_bgr)
        cv2.waitKey(1)