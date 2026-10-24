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
        y, x = np.ogrid[-radius_px:radius_px+1, -radius_px:radius_px+1]
        structuring_element = x**2 + y**2 <= (radius_px)**2

        # PRECOMPUTE: Dilate obstacles once using C-optimized SciPy binary dilation.
        # This permanently expands walls outward by the robot's full safety footprint.
        obstacles = grid > 127
        inflated_obstacles = binary_dilation(obstacles, structure=structuring_element)

        self.grid = np.zeros_like(grid)
        self.grid[inflated_obstacles] = 255

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

    def map_to_world(self, row, col):
        """Convert map coordinates (pixels) to world coordinates (m)."""
        x = self.origin[0] + (col + 0.5) * self.resolution
        y = self.origin[1] + (self.height - 1 - row + 0.5) * self.resolution
        return x, y

    def world_to_map(self, x, y):
        """Convert world coordinates (m) to map coordinates (pixels)."""
        col = int((x - self.origin[0]) / self.resolution - 0.5)
        row = int(self.height - 1 - ((y - self.origin[1]) / self.resolution - 0.5))
        return row, col

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