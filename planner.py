"""Occupancy grid planner for the vacuum robot."""

import heapq
import math
from io import BytesIO
from PIL import Image
import numpy as np

class Planner:
    def __init__(self, map_info, png):
        self.resolution = map_info["resolution"]

        # Row 0 is the top of the image (largest y); rows grow towards −y, columns towards +x.
        # 0 = free, 255 = occupied
        self.origin = map_info["origin"]
        
        self.grid = np.array(Image.open(BytesIO(png)))
        self.width, self.height = map_info["width"], map_info["height"]

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
        if self.grid[start_row, start_col] > 127 or self.grid[goal_row, goal_col] > 127:
            return []

        def heuristic(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        open_set = []
        heapq.heappush(open_set, (0, start_row, start_col))
        came_from = {}
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
                if 0 <= neighbor_row < self.height and 0 <= neighbor_col < self.width:
                    if self.grid[neighbor_row, neighbor_col] > 127:
                        continue  # Skip obstacles

                    move_cost = math.hypot(dr, dc)
                    tentative_g = g_score[(current_row, current_col)] + move_cost

                    if tentative_g < g_score.get((neighbor_row, neighbor_col), float("inf")):
                        came_from[(neighbor_row, neighbor_col)] = (current_row, current_col)
                        g_score[(neighbor_row, neighbor_col)] = tentative_g
                        f_score = tentative_g + heuristic((neighbor_row, neighbor_col), (goal_row, goal_col))
                        heapq.heappush(open_set, (f_score, neighbor_row, neighbor_col))

        return []  # No path found