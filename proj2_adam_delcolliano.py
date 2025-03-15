#Adam Del Colliano
#U ID: 115846982

import numpy as np
import cv2
import heapq
from collections import deque
import time

# Define map size
scale_factor = 5  # Scaling factor
width_mm, height_mm = 180, 50  # Map dimensions
width, height = width_mm * scale_factor, height_mm * scale_factor  # Scaled map size
clearance = 2 * scale_factor  # 2mm clearance scaled

# Define possible actions and their costs
actions = [(1 * scale_factor, 0), (-1 * scale_factor, 0), (0, 1 * scale_factor), (0, -1 * scale_factor),
           (1 * scale_factor, 1 * scale_factor), (-1 * scale_factor, 1 * scale_factor),
           (1 * scale_factor, -1 * scale_factor), (-1 * scale_factor, -1 * scale_factor)]
bfs_costs = [1] * 8  # BFS treats all moves equally
dijkstra_costs = [1, 1, 1, 1, 1.4, 1.4, 1.4, 1.4]  # Diagonal moves cost more

# Colors
dark_blue = (139, 0, 0)  # Border
medium_blue = (255, 140, 0)  # Inside obstacles
light_blue = (255, 255, 224)  # Free space
yellow = (0, 255, 255)  # Buffer zone
red = (0, 0, 255)
green = (0, 255, 0)

def is_inside_half_plane(p, a, b, c):
    """Check if a point (x, y) satisfies the half-plane inequality ax + by + c <= 0"""
    x, y = p
    return a * x + b * y + c <= 0

def is_inside_circle(p, center, radius):
    """Check if a point (x, y) is inside a circle."""
    x, y = p
    h, k = center
    return (x - h) ** 2 + (y - k) ** 2 <= radius ** 2

def is_inside_rectangle(p, x0, y0, dx, dy):
    if (is_inside_half_plane(p, -1, 0, x0) and
        is_inside_half_plane(p, 1, 0, -x0 - dx) and
        is_inside_half_plane(p, 0, -1, y0) and
        is_inside_half_plane(p, 0, 1, -y0 - dy)):
        return True
    return False

def is_edge(p, thickness):
    """Check if a point is on the edge of any letter/number."""
    x, y = p
    if is_obstacle((x, y)):
        return False
    if x < thickness or y < thickness or x >= width - thickness or y >= height - thickness:
        return True
    neighbors = []
    for i in range(x - thickness, x + thickness + 1, scale_factor):
        for j in range(y - thickness, y + thickness + 1, scale_factor):
            if np.sqrt(((x - i) ** 2) + ((y - j) ** 2)) <= thickness:
                neighbors.append((i, j))
    for point in neighbors:
        if is_obstacle(point):
            return True
    return False

def is_obstacle(p):
    """Check if a point is inside any defined obstacles using half-planes and circles."""
    x, y = p

    # Letter 'E'
    xE = 15 * scale_factor
    yE = 15 * scale_factor
    if (is_inside_rectangle(p, xE, yE, 5 * scale_factor, 25 * scale_factor) or
        is_inside_rectangle(p, xE + 5 * scale_factor, yE, 8 * scale_factor, 5 * scale_factor) or
        is_inside_rectangle(p, xE + 5 * scale_factor, yE + 10 * scale_factor, 8 * scale_factor, 5 * scale_factor) or
        is_inside_rectangle(p, xE + 5 * scale_factor, yE + 20 * scale_factor, 8 * scale_factor, 5 * scale_factor)):
        return True

    # Letter 'N'
    xN = xE + 13 * scale_factor + 6 * scale_factor
    if (is_inside_rectangle(p, xN, yE, 5 * scale_factor, 25 * scale_factor) or
        is_inside_rectangle(p, xN + 10 * scale_factor, yE, 5 * scale_factor, 25 * scale_factor) or
        (is_inside_half_plane(p, -1, 0, xN + 5 * scale_factor) and
         is_inside_half_plane(p, 1, 0, -xN - 10 * scale_factor) and
         is_inside_half_plane(p, 3, -1, yE - 3 * xN - 15 * scale_factor) and
         is_inside_half_plane(p, -3, 1, -yE + 3 * xN + 5 * scale_factor))):
        return True

    # Letter 'P'
    xP = xN + 15 * scale_factor + 6 * scale_factor
    if (is_inside_rectangle(p, xP, yE, 5 * scale_factor, 25 * scale_factor) or is_inside_circle(p, (xP + 6 * scale_factor, yE + 6 * scale_factor), 6 * scale_factor)):
        return True

    # Letter 'M'
    xM = xP + 11 * scale_factor + 6 * scale_factor
    if (is_inside_rectangle(p, xM, yE, 5 * scale_factor, 25 * scale_factor) or
        is_inside_rectangle(p, xM + 9 * scale_factor, yE + 20 * scale_factor, 7 * scale_factor, 5 * scale_factor) or
        is_inside_rectangle(p, xM + 20 * scale_factor, yE, 5 * scale_factor, 25 * scale_factor) or
        (is_inside_half_plane(p, -1, 0, xM + 5 * scale_factor) and
         is_inside_half_plane(p, 1, 0, -xM - 10 * scale_factor) and
         is_inside_half_plane(p, 4, -1, yE - 4 * xM - 20 * scale_factor) and
         is_inside_half_plane(p, -4, 1, -yE + 4 * xM + 11 * scale_factor) and
         is_inside_half_plane(p, 0, 1, -yE - 25 * scale_factor)) or
        (is_inside_half_plane(p, -1, 0, xM + 15 * scale_factor) and
         is_inside_half_plane(p, 1, 0, -xM - 20 * scale_factor) and
         is_inside_half_plane(p, -4, -1, yE + 80 * scale_factor + 4 * xM) and
         is_inside_half_plane(p, 4, 1, -yE - 89 * scale_factor - 4 * xM) and
         is_inside_half_plane(p, 0, 1, -yE - 25 * scale_factor))):
        return True

    # Number '6' (First)
    x61 = xM + 25 * scale_factor + 6 * scale_factor
    if ((is_inside_circle(p, (x61 + 9 * scale_factor, yE + 16 * scale_factor), 9 * scale_factor) and not is_inside_circle(p, (x61 + 9 * scale_factor, yE + 16 * scale_factor), 4 * scale_factor)) or
        is_inside_circle(p, (x61 + 13 * scale_factor, yE - 0.5 * scale_factor), 2.5 * scale_factor) or
        (is_inside_circle(p, (x61 + 21 * scale_factor, yE + 17 * scale_factor), 21.5 * scale_factor) and not is_inside_circle(p, (x61 + 21 * scale_factor, yE + 17 * scale_factor), 16.5 * scale_factor) and is_inside_rectangle(p, x61, yE - 3 * scale_factor, 13 * scale_factor, 19 * scale_factor))):
        return True

    # Number '6' (Second)
    x62 = x61 + 18 * scale_factor + 6 * scale_factor
    if ((is_inside_circle(p, (x62 + 9 * scale_factor, yE + 16 * scale_factor), 9 * scale_factor) and not is_inside_circle(p, (x62 + 9 * scale_factor, yE + 16 * scale_factor), 4 * scale_factor)) or
        is_inside_circle(p, (x62 + 13 * scale_factor, yE - 0.5 * scale_factor), 2.5 * scale_factor) or
        (is_inside_circle(p, (x62 + 21 * scale_factor, yE + 17 * scale_factor), 21.5 * scale_factor) and not is_inside_circle(p, (x62 + 21 * scale_factor, yE + 17 * scale_factor), 16.5 * scale_factor) and is_inside_rectangle(p, x62, yE - 3 * scale_factor, 13 * scale_factor, 19 * scale_factor))):
        return True

    # Number '1'
    x1 = x62 + 18 * scale_factor + 6 * scale_factor
    if (is_inside_rectangle(p, x1, yE - 3 * scale_factor, 5 * scale_factor, 28 * scale_factor)):
        return True

    return False

def is_valid(node):
    """Check if a node is valid (inside boundaries and not an obstacle)."""
    x, y = node
    return 0 <= x < width and 0 <= y < height and not is_obstacle((x, y)) and not is_edge((x, y), clearance)

def move(node, action):
    """Move to a new node based on an action."""
    return (node[0] + action[0], node[1] + action[1])

def reconstruct_path(parent_map, start, goal):
    """Backtrack from goal to start to retrieve the optimal path."""
    path = []
    current = goal
    while current != start:
        path.append(current)
        current = parent_map[current]
    path.append(start)
    path.reverse()
    return path

def bfs(start, goal):
    """Breadth-First Search algorithm for the maze, structured like project 1 BFS."""
    # If the start is already the goal, return immediately.
    if start == goal:
        return [start]

    queue = deque([start])
    visited = set([start])
    parents = {}
    explored_nodes = []  # Store nodes as they are explored

    while queue:
        current = queue.popleft()
        explored_nodes.append(current)

        if current == goal:
            return reconstruct_path(parents, start, goal), explored_nodes

        for action in actions:
            next_node = move(current, action)
            if is_valid(next_node) and next_node not in visited:
                visited.add(next_node)
                parents[next_node] = current
                queue.append(next_node)

    return None, explored_nodes  # No path found

def dijkstra(start, goal):
    """Dijkstra's algorithm for the maze, following a similar structure to the BFS above."""
    if start == goal:
        return [start]

    open_list = []
    visited = set()
    parents = {}
    cost_map = {}
    explored_nodes = []  # Store nodes as they are explored

    cost_map[start] = 0
    heapq.heappush(open_list, (0, start))

    while open_list:
        current_cost, current = heapq.heappop(open_list)
        if current in visited:
            continue
        visited.add(current)
        explored_nodes.append(current)

        if current == goal:
            return reconstruct_path(parents, start, goal), explored_nodes

        for i, action in enumerate(actions):
            next_node = move(current, action)
            new_cost = current_cost + dijkstra_costs[i]

            if is_valid(next_node) and (next_node not in cost_map or new_cost < cost_map[next_node]):
                cost_map[next_node] = new_cost
                parents[next_node] = current
                heapq.heappush(open_list, (new_cost, next_node))

    return None, explored_nodes  # No path found

def show_map(start, goal):
    """Display the generated map with obstacles, start point, and end point."""
    map_img = np.ones((height, width, 3), dtype=np.uint8) * np.array(light_blue, dtype=np.uint8)  # Light blue background

    # Draw obstacles and edges
    for y in range(height):
        for x in range(width):
            if is_obstacle((x, y)):
                map_img[y, x] = medium_blue  # Medium blue for obstacles
            elif is_edge((x, y), 2 * scale_factor):
                map_img[y, x] = yellow  # Yellow for buffer zone

    # Draw the border
    cv2.rectangle(map_img, (0, 0), (width - 1, height - 1), dark_blue, 1)

    cv2.circle(map_img, (start[0], start[1]), 1 * scale_factor, (0, 0, 0), -1) 
    cv2.circle(map_img, (goal[0], goal[1]), 1 * scale_factor, (0, 0, 0), -1)   

    cv2.imshow("Map", map_img)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()
    cv2.waitKey(1)  # Ensures window closes completely

def visualize(path, explored_nodes, algorithm_name,start,goal):
    """Visualize the search process and final path using OpenCV."""
    map_img = np.ones((height, width, 3), dtype=np.uint8) * np.array(light_blue, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            if is_obstacle((x, y)):
                map_img[y, x] = medium_blue
            elif is_edge((x, y), 2 * scale_factor):
                map_img[y, x] = yellow

    cv2.rectangle(map_img, (0, 0), (width - 1, height - 1), dark_blue, 1)
    cv2.circle(map_img, start, 1 * scale_factor, (0, 0, 0), -1)  
    cv2.circle(map_img, goal, 1 * scale_factor, (0, 0, 0), -1)    

    # Show search process in red
    for node in explored_nodes:
        cv2.circle(map_img, (node[0], node[1]), 1 * scale_factor, red, -1)
        cv2.circle(map_img, start, 1 * scale_factor, (0, 0, 0), -1)  #keep start on top
        cv2.circle(map_img, goal, 1 * scale_factor, (0, 0, 0), -1)  #keep goal on top
        cv2.imshow(f"{algorithm_name} - Search and Path", map_img)
        cv2.waitKey(4)  # Show search process

    # Overlay the final path in green on the same image
    if path:
        for node in path:
            cv2.circle(map_img, (node[0], node[1]), 1 * scale_factor, green, -1)
            cv2.circle(map_img, start, 1 * scale_factor, (0, 0, 0), -1)  #keep start on top
            cv2.circle(map_img, goal, 1 * scale_factor, (0, 0, 0), -1)  #keep goal on top
            cv2.imshow(f"{algorithm_name} - Search and Path", map_img)
            cv2.waitKey(20)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.waitKey(1)

def get_point(loc = "start"):
    while True:
        user_input = input(f"Enter x and y location for {loc} separated by comma, in format x,y (x from 1 to 180, y from 1 to 50): ").strip()
        if user_input == "" and loc == "start":
            return (2*scale_factor,2*scale_factor) #(3,48)
        elif user_input == "" and loc == "goal":
            return (147*scale_factor,47*scale_factor) #(148,3)
        parts = user_input.split(",")
        if len(parts) == 2:
            try:
                x = int(parts[0].strip())
                y = int(parts[1].strip())
                if 1<=x<=180 and 1<=y<=50:
                    x = (x-1) * scale_factor
                    y = (50-y) * scale_factor
                    if is_valid((x,y)):
                        return (x,y)
                    else:
                        print("Sorry this point is within the obstacle space. Try again.")
                else:
                    print("Invalid input. Please ensure both x and y are within the bounds of the space.")
            except ValueError:
                print("Invalid input. Please enter integers for both x and y.")
        else:
            print("Invalid input. Please enter exactly two integers separated by a comma.")



# Main execution
if __name__ == "__main__":

    start = get_point("start")
    goal = get_point("goal")

    #show_map(start, goal)

    print("Running BFS...")
    start_time_b = time.time()
    path_bfs, explored_bfs = bfs(start, goal)
    end_time_b = time.time()
    elapsed_time_b = end_time_b - start_time_b
    if path_bfs:
        print(f"BFS found a path in {elapsed_time_b:.4f} seconds!!")
        visualize(path_bfs, explored_bfs, "BFS",start,goal)
    else:
        print("BFS could not find a path.")

    print("Running Dijkstra...")
    start_time_d = time.time()
    path_dijkstra, explored_dijkstra = dijkstra(start, goal)
    end_time_d = time.time()
    elapsed_time_d = end_time_d - start_time_d
    if path_dijkstra:
        print(f"Dijkstra found a path in {elapsed_time_d:.4f} seconds!")
        visualize(path_dijkstra, explored_dijkstra, "Dijkstra",start,goal)
    else:
        print("Dijkstra could not find a path.")