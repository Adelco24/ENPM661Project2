# Project 2 - Adam Del Colliano

This project uses a Breadth First Search and Dijkstra to solve a maze with obstacles. The algorithm finds all the possible paths of the maze starting from the given initial nodes until the goal node is reached.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install numpy.
OpenCV is also used for maze generation.

```bash
pip install numpy
pip install opencv-python
```

This project also uses deque (short for double-ended queue, from the collections package), heapq, and time, but these are built-in modules and come with the base installation of python. They do not, therefore, require any install.

## Usage

The files BFS_adam_delcolliano.py and dijkstra_adam_delcolliano.py are ran by opening the terminal, navigating to the directory the file is saved in, and then running the following commands:
```bash
python3 BFS_adam_delcolliano.py
python3 dijkstra_adam_delcolliano.py
```
They each will ask for a x,y location for the start as user input, and will not allow for erroreous inputs. The user also has an option to simply hit Enter to skip past this and the code will run the hardcoded value in the proj2_adam_delcolliano.py file. The same is true for the next step, which asks for an x,y location for the goal. If either point is in the obstacles, the user will be made aware.

Then the algorithm will find its path (usually within 10 seconds), and the maze will generate (takes about 60 seconds). By clicking on the maze and hitting 'q', it closes the window.

If the maze isn't solvable (the user puts a start or end point inside the hole of the '6' obstacle for example) the terminal will print that it can't be solved.
