# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""An environment which is built by a learning adversary.
Has additional functions, step_adversary, and reset_agent. How to use:
1. Call reset() to reset to an empty environment
2. Call step_adversary() to place the goal, agent, and obstacles. Repeat until
   a done is received.
3. Normal RL loop. Use learning agent to generate actions and use them to call
   step() until a done is received.
4. If required, call reset_agent() to reset the environment the way the
   adversary designed it. A new agent can now play it using the step() function.
"""
import logging
import random

import gym
import networkx as nx
from networkx import grid_graph
import numpy as np

from social_rl.gym_multigrid import multigrid
from social_rl.gym_multigrid import register
from social_rl.gym_multigrid.gym_minigrid import minigrid
from social_rl.gym_multigrid.gym_minigrid.minigrid import COLORS, Grid

# import ALL the constants
from social_rl.gym_multigrid.gym_minigrid.minigrid import (
    COLORS,
    COLOR_NAMES,
    COLOR_TO_IDX,
    IDX_TO_COLOR,
    OBJECT_TO_IDX,
    IDX_TO_OBJECT,
)

WALL = "wall"
GOAL = "goal"
FLOOR = "floor"


ADV_STEPS = {
    0: WALL,
    1: GOAL,
    2: FLOOR,
}


class AdversarialEnv(multigrid.MultiGridEnv):
    """Grid world where an adversary build the environment the agent plays.
    The adversary places the goal, agent, and up to n_clutter blocks in sequence.
    The action dimension is the number of squares in the grid, and each action
    chooses where the next item should be placed.
    """

    def __init__(
        self,
        n_clutter=50,
        size=15,
        agent_view_size=5,
        max_steps=250,
        goal_noise=0.0,
        random_z_dim=50,
        choose_goal_last=False,
        domain_shifts: bool = False,
        random_shifts: bool = False,
    ):
        """Initializes environment in which adversary places goal, agent, obstacles.
        Args:
          n_clutter: The maximum number of obstacles the adversary can place.
          size: The number of tiles across one side of the grid; i.e. make a
            size x size grid.
          agent_view_size: The number of tiles in one side of the agent's partially
            observed view of the grid.
          max_steps: The maximum number of steps that can be taken before the
            episode terminates.
          goal_noise: The probability with which the goal will move to a different
            location than the one chosen by the adversary.
          random_z_dim: The environment generates a random vector z to condition the
            adversary. This gives the dimension of that vector.
          choose_goal_last: If True, will place the goal and agent as the last
            actions, rather than the first actions.
          domain_shifts: Set to True if training with domain shifts. This will make
            the first 3 actions of step_adversary() select the colors of:
                1. Wall
                2. Goal
                3. Floor
            
        """
        self.agent_start_pos = None
        self.goal_pos = None
        self.n_clutter = n_clutter
        self.goal_noise = goal_noise
        self.random_z_dim = random_z_dim
        self.choose_goal_last = choose_goal_last

        # Add two actions for placing the agent and goal.
        self.adversary_max_steps = self.n_clutter + 2

        # WARNING: 'self.doing_shifts' is now used in 
        #   - 'social_rl/adversarial_env/adversarial_env.py'
        #   - 'social_rl/adversarial_env/adversarial_parallel_env.py'
        # for logging purposes

        if domain_shifts and random_shifts:
            raise TypeError("Cannot specify both random domain shifts and adversarial environments")
        
        self.random_shifts = random_shifts
        self.doing_shifts = domain_shifts
        
        shifts = "None"
        if domain_shifts:
            shifts = "Adversarial env"
        elif random_shifts:
            shifts = "Random color shifts"

        logging.info(f"Initializing env with domain shifts: {shifts}")

        super().__init__(
            n_agents=1,
            minigrid_mode=True,
            grid_size=size,
            max_steps=max_steps,
            agent_view_size=agent_view_size,
            see_through_walls=True,  # Set this to True for maximum speed
            competitive=True,
        )

        # Metrics
        self.reset_metrics()

        # Create spaces for adversary agent's specs.
        self.adversary_action_dim = (size - 2) ** 2
        self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)
        self.adversary_ts_obs_space = gym.spaces.Box(low=0, high=self.adversary_max_steps, shape=(1,), dtype="uint8")
        self.adversary_randomz_obs_space = gym.spaces.Box(low=0, high=1.0, shape=(random_z_dim,), dtype=np.float32)
        self.adversary_image_obs_space = gym.spaces.Box(
            low=0, high=255, shape=(self.width, self.height, 3), dtype="uint8"
        )

        # Adversary observations are dictionaries containing an encoding of the
        # grid, the current time step, and a randomly generated vector used to
        # condition generation (as in a GAN).
        self.adversary_observation_space = gym.spaces.Dict(
            {
                "image": self.adversary_image_obs_space,
                "time_step": self.adversary_ts_obs_space,
                "random_z": self.adversary_randomz_obs_space,
            }
        )

        # NetworkX graph used for computing shortest path
        self.graph = grid_graph(dim=[size - 2, size - 2])
        self.wall_locs = []
    
    # TODO: @busycalibrating - hacky workaround to access this value from the underlying gym object
    def domain_shifts(self):
        return self.doing_shifts

    def _gen_grid(self, width, height):
        """Grid is initially empty, because adversary will create it."""
        # Create an empty grid
        self.grid = multigrid.Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

    def get_goal_x(self):
        if self.goal_pos is None:
            return -1
        return self.goal_pos[0]

    def get_goal_y(self):
        if self.goal_pos is None:
            return -1
        return self.goal_pos[1]

    def reset_metrics(self):
        self.distance_to_goal = -1
        self.n_clutter_placed = 0
        self.deliberate_agent_placement = -1
        self.passable = -1
        self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

    def reset(self):
        """Fully resets the environment to an empty grid with no agent or goal."""
        self.used_colors = set()

        # moved to init method
        # self.doing_shifts = True

        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])
        self.wall_locs = []

        self.step_count = 0
        self.adversary_step_count = 0

        self.agent_start_dir = self._rand_int(0, 4)

        # Current position and direction of the agent
        self.reset_agent_status()

        self.agent_start_pos = None
        self.goal_pos = None

        # Extra metrics
        self.reset_metrics()

        # Generate the grid. Will be random by default, or same environment if
        # 'fixed_environment' is True.
        self._gen_grid(self.width, self.height)

        # expecting adversarial agent to select colours
        if self.doing_shifts:
            self.goal_color = None
            self.wall_color = None
            self.floor_color = None
            Grid.tile_cache = {}

        # random shift; select random colours
        elif self.random_shifts:
            numbers = set()

            while (len(numbers) < 3):
                numbers.add(random.randint(0, len(COLORS)-1))

            numbers = list(numbers)
            self.goal_color = COLOR_NAMES[numbers[0]]
            self.wall_color = COLOR_NAMES[numbers[1]]
            self.grid.floor_color = COLOR_NAMES[numbers[2]]
            self.floor_color = COLOR_NAMES[numbers[2]]
            
            Grid.tile_cache = {}
            self.grid.wall_rect(0, 0, self.width, self.height, self.gen_wall)

        # No shift; generate default box
        else:
            self.goal_color = "green"
            self.wall_color = "grey"
            self.grid.floor_color = "black"
            self.floor_color = "black"
            self.grid.wall_rect(0, 0, self.width, self.height, self.gen_wall)


        image = self.grid.encode(False)  # TODO you can change False to True if the env does not use a domain shift
        obs = {"image": image, "time_step": [self.adversary_step_count], "random_z": self.generate_random_z()}

        return obs

    def reset_agent_status(self):
        """Reset the agent's position, direction, done, and carrying status."""
        self.agent_pos = [None] * self.n_agents
        self.agent_dir = [self.agent_start_dir] * self.n_agents
        self.done = [False] * self.n_agents
        self.carrying = [None] * self.n_agents

    def reset_agent(self):
        """Resets the agent's start position, but leaves goal and walls."""
        # Remove the previous agents from the world
        for a in range(self.n_agents):
            if self.agent_pos[a] is not None:
                self.grid.set(self.agent_pos[a][0], self.agent_pos[a][1], None)

        # Current position and direction of the agent
        self.reset_agent_status()

        if self.agent_start_pos is None:
            raise ValueError("Trying to place agent at empty start position.")
        else:
            self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)

        for a in range(self.n_agents):
            assert self.agent_pos[a] is not None
            assert self.agent_dir[a] is not None

            # Check that the agent doesn't overlap with an object
            start_cell = self.grid.get(*self.agent_pos[a])
            if not (start_cell.type == "agent" or start_cell is None or start_cell.can_overlap()):
                raise ValueError("Wrong object in agent start position.")

        # Step count since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs

    def remove_wall(self, x, y):
        if (x - 1, y - 1) in self.wall_locs:
            self.wall_locs.remove((x - 1, y - 1))
        obj = self.grid.get(x, y)
        if obj is not None and obj.type == "wall":
            self.grid.set(x, y, None)

    def compute_shortest_path(self):
        if self.agent_start_pos is None or self.goal_pos is None:
            return

        self.distance_to_goal = abs(self.goal_pos[0] - self.agent_start_pos[0]) + abs(
            self.goal_pos[1] - self.agent_start_pos[1]
        )

        # Check if there is a path between agent start position and goal. Remember
        # to subtract 1 due to outside walls existing in the Grid, but not in the
        # networkx graph.
        self.passable = nx.has_path(
            self.graph,
            source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
            target=(self.goal_pos[0] - 1, self.goal_pos[1] - 1),
        )
        if self.passable:
            # Compute shortest path
            self.shortest_path_length = nx.shortest_path_length(
                self.graph,
                source=(self.agent_start_pos[0] - 1, self.agent_start_pos[1] - 1),
                target=(self.goal_pos[0] - 1, self.goal_pos[1] - 1),
            )
        else:
            # Impassable environments have a shortest path length 1 longer than
            # longest possible path
            self.shortest_path_length = (self.width - 2) * (self.height - 2) + 1

    def generate_random_z(self):
        return np.random.uniform(size=(self.random_z_dim,)).astype(np.float32)

    def gen_goal(self):
        return minigrid.Goal(self.goal_color)

    def gen_wall(self):
        return minigrid.Wall(self.wall_color)
    
    def domain_settings(self):
        settings = {WALL: self.wall_color, GOAL: self.goal_color, FLOOR: self.floor_color}
        return settings

    def step_adversary(self, loc):
        """The adversary gets n_clutter + 2 moves to place the goal, agent, blocks.
        The action space is the number of possible squares in the grid. The squares
        are numbered from left to right, top to bottom.
        Args:
          loc: An integer specifying the location to place the next object which
            must be decoded into x, y coordinates.
        Returns:
          Standard RL observation, reward (always 0), done, and info
        """

        if loc >= self.adversary_action_dim:
            raise ValueError("Position passed to step_adversary is outside the grid.")

        # Add offset of 1 for outside walls
        x = int(loc % (self.width - 2)) + 1
        y = int(loc / (self.width - 2)) + 1
        done = False

        if self.doing_shifts and self.adversary_step_count < 3:  # this toggles if there's a domain shift or not

            color_str = list(COLORS.keys())[loc % len(COLORS)]
            counter = 1
            while color_str in self.used_colors:
                color_str = list(COLORS.keys())[(loc + counter) % len(COLORS)]
                counter += 1
            self.used_colors.add(color_str)

            # TODO to do a static shift, we can just
            if self.adversary_step_count == 0:
                self.wall_color = color_str
                self.grid.wall_rect(0, 0, self.width, self.height, self.gen_wall)

            elif self.adversary_step_count == 1:
                self.goal_color = color_str

            elif self.adversary_step_count == 2:
                self.floor_color = color_str
                self.grid.floor_color = self.floor_color
                Grid.tile_cache = {}

        else:  # if we're not choosing a color, we're putting down a tile
            if self.choose_goal_last:
                should_choose_goal = self.adversary_step_count == self.adversary_max_steps - 2
                should_choose_agent = self.adversary_step_count == self.adversary_max_steps - 1
                raise NotImplementedError("CHARLIE SAYS THIS DOES NOT HAPPEN BUT SHE'S AN IDIOT")
            else:
                should_choose_goal = self.adversary_step_count == (0 if not self.doing_shifts else 3)
                should_choose_agent = self.adversary_step_count == (1 if not self.doing_shifts else 4)

            # Place goal
            if should_choose_goal:
                # If there is goal noise, sometimes randomly place the goal
                if random.random() < self.goal_noise:
                    self.goal_pos = self.place_obj(self.gen_goal(), max_tries=100)
                else:
                    self.remove_wall(x, y)  # Remove any walls that might be in this loc
                    self.put_obj(self.gen_goal(), x, y)
                    self.goal_pos = (x, y)

            # Place the agent
            elif should_choose_agent:
                self.remove_wall(x, y)  # Remove any walls that might be in this loc

                # Goal has already been placed here
                if self.grid.get(x, y) is not None:
                    # Place agent randomly
                    self.agent_start_pos = self.place_one_agent(0, rand_dir=False)
                    self.deliberate_agent_placement = 0
                else:
                    self.agent_start_pos = np.array([x, y])
                    self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
                    self.deliberate_agent_placement = 1

            # Place wall
            elif self.adversary_step_count < self.adversary_max_steps:
                # If there is already an object there, action does nothing
                if self.grid.get(x, y) is None:
                    self.put_obj(self.gen_wall(), x, y)
                    self.n_clutter_placed += 1
                    self.wall_locs.append((x - 1, y - 1))

        self.adversary_step_count += 1

        # End of episode
        if self.adversary_step_count >= self.adversary_max_steps:
            done = True
            # Build graph after we are certain agent and goal are placed
            for w in self.wall_locs:
                self.graph.remove_node(w)
            self.compute_shortest_path()

        image = self.grid.encode(False)  # TODO change this if env was not shifted
        obs = {"image": image, "time_step": [self.adversary_step_count], "random_z": self.generate_random_z()}

        return obs, 0, done, {}

    def reset_random(self):
        """Use domain randomization to create the environment."""
        self.graph = grid_graph(dim=[self.width - 2, self.height - 2])

        self.step_count = 0
        self.adversary_step_count = 0

        # Current position and direction of the agent
        self.reset_agent_status()

        self.agent_start_pos = None
        self.goal_pos = None

        # Extra metrics
        self.reset_metrics()

        # Create empty grid
        self._gen_grid(self.width, self.height)

        # Randomly place goal
        self.goal_pos = self.place_obj(minigrid.Goal(), max_tries=100)

        # Randomly place agent
        self.agent_start_dir = self._rand_int(0, 4)
        self.agent_start_pos = self.place_one_agent(0, rand_dir=False)

        # Randomly place walls
        for _ in range(int(self.n_clutter / 2)):
            self.place_obj(minigrid.Wall(), max_tries=100)

        self.compute_shortest_path()
        self.n_clutter_placed = int(self.n_clutter / 2)

        return self.reset_agent()


class ReparameterizedAdversarialEnv(AdversarialEnv):
    """Grid world where an adversary builds the environment the agent plays.
    In this version, the adversary takes an action for each square in the grid.
    There is no limit on the number of blocks it can place. The action space has
    dimension 4; at each step the adversary can place the goal, agent, a wall, or
    nothing. If it chooses to place the goal or agent when they have previously
    been placed at a different location, they will move to the new location.
    """

    def __init__(self, n_clutter=50, size=15, agent_view_size=5, max_steps=250):
        super().__init__(n_clutter=n_clutter, size=size, agent_view_size=agent_view_size, max_steps=max_steps)

        # Adversary has four actions: place agent, goal, wall, or nothing
        self.adversary_action_dim = 4
        self.adversary_action_space = gym.spaces.Discrete(self.adversary_action_dim)

        # Reparam adversaries have additional inputs for the current x,y coords
        self.adversary_xy_obs_space = gym.spaces.Box(low=1, high=size - 2, shape=(1,), dtype="uint8")

        # Observations are dictionaries containing an encoding of the grid and the
        # agent's direction
        self.adversary_observation_space = gym.spaces.Dict(
            {
                "image": self.adversary_image_obs_space,
                "time_step": self.adversary_ts_obs_space,
                "random_z": self.adversary_randomz_obs_space,
                "x": self.adversary_xy_obs_space,
                "y": self.adversary_xy_obs_space,
            }
        )

        self.adversary_max_steps = (size - 2) ** 2

        self.wall_locs = []

    def reset(self):
        self.wall_locs = []
        obs = super().reset()
        obs["x"] = [1]
        obs["y"] = [1]
        return obs

    def select_random_grid_position(self):
        return np.array([self._rand_int(1, self.grid.width - 1), self._rand_int(1, self.grid.height - 1)])

    def get_xy_from_step(self, step):
        # Add offset of 1 for outside walls
        x = int(step % (self.width - 2)) + 1
        y = int(step / (self.width - 2)) + 1
        return x, y

    def step_adversary(self, action):
        """The adversary gets a step for each available square in the grid.
        At each step it chooses whether to place the goal, the agent, a block, or
        nothing. If it chooses agent or goal and they have already been placed, they
        will be moved to the new location.
        Args:
          action: An integer in range 0-3 specifying which object to place:
            0 = goal
            1 = agent
            2 = wall
            3 = nothing
        Returns:
          Standard RL observation, reward (always 0), done, and info
        """
        done = False

        if self.adversary_step_count < self.adversary_max_steps:
            x, y = self.get_xy_from_step(self.adversary_step_count)

            # Place goal
            if action == 0:
                if self.goal_pos is None:
                    self.put_obj(minigrid.Goal(), x, y)
                else:
                    goal = self.grid.get(self.goal_pos[0], self.goal_pos[1])
                    self.grid.set(self.goal_pos[0], self.goal_pos[1], None)
                    self.put_obj(goal, x, y)
                self.goal_pos = (x, y)

            # Place the agent
            elif action == 1:
                if self.agent_start_pos is not None:
                    agent = self.grid.get(self.agent_start_pos[0], self.agent_start_pos[1])
                    self.grid.set(self.agent_start_pos[0], self.agent_start_pos[1], None)
                else:
                    agent = None
                self.agent_start_pos = np.array([x, y])
                self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False, agent_obj=agent)

            # Place wall
            elif action == 2:
                self.put_obj(minigrid.Wall(), x, y)
                self.n_clutter_placed += 1

                self.wall_locs.append((x - 1, y - 1))

        self.adversary_step_count += 1

        # End of episode
        if self.adversary_step_count >= self.adversary_max_steps:
            done = True

            # If the adversary has not placed the agent or goal, place them randomly
            if self.agent_start_pos is None:
                self.agent_start_pos = self.select_random_grid_position()
                # If wall exists here, remove it
                self.remove_wall(self.agent_start_pos[0], self.agent_start_pos[1])
                self.place_agent_at_pos(0, self.agent_start_pos, rand_dir=False)
                self.deliberate_agent_placement = 0
            else:
                self.deliberate_agent_placement = 1

            if self.goal_pos is None:
                self.goal_pos = self.select_random_grid_position()
                # If wall exists here, remove it
                self.remove_wall(self.goal_pos[0], self.goal_pos[1])
                self.put_obj(minigrid.Goal(), self.goal_pos[0], self.goal_pos[1])

            # Build graph after we are certain agent and goal are placed
            for w in self.wall_locs:
                self.graph.remove_node(w)
            self.compute_shortest_path()
        else:
            x, y = self.get_xy_from_step(self.adversary_step_count)

        image = self.grid.encode()
        obs = {
            "image": image,
            "time_step": [self.adversary_step_count],
            "random_z": self.generate_random_z(),
            "x": [x],
            "y": [y],
        }

        return obs, 0, done, {}


class MiniAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50)


class MiniReparameterizedAdversarialEnv(ReparameterizedAdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50)


class NoisyAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(goal_noise=0.3)


class MediumAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=30, size=10, agent_view_size=5, max_steps=200)


class GoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(choose_goal_last=True)


class MiniGoalLastAdversarialEnv(AdversarialEnv):
    def __init__(self):
        super().__init__(n_clutter=7, size=6, agent_view_size=5, max_steps=50, choose_goal_last=True)


if hasattr(__loader__, "name"):
    module_path = __loader__.name
elif hasattr(__loader__, "fullname"):
    module_path = __loader__.fullname

register.register(env_id="MultiGrid-Adversarial-v0", entry_point=module_path + ":AdversarialEnv")

register.register(
    env_id="MultiGrid-ReparameterizedAdversarial-v0", entry_point=module_path + ":ReparameterizedAdversarialEnv"
)

register.register(env_id="MultiGrid-MiniAdversarial-v0", entry_point=module_path + ":MiniAdversarialEnv")

register.register(
    env_id="MultiGrid-MiniReparameterizedAdversarial-v0", entry_point=module_path + ":MiniReparameterizedAdversarialEnv"
)

register.register(env_id="MultiGrid-NoisyAdversarial-v0", entry_point=module_path + ":NoisyAdversarialEnv")

register.register(env_id="MultiGrid-MediumAdversarial-v0", entry_point=module_path + ":MediumAdversarialEnv")

register.register(env_id="MultiGrid-GoalLastAdversarial-v0", entry_point=module_path + ":GoalLastAdversarialEnv")

register.register(
    env_id="MultiGrid-MiniGoalLastAdversarial-v0", entry_point=module_path + ":MiniGoalLastAdversarialEnv"
)
