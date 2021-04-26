"""
1. load env
2. run trained agent in random env 1000 times vs trained agent without domain shift
3. Compare % success
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf
from tf_agents.environments import tf_py_environment
from tf_agents.policies import policy_loader
from tf_agents.trajectories import time_step as ts_lib

# gym stuff
import gym
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from tf_agents.typing.types import PyEnv

# multigrid
from social_rl import gym_multigrid
from social_rl.adversarial_env import adversarial_env
from social_rl.multiagent_tfagents import multiagent_gym_suite

try:
    from social_rl.gym_multigrid.gym_minigrid.minigrid import COLORS, IDX_TO_COLOR, COLOR_TO_IDX
except ImportError as e:
    logging.warning(
        """Not able to import any color info - you are likely running in the original rep.
This will work, but just ensure that all color options are set to None (in the BaseEnv and AdvEnv)."""
    )

tf.compat.v1.enable_v2_behavior()
logging.basicConfig(level=logging.INFO)


VAL_ENVS = [
    "MultiGrid-TwoRooms-Minigrid-v0",
    "MultiGrid-Cluttered40-Minigrid-v0",
    "MultiGrid-Cluttered10-Minigrid-v0",
    "MultiGrid-SixteenRooms-v0",
    "MultiGrid-Maze2-v0",
    "MultiGrid-Maze3-v0",
    "MultiGrid-Labyrinth2-v0",
]
TEST_ENVS = [
    "MultiGrid-FourRooms-Minigrid-v0",
    "MultiGrid-Cluttered50-Minigrid-v0",
    "MultiGrid-Cluttered5-Minigrid-v0",
    "MultiGrid-Empty-Random-15x15-Minigrid-v0",
    "MultiGrid-SixteenRoomsFewerDoors-v0",
    "MultiGrid-Maze-v0",
    "MultiGrid-Labyrinth-v0",
]
MINI_VAL_ENVS = [
    "MultiGrid-MiniTwoRooms-Minigrid-v0",
    "MultiGrid-Empty-Random-6x6-Minigrid-v0",
    "MultiGrid-MiniCluttered6-Minigrid-v0",
    "MultiGrid-MiniCluttered-Lava-Minigrid-v0",
    "MultiGrid-MiniMaze-v0",
]
MINI_TEST_ENVS = [
    "MultiGrid-MiniFourRooms-Minigrid-v0",
    "MultiGrid-MiniCluttered7-Minigrid-v0",
    "MultiGrid-MiniCluttered1-Minigrid-v0",
]


# TODO: unused
def backwards_compatible_timestep(new_ts):
    """Remove new observations added in later versions of the environment."""
    old_obs = {"image": new_ts.observation["image"], "time_step": new_ts.observation["time_step"]}
    return ts_lib.TimeStep(new_ts.step_type, new_ts.reward, new_ts.discount, old_obs)


def load_agent(checkpoint_path):
    agent = tf.compat.v2.saved_model.load(checkpoint_path)
    return agent


# Define the Environment loaders for base and adversarial environments
class BaseEnv:
    def __init__(self, name: str, colors: str = None, video_fp: str = None, gym_kwargs=None):
        """
        Args:
            name: defines the gym environment (eg. 'MultiGrid-FourRooms-Minigrid-v0')
            colors: optional list corresponding to (wall, goal, floor) color in idx form
            video_fp: specifies where the video recording will be saved. If None, 
                no recording
            gym_kwargs: anything else to get passed into the gym creation call
        """
        self.name = name
        self.video_fp = video_fp

        if gym_kwargs is None:
            gym_kwargs = {}
        self.gym_kwargs = gym_kwargs
        self.colors = colors

        self.py_env = None
        self.tf_env = None

    def set_video_fp(self, fp: str):
        print(f"Resetting video filepath: {self.video_fp} -> {fp}")
        self.video_fp = fp

    def init_env(self):
        """Instantiates the environment, starts the video recorder. Generally
        don't need to call this, this is called by reset()"""

        # Hack to work around the  bug in adversarial.py for picking a color

        kwargs = self.gym_kwargs.copy()
        if self.colors is not None:
            _get_color = lambda x: list(COLORS.keys())[x % len(COLORS)]
            kwargs.update(
                {
                    "wall_color": _get_color(self.colors[0]),
                    "goal_color": _get_color(self.colors[1]),
                    "floor_color": _get_color(self.colors[2]),
                }
            )

        py_env = multiagent_gym_suite.load(self.name, gym_kwargs=kwargs)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        py_env.reset()
        tf_env.reset()
        if self.video_fp is not None:
            py_env = VideoRecorder(py_env, self.video_fp)
            py_env.capture_frame()

        self.py_env, self.tf_env = py_env, tf_env
        return py_env, tf_env

    def reset(self):
        """After loading the environment, do anything that you may need to do
        prior to running an agent through it (relevant in adversarial env)
        
        Generally will want to call this method rather than just init if 
        manually looping shit
        """
        if self.py_env is None or self.tf_env is None:
            return self.init_env()

        self.tf_env.reset()
        return self.py_env, self.tf_env

    def soft_reset(self):
        """This reset is called in the eval loop as opposed to the beginning"""
        if self.tf_env is None:
            raise RuntimeError("Must run env.reset() first")
        return self.tf_env.reset()

    def close(self):
        """Kills the current envs, saves video. Reinits envs to None, so the
        next reset call will automatically regenerate the envs"""
        if self.py_env is not None:
            self.py_env.close()
            del self.py_env, self.tf_env
            self.py_env, self.tf_env = None, None


class AdvEnv(BaseEnv):
    def __init__(
        self,
        sequence: list = None,
        adversary_env_path: str = None,
        colors: list = None,
        video_fp: str = None,
    ):
        """
        Note: exactly one of 'sequence' or 'adversary_env_path' must be specified

        Args:
            sequence: list of locs where to place the agent, goal, and then walls
            adversary_env_path: path to the saved_model for the adversary
            colors (optional): specifies the (wall, goal, floor) color. If None,
                defaults to the regular color scheme
            video_fp: specifies where the video recording will be saved. If None, 
                no recording
        """
        gym_kwargs = {}
        if colors is not None:
            gym_kwargs["domain_shifts"] = True

        super().__init__(
            name="MultiGrid-Adversarial-v0", colors=colors, video_fp=video_fp, gym_kwargs=gym_kwargs
        )

        self.py_env = None
        self.tf_env = None

        # TODO: figure out if using sequence or agent
        if sequence is not None and adversary_env_path is not None:
            raise TypeError

        self.agent = None
        self.sequence = None
        self.agent = None

        if sequence is not None:
            self.mode = "sequence"
            self.sequence = sequence

        elif adversary_env_path is not None:
            self.mode = "agent"
            self.agent = load_agent(adversary_env_path)

        else:
            raise NotImplementedError

    def init_env(self):
        """Also starts video recorder"""
        py_env = adversarial_env.load(self.name, gym_kwargs=self.gym_kwargs)
        tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)

        py_env.reset()
        tf_env.reset()
        if self.video_fp is not None:
            py_env = VideoRecorder(py_env, self.video_fp)

        self.py_env, self.tf_env = py_env, tf_env
        return py_env, tf_env

    def reset(self):
        # check if envs have been instantiated yet
        first_run = False
        if self.py_env is None and self.tf_env is None:
            first_run = True
            self.init_env()

        timestep = self.tf_env.reset()
        actions = []

        num_actions = 0
        max_actions = 52  # FIXME: n_clutter + 2; hack for now
        last_action = -0

        # TODO: does not support color generating adversary yet
        if self.colors is not None:
            for c in self.colors:
                timestep = self.tf_env.step_adversary(c)
                num_actions += 1
                if self.video_fp is not None:
                    self.py_env.capture_frame()

        # TODO: process sequence or process agent
        if self.mode == "sequence":
            actions = self.sequence
            for s in self.sequence:
                _, _, done, _ = self.tf_env.step_adversary(s)
                # for some reason if I remake the environment, the agent behaves differently
                # so I recreate the environment but just don't capture frames (if I just used
                # reset_agent(), the agent will take the same actions)
                if self.video_fp is not None and first_run:
                    self.py_env.capture_frame()

                # required in order to generate a valid graph
                num_actions += 1
                last_action = s

            while num_actions < max_actions:
                # fake the last actions to generate a valid graph
                self.tf_env.step_adversary(last_action)
                num_actions += 1

        elif self.mode == "agent":
            policy_state = self.agent.get_initial_state(1)

            for i in range(num_actions + 1, max_actions):
                policy_step = self.agent.action(timestep, policy_state=policy_state)

                policy_state = policy_step.state
                a = policy_step.action.numpy()[0]
                actions.append(a)

                timestep = self.tf_env.step_adversary(policy_step.action)
                _, rew, disc, obs = timestep

                # record video if relevant
                if hasattr(self.py_env, "capture_frame"):
                    self.py_env.capture_frame()

                # required in order to generate a valid graph
                num_actions += 1
                last_action = a

        else:
            raise NotImplementedError

        logging.info(f"Generated env with steps: {actions}")
        return self.py_env, self.tf_env

    def soft_reset(self):
        """This reset is called in the eval loop as opposed to the beginning"""
        if self.tf_env is None:
            raise RuntimeError("Must run env.reset() first")
        return self.tf_env.reset_agent()


class Metrics:
    """so useless"""

    def __init__(self, agent_name: str, env_name: str, colors):
        self.agent_name = agent_name
        self.env_name = env_name
        self.colors = colors
        self.data = []

    def log_timesteps(self, actions, spl, reward):
        # any other metrics??
        self.data.append({"actions": actions, "spl": spl, "reward": reward})


class EvalAgent:
    """
    Defines a class which encapsulates an agent and defines various evaluation
    hyperparemeters that would control an eval run on generic BaseEnvs
    """

    def __init__(
        self,
        name: str,
        model_path: str,
        num_eval_ep: int = 5,
        max_steps_per_ep: int = 200,
        prepend_video_with_name: bool = True,
    ):
        self.name = name
        self.model_path = model_path
        self.agent = load_agent(model_path)

        self.num_eval_ep = num_eval_ep
        self.max_steps_per_ep = max_steps_per_ep
        self.prepend_name = prepend_video_with_name

        self.accumulated_metrics = []

    def eval_env(self, env: BaseEnv):
        # ensure no existing env is lingering
        env.close()
        metrics = Metrics(self.name, env.name, env.colors)

        # make filepaths a bit more informative
        if self.prepend_name:
            video_fp = Path(env.video_fp)
            parent = video_fp.parent
            name = video_fp.name
            env.set_video_fp(str(parent / f"{self.name}_{name}"))

        # loop over n eval episodes for this configuration
        for ep in range(self.num_eval_ep):
            # Creates the environment
            py_env, tf_env = env.reset()

            # Reinit everything
            timestep = env.soft_reset()
            policy_state = self.agent.get_initial_state(1)

            actions = []

            for i in range(self.max_steps_per_ep):
                # main loop
                policy_step = self.agent.action(timestep, policy_state=policy_state)
                policy_state = policy_step.state
                timestep = tf_env.step(policy_step.action)

                # logging stuff
                a = policy_step.action.numpy()[0]
                actions.append(a)

                # record video if relevant
                if hasattr(py_env, "capture_frame"):
                    py_env.capture_frame()

                # check if end of episode (via reward > 0)
                _, rew, disc, obs = timestep
                rew = rew.numpy()[0]  # note that we only have one agent
                if rew > 0.0:
                    break

            spl = -1
            if hasattr(tf_env, "get_shortest_path_length"):
                spl = tf_env.get_shortest_path_length().numpy()[0]
            metrics.log_timesteps(actions, spl, rew)

        env.close()
        self.accumulated_metrics.append(metrics)
        return metrics


def main():
    example_colors = [3, 1, 5]  # (purple green gray) -> (wall, goal, floor)

    # Example sequence-based adversarial env
    charlie_static_bad_enc = "/home/ddobre/Projects/game_theory/saved_models/static_apr22/policy_saved_model/agent/0/policy_000499950/"
    agent = EvalAgent("static_000499950", charlie_static_bad_enc)

    # define environments
    sequence = [
        71,
        111,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        55,
        61,
        68,
        74,
        81,
        84,
        85,
        87,
        94,
        100,
        107,
        113,
        120,
        121,
        122,
        123,
        124,
        125,
        126,
    ]
    seq_env = AdvEnv(sequence=sequence, video_fp="videos/seq_room.mp4")
    domain_shift_env = AdvEnv(
        sequence=sequence, colors=example_colors, video_fp="videos/color_seq_room.mp4"
    )

    # evaluate on these envs and agents
    agent.eval_env(seq_env)
    agent.eval_env(domain_shift_env)

    # example on baseline envs
    name = MINI_TEST_ENVS[0]
    base_env = BaseEnv(name, video_fp=f"videos/{name}.mp4")
    color_base_env = BaseEnv(name, colors=example_colors, video_fp=f"videos/{name}_COLOR.mp4")

    # evaluate on these envs and agents
    agent.eval_env(base_env)
    agent.eval_env(color_base_env)
