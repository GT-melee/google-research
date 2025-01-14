"""
1. load env
2. run trained agent in random env 1000 times vs trained agent without domain shift
3. Compare % success
"""
import gym
import tensorflow as tf
import numpy as np
from tf_agents.policies import policy_loader
from social_rl import gym_multigrid
import matplotlib.pyplot as plt
from tf_agents.trajectories import time_step as ts_lib, trajectory

tf.compat.v1.enable_v2_behavior()
#source_shift = []
#target_shift = []
from social_rl.adversarial_env import adversarial_env

all_source_agents = ["/home/charlie/SDRIVE/datasets/any_shifts/paired/policy_saved_model/adversary_env/0/policy_000499950"]
all_variable_agents = []

all_source_agents = [tf.compat.v2.saved_model.load(a) for a in all_source_agents]
all_variable_agents = [tf.compat.v2.saved_model.load(a) for a in all_variable_agents]

sequences = [
[
    12,
    0,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24,
    25
  ] # corridor
]

def get_envs():
  names = []
  targets = []
  sources = []

  for i, sequence in enumerate(sequences):
    #for shift in [source_shift, target_shift]:
    py_env = adversarial_env.load("MultiGrid-Adversarial-v0")
    tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)
    tf_env.reset()

    done = False
    for s in sequence:
      _, _, done, _ = tf_env.step_adversary(s)

#    plt.imshow(tf_env.render('rgb_array'))
   # plt.show()
    while not done:
      _, _, done, _ = tf_env.step_adversary(s)



    plt.imshow(py_env.render())
    plt.show()

    #todo apply shift
    #targets.append(TODO)
    #sources.append(TODO)
    targets.append(tf_env)
    sources.append(tf_env)
    names.append(i)

  return zip(sources, targets, names)

def load_agent(checkpoint_path):
  agent = tf.compat.v2.saved_model.load(checkpoint_path)
  return agent

def backwards_compatible_timestep(new_ts):
  """Remove new observations added in later versions of the environment."""
  old_obs = {
      'image': new_ts.observation['image'],
      'time_step': new_ts.observation['time_step']
  }
  return ts_lib.TimeStep(
      new_ts.step_type,
      new_ts.reward,
      new_ts.discount,
      old_obs)

def run_agent_on_env(agent, env):
  done = False
  rewards = []

  timestep = env.reset_agent()
  policy_state = agent.get_initial_state(1)

  while not done:
    policy_step = agent.action(timestep, policy_state=policy_state)

    #print("printy")
    plt.imshow(env.render('rgb_array')[0])
    plt.show()


    policy_state = policy_step.state

    next_time_step = env._step(policy_step.action)
    #traj = trajectory.from_transition(timestep, policy_step, next_time_step)
    timestep = next_time_step


    #rewards.append(reward)
  return np.array(rewards)

by_env = {}
for i, seq in enumerate(sequences):
  by_env[i] = {}
  for j, agent in enumerate(all_source_agents + all_variable_agents):
    by_env[i] = {}
    by_env[i]["source"] = {}
    by_env[i]["target"] = {}

def eval_agent(agent, agent_name):
  global by_env

  for source, target, env_name in get_envs():
    source_rews = run_agent_on_env(agent, source)
    target_rews = run_agent_on_env(agent, target)

    by_env[env_name][agent_name]["source"] = source_rews
    by_env[env_name][agent_name]["target"] = target_rews

for i, agent in enumerate(all_source_agents+all_variable_agents):
  eval_agent(agent, i)