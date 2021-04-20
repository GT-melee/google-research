"""
1. load env
2. run trained agent in random env 1000 times vs trained agent without domain shift
3. Compare % success
"""
import gym
import tensorflow as tf
import numpy as np

#source_shift = []
#target_shift = []

all_source_agents = ["/home/charlie/SDRIVE/datasets/any_shifts/paired/policy_saved_model/agent/0/policy_000499950"]
all_variable_agents = []

all_source_agents = [tf.compat.v2.saved_model.load(a) for a in all_source_agents]
all_variable_agents = [tf.compat.v2.saved_model.load(a) for a in all_variable_agents]

sequences = [
  []

]

def get_envs():
  names = []
  targets = []
  sources = []

  for i, sequence in enumerate(sequences):
    for shift in [source_shift, target_shift]:
      env = gym.make("MultiGrid-Adversarial-v0")
      #todo apply shift
      #targets.append(TODO)
      #sources.append(TODO)
      targets.append(env)
      sources.append(env)
    names.append(i)

  return sources, targets, names

def load_agent(checkpoint_path):
  agent = tf.compat.v2.saved_model.load(checkpoint_path)
  return agent

def run_agent_on_env(agent, env):
  done = False
  rewards = []

  obs = env.reset_agent()
  policy_state = agent.get_initial_state(1)

  while not done:
    policy_step = agent.action(obs, policy_state=policy_state)

    policy_state = policy_step.state

    obs, reward, done, _ = env.step(policy_step.action)
    rewards.append(reward)
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