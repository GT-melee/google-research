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

#global ctr
ctr = 0
def prt_place(message=''):
  global ctr
  eqstr = '=' * max(12, len(message))
  eqstr = '\n' + eqstr + '\n'
  if len(message) > 0:
    message = '\n' + message

  print(eqstr + "ADAM PRINT", ctr, message, eqstr)
  ctr += 1

#all_source_agents = ["/home/charlie/SDRIVE/datasets/any_shifts/paired/policy_saved_model/adversary_env/0/policy_000499950"]
#all_source_agents  = ["/home/adam/Documents/2021-winter/ift-6756/gt-org/any_shifts_apr_17/paired/policy_saved_model/adversary_env/0/policy_000499950"]
all_source_agents  = ["/home/adam/Documents/2021-winter/ift-6756/gt-org/dave_weights_apr_23"]
all_variable_agents = []



all_source_agents = [tf.compat.v2.saved_model.load(a) for a in all_source_agents]
all_variable_agents = [tf.compat.v2.saved_model.load(a) for a in all_variable_agents]

sequences = [
  [12,0,13,14,15,16,17,18,19,20,21,22,23,24,25]#,
  # corridor
  #[ 10, 0, 1, 3, 5, 6, 7, 9, 14, 16, 22, 23, 27, 29, 30, 31, 32, 33, 36, 40, 44, 49, 50, 57, 60, 65, 66, 67, 70, 73, 75, 77, 80, 81, 85, 86, 88, 90, 98, 101, 105, 107, 108, 109, 111, 114, 118, 121, 124, 125, 126, 127, 131, 134, 136, 137, 144, 145, 146, 147, 152, 153, 154, 155, 163 ]
  # maze
]


def get_envs():
  names = []
  targets = []
  sources = []
  prt_place('in get_envs')

  for i, sequence in enumerate(sequences):
    #for shift in [source_shift, target_shift]:
    prt_place('get_envs for loop')
    py_env = adversarial_env.load("MultiGrid-Adversarial-v0")

    # prt_place('py_env on load')
    # plt.imshow(py_env.render())
    # plt.show()


    tf_env = adversarial_env.AdversarialTFPyEnvironment(py_env)
    tf_env.reset()
    # ^can't be shown with plt.imshow() at this point for some reason

    # prt_place('tf_env on reset')
    # plt.imshow(tf_env.render('rgb_array'))
    # plt.show()

    done = False
    prt_place('all s in sequence:')
    # I think: populate the empty grid
    for s in sequence:
      print(s)
      _, _, done, _ = tf_env.step_adversary(s)
    prt_place('does this work?')

#    plt.imshow(tf_enunzip()v.render('rgb_array'))
   # plt.show()
    while not done:
      _, _, done, _ = tf_env.step_adversary(s)


    prt_place('gonna show in get_envs')
    # plt.imshow(py_env.render())
    # plt.show()

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
  prt_place('gonna reset')
  timestep = env.reset_agent()
  policy_state = agent.get_initial_state(1)

  while not done:
    policy_step = agent.action(timestep, policy_state=policy_state)

    print('Policy step:')
    #print(policy_step.action)
    tf.print(policy_step.action)
    #print("printy")
    #prt_place('gonna show')
    #plt.imshow(env.render('rgb_array')[0])
    #plt.show()
    #prt_place('just showed (jk)')

    policy_state = policy_step.state
    # print('Policy state:')
    # print(policy_state)


    next_time_step = env._step(policy_step.action)
    #traj = trajectory.from_transition(timestep, policy_step, next_time_step)
    timestep = next_time_step


    #rewards.append(reward)
  return np.array(rewards)

prt_place()

# turns env generation sequences into dictionary
by_env = {}
for i, seq in enumerate(sequences):
  by_env[i] = {}
  for j, agent in enumerate(all_source_agents + all_variable_agents):
    by_env[i] = {}
    by_env[i]["source"] = {}
    by_env[i]["target"] = {}

def eval_agent(agent, agent_name):
  global by_env
  prt_place('in eval')
  for source, target, env_name in get_envs():
    prt_place('gonna source')
    # TODO: MAKE THE ENV STATIC AND THE PROTAGONIST DIFFERENT
    # TODO: WE WANT THE SOURCE ENV TO BE THE SAME, EXCEPT COLOURS ARE DIFFERENT
    source_rews = run_agent_on_env(agent, source)
    prt_place('after source_rews')
    target_rews = run_agent_on_env(agent, target)

    # by_env[env_name][agent_name]["source"] = source_rews
    # by_env[env_name][agent_name]["target"] = target_rews
  prt_place('end eval agent')

def eval_agent_w_env(agent, agent_name, env):
  global by_env
  prt_place('in eval')
  for source, target, env_name in get_envs():
    prt_place('gonna source')
    # TODO: MAKE THE ENV STATIC AND THE PROTAGONIST DIFFERENT
    # TODO: WE WANT THE SOURCE ENV TO BE THE SAME, EXCEPT COLOURS ARE DIFFERENT
    source_rews = run_agent_on_env(agent, source)
    prt_place('after source_rews')
    target_rews = run_agent_on_env(agent, target)

    by_env[env_name][agent_name]["source"] = source_rews
    by_env[env_name][agent_name]["target"] = target_rews
  prt_place('end eval agent')

## Loop used for charlie's domain shift experiments
# for i, agent in enumerate(all_source_agents+all_variable_agents):
#   prt_place()
#   eval_agent(agent, i)

def adam():
  #source env, target env, env names
  source, t_env, names = list(get_envs())[0]
  # print(zip)
  # plt.imshow(target[1:].render('rgb_array'))
  # plt.show()

  # trained protagonist
  prot = all_source_agents[0]
  #print(prot)
  run_agent_on_env(prot, t_env)

if __name__ == "__main__":
  adam()
