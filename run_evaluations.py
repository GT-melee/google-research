from eval import *
import tqdm
import multiprocessing.pool


# Overwrite the eval agent to manually fix colours because nothing is sacred
class BaselineEvalAgent(EvalAgent):
    def eval_env(self, env: BaseEnv):
        # ensure no existing env is lingering
        env.close()
        metrics = Metrics(self.name, env.name, env.colors)

        # make filepaths a bit more informative
        if env.video_fp is not None and self.prepend_name:
            video_fp = Path(env.video_fp)
            parent = video_fp.parent
            name = video_fp.name
            env.set_video_fp(str(parent / f"{self.name}_{name}"))

        # loop over n eval episodes for this configuration
        for ep in tqdm.trange(self.num_eval_ep):
            # Creates the environment
            py_env, tf_env = env.reset()

            # Reinit everything
            timestep = env.soft_reset()
            policy_state = self.agent.get_initial_state(1)

            actions = []

            for i in range(self.max_steps_per_ep):
                # fix colour encodings 
                obs = timestep.observation
                img = np.zeros(obs["image"].shape, dtype=np.uint8)
                x = np.array(obs["image"])
                img[np.all(x == [0, 0, 0], axis=-1)] = [1, 0, 0]
                img[np.all(x == [4, 4, 0], axis=-1)] = [2, 5, 0]
                img[np.all(x == [3, 3, 0], axis=-1)] = [8, 1, 0]
                timestep.observation["image"] = tf.convert_to_tensor(img)  # ( ͡° ͜ʖ ͡°)

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
                  #  print("FCUKY WUCKY")
                    break

            spl = -1
            if hasattr(tf_env, "get_shortest_path_length"):
                spl = tf_env.get_shortest_path_length().numpy()[0]
            metrics.log_timesteps(actions, spl, rew)

        env.close()
        self.accumulated_metrics.append(metrics)
        return metrics


def do_inner_loop(name, weights, env_name, example_colors):
    num_ep = 500

    # FIXME: JANK
    if "PAIRED_BASELINE" in name and example_colors is None:
        name = Path(name).name
        print("jank agent", name)
        agent = BaselineEvalAgent(name, weights, num_eval_ep=num_ep, max_steps_per_ep=250)
    else:
        name = Path(name).name
        print("non jank agent: ", example_colors, name)
        agent = EvalAgent(name, weights, num_eval_ep=num_ep, max_steps_per_ep=250)

    color_env = BaseEnv(env_name, colors=example_colors, video_fp=None)  # f"videos/{name}_COLOR.mp4")

    agent.eval_env(color_env)
    return agent.accumulated_metrics


def run_for_one_weight(pool, weight, envs):
    RUN_NAME = weight  # gets the last part of the filepath

    inputs = []
    for example_colors in [
        None,       # static
        (8,7,2),  # baseline colors
        (4,3,0),  # static sanity
        (2,3,4),    # collapse
        (1,6,7),    # new seen colors
        (12,13,14)  # new unseen colors
    ]:
        run_names = [RUN_NAME] * len(envs)
        weights = [weight] * len(envs)
        example_colors = [example_colors] * len(envs)
        inputs.extend([(n, w, e, c) for n, w, e, c in zip(run_names, weights, envs, example_colors)])

    output = pool.starmap(do_inner_loop, inputs)

    mergy_mc_mergeface = output[0]
    for agent in output[1:]:
        mergy_mc_mergeface.extend(agent)
    # agent = EvalAgent(weight.replace('/', '-'), weight, num_eval_ep=1, max_steps_per_ep=250)
    # agent.accumulated_metrics = mergy_mc_mergeface

    for metrics in mergy_mc_mergeface:
        print(metrics)
        print(metrics.get_stats())
        metrics.summarize()

    dico = summary(mergy_mc_mergeface)
    out_str = ""
    for key, item in dico.items():
        out_str += f"{key.upper()}: {item['all']}\n"
    print(out_str)
    with open(f"./videos/eval_results_{weight.replace('/', '-')}.txt", "w") as f:
        f.write(out_str)

def main():
    pool = multiprocessing.pool.Pool(6)
    for weight in [
        "/home/ddobre/Projects/game_theory/saved_models/PAIRED_BASELINE/agents/policy_000000000", 
        "/home/ddobre/Projects/game_theory/saved_models/PAIRED_BASELINE/agents/policy_000050100", 
        "/home/ddobre/Projects/game_theory/saved_models/PAIRED_BASELINE/agents/policy_000100200", 
        "/home/ddobre/Projects/game_theory/saved_models/PAIRED_BASELINE/agents/policy_000230700", 
        "/home/ddobre/Projects/game_theory/saved_models/PAIRED_BASELINE/agents/policy_000438000", 
    ]:
        run_for_one_weight(pool, weight, MINI_TEST_ENVS+MINI_VAL_ENVS)


if __name__ == '__main__':
    main()

