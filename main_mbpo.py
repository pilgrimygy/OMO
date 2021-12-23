import argparse
import time
import gym
import torch
import numpy as np
from itertools import count
from tqdm import tqdm

import os
import os.path as osp
import json

from sac.replay_memory import ReplayMemory
from sac.sac import SAC
from model import EnsembleDynamicsModel
from predict_env import PredictEnv
from sample_env import EnvSampler
from tf_models.constructor import construct_model, format_samples_for_training

import click
from utils import Logger, load_config


def train(args, env_sampler, eval_env_sampler, predict_env, agent, env_pool, model_pool, logger):
    rollout_length = 1

    start = time.time()

    exploration_before_start(args, env_sampler, env_pool, agent)

    total_step = args["init_exploration_steps"]

    logger.log_var("time", time.time() - start, total_step)

    for epoch_step in tqdm(range(args["num_epoch"])):
        logger.log_str(f"Epoch: {epoch_step+1}")
        start_step = total_step
        train_policy_steps = 0
        for i in count():
            cur_step = total_step - start_step

            if cur_step >= args['epoch_length'] and len(env_pool) > args['min_pool_size']:
                break

            if cur_step % args['model_train_freq'] == 0 and args['real_ratio'] < 1.0:  
                logger.log_str("Training model")
                train_predict_model(args, env_pool, predict_env)

                new_rollout_length = set_rollout_length(args, epoch_step)
                if rollout_length != new_rollout_length:
                    logger.log_str(f"Rollout length: {rollout_length} -> {new_rollout_length}")
                    rollout_length = new_rollout_length
                    model_pool = resize_model_pool(args, rollout_length, model_pool)

                logger.log_str("Rollout")
                rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, total_step, logger)

            cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)

            env_pool.push(cur_state, action, reward, next_state, done)

            if len(env_pool) > args["min_pool_size"]:
                train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, env_pool, model_pool, agent, logger)

            total_step += 1

            if total_step % args["epoch_length"] == 0:
                logger.log_str("Evalating agent")
                eval_rewards = []
                while len(eval_rewards) < args["eval_num"]:
                    eval_env_sampler.current_state = None
                    sum_reward = 0
                    done = False
                    test_step = 0
                    while (not done) and (test_step != args["max_path_length"]):
                        cur_state, action, next_state, reward, done, info = eval_env_sampler.sample(agent, eval_t=True)
                        sum_reward += reward
                        test_step += 1
                    
                    eval_rewards.append(sum_reward)
                
                eval_reward = sum(eval_rewards) / len(eval_rewards)

                logger.log_var("eval_reward", eval_reward, total_step)
                logger.log_var("time", time.time() - start, total_step)


def exploration_before_start(args, env_sampler, env_pool, agent):
    for i in range(args["init_exploration_steps"]):
        cur_state, action, next_state, reward, done, info = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)


def set_rollout_length(args, epoch_step):
    rollout_length = (min(max(args["rollout_min_length"] + (epoch_step - args["rollout_min_epoch"]) / (args["rollout_max_epoch"] - args["rollout_min_epoch"]) * (args["rollout_max_length"] - args["rollout_min_length"]), args["rollout_min_length"]), args["rollout_max_length"]))
    return int(rollout_length)


def train_predict_model(args, env_pool, predict_env):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)

    predict_env.model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)


def resize_model_pool(args, rollout_length, model_pool):
    rollouts_per_epoch = args["rollout_batch_size"] * args["epoch_length"] / args["model_train_freq"]
    model_steps_per_epoch = int(rollout_length * rollouts_per_epoch)
    new_pool_size = args["model_retain_epochs"] * model_steps_per_epoch

    sample_all = model_pool.return_all()
    new_model_pool = ReplayMemory(new_pool_size)
    new_model_pool.push_batch(sample_all)

    return new_model_pool


def rollout_model(args, predict_env, agent, model_pool, env_pool, rollout_length, total_step, logger):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args["rollout_batch_size"])
    num_total_samples = 0
    for i in range(rollout_length):
        # TODO: Get a batch of actions
        action = agent.select_action(state)
        next_states, rewards, terminals, info = predict_env.step(state, action)
        # TODO: Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], terminals[j]) for j in range(state.shape[0])])
        nonterm_mask = ~terminals.squeeze(-1)
        if nonterm_mask.sum() == 0:
            break
        num_total_samples += next_states.shape[0]
        state = next_states[nonterm_mask]
    logger.log_var("Rollout length", num_total_samples / args["rollout_batch_size"], total_step)


def train_policy_repeats(args, total_step, train_step, env_pool, model_pool, agent, logger):
    if train_step > args["max_train_repeat_per_step"] * total_step:
        return 0
    
    qf1_loss_step = 0.
    qf2_loss_step = 0.
    policy_loss_step = 0.
    alpha_loss_step = 0.

    for i in range(args["num_train_repeat"]):
        env_batch_size = int(args["agent"]["batch_size"] * args["real_ratio"])
        model_batch_size = args["agent"]["batch_size"] - env_batch_size

        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), np.concatenate((env_action, model_action), axis=0), np.concatenate((np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), np.concatenate((env_next_state, model_next_state), axis=0), np.concatenate((np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done

        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        qf1_loss, qf2_loss, policy_loss, alpha_loss, _ = agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args["agent"]["batch_size"], i)

        qf1_loss_step += qf1_loss
        qf2_loss_step += qf2_loss
        policy_loss_step += policy_loss
        alpha_loss_step += alpha_loss
    
    if total_step % args["log_interval"] == 0:
        logger.log_var("q1_loss", qf1_loss_step / args["num_train_repeat"], total_step)
        logger.log_var("q2_loss", qf2_loss_step / args["num_train_repeat"], total_step)
        logger.log_var("policy_loss", policy_loss_step / args["num_train_repeat"], total_step)
        logger.log_var("alpha_loss", alpha_loss_step / args["num_train_repeat"], total_step)

    return args["num_train_repeat"]


from gym.spaces import Box


class SingleEnvWrapper(gym.Wrapper):
    def __init__(self, env):
        super(SingleEnvWrapper, self).__init__(env)
        obs_dim = env.observation_space.shape[0]
        obs_dim += 2
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]  # Need this in the obs for determining when to stop
        obs = np.append(obs, [torso_height, torso_ang])

        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()
        torso_height, torso_ang = self.env.sim.data.qpos[1:3]
        obs = np.append(obs, [torso_height, torso_ang])
        return obs


@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str)
@click.option("--log-dir", default="results/")
@click.option("--gpu", type=int, default=0)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="")
@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, seed, info, args):
    print(args)
    
    alg_name = "mbpo"

    args = load_config(config_path, alg_name, args)

    env_name = args['env_name']

    # Logger
    logger = Logger(log_dir, prefix=env_name+"-"+alg_name+"-"+info, print_to_terminal=print_log)
    logger.log_str("logging to {}".format(logger.log_path))
    logger.log_str_object("parameters", log_dict=args)

    # Initial environment
    logger.log_str("Initializing Environment")
    env = gym.make(env_name)
    eval_env = gym.make(env_name)

    # Set random seed
    torch.manual_seed(seed)
    np.random.seed(seed) 
    env.seed(seed)

    # Intial agent
    logger.log_str("Initializing Agent")
    agent = SAC(env.observation_space.shape[0], env.action_space, args["agent"])

    # Initial ensemble model
    logger.log_str("Initializing Model")
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)
    model_args = args["model"]
    if model_args['model_type'] == 'pytorch':
        env_model = EnsembleDynamicsModel(model_args['num_networks'], model_args['num_elites'], state_size, action_size, model_args['reward_size'], model_args['pred_hidden_size'], use_decay=model_args['use_decay'])
    else:
        env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args['pred_hidden_size'], num_networks=args['num_networks'], num_elites=args['num_elites'])

    # Predict environments
    predict_env = PredictEnv(env_model, args['env_name'], model_args['model_type'])

    # Initial pool for env
    logger.log_str("Initializing Buffer")
    env_pool = ReplayMemory(args['replay_size'])
    # Initial pool for model
    rollouts_per_epoch = args['rollout_batch_size'] * args['epoch_length'] / args['model_train_freq']
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args['model_retain_epochs'] * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args['max_path_length'])
    eval_env_sampler = EnvSampler(eval_env, max_path_length=args['max_path_length'])

    logger.log_str("Start Training")
    train(args, env_sampler, eval_env_sampler, predict_env, agent, env_pool, model_pool, logger)


if __name__ == '__main__':
    main()
