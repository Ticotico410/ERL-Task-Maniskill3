import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import tqdm

import gymnasium as gym
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
from mani_skill.utils.wrappers.record import RecordEpisode
from mani_skill.utils import gym_utils

LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, obs, act):
        return self.net(torch.cat([obs, act], dim=-1))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_low, act_high, device):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
        )
        self.mean_linear = nn.Linear(256, act_dim)
        self.logstd_linear = nn.Linear(256, act_dim)
        self.action_scale = torch.tensor((act_high - act_low) / 2.0, dtype=torch.float32, device=device)
        self.action_bias = torch.tensor((act_high + act_low) / 2.0, dtype=torch.float32, device=device)

    def forward(self, obs):
        h = self.net(obs)
        mean = self.mean_linear(h)
        log_std = self.logstd_linear(h)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def get_action(self, obs):
        mean, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mean, std)
        x_t = dist.rsample()  # reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = dist.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def get_deterministic_action(self, obs):
        mean, _ = self(obs)
        y_t = torch.tanh(mean)
        action = y_t * self.action_scale + self.action_bias
        return action


class ReplayBuffer:
    def __init__(self, obs_shape, act_shape, buffer_size, num_envs, device):
        self.buffer_size = buffer_size // num_envs
        self.num_envs = num_envs
        self.device = device
        self.pos = 0
        self.full = False

        self.obs = torch.zeros((self.buffer_size, num_envs) + obs_shape, device=device)
        self.next_obs = torch.zeros((self.buffer_size, num_envs) + obs_shape, device=device)
        self.actions = torch.zeros((self.buffer_size, num_envs) + act_shape, device=device)
        self.rewards = torch.zeros((self.buffer_size, num_envs), device=device)
        self.dones = torch.zeros((self.buffer_size, num_envs), device=device)

    def add(self, obs, next_obs, action, reward, done):
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.dones[i] = done

        self.pos += 1
        if self.pos >= self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size):
        high = self.buffer_size if self.full else self.pos
        idxs = torch.randint(0, high, size=(batch_size,))
        env_idxs = torch.randint(0, self.num_envs, size=(batch_size,))
        batch_obs = self.obs[idxs, env_idxs]
        batch_next_obs = self.next_obs[idxs, env_idxs]
        batch_actions = self.actions[idxs, env_idxs]
        batch_rewards = self.rewards[idxs, env_idxs].unsqueeze(-1)
        batch_dones = self.dones[idxs, env_idxs].unsqueeze(-1)
        return batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones


def run_sac(args):
    """
    Execute training or evaluation based on the settings in args:
      - If args.evaluate=True, only the evaluation process is executed.
      - Otherwise, training is executed, while recording loss and reward information in TensorBoard,
        and recording cumulative reward at the end of each episode.
    """
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    # --- Initialize environments ---
    env_kwargs = dict(obs_mode="state", render_mode="rgb_array", sim_backend="gpu")
    if args.control_mode:
        env_kwargs["control_mode"] = args.control_mode

    train_env = gym.make(args.env_id, num_envs=args.num_envs if not args.evaluate else 1, **env_kwargs)
    eval_env = gym.make(args.env_id, num_envs=args.eval_num_envs, **env_kwargs)

    if args.capture_video:
        train_env = RecordEpisode(
            train_env,
            output_dir=os.path.join(args.log_dir, "train_videos"),
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=500,
            video_fps=30,
        )
        eval_env = RecordEpisode(
            eval_env,
            output_dir=os.path.join(args.log_dir, "eval_videos"),
            save_trajectory=False,
            save_video=True,
            max_steps_per_video=args.eval_steps,
            video_fps=30,
        )

    if isinstance(train_env.action_space, gym.spaces.Dict):
        train_env = FlattenActionSpaceWrapper(train_env)
    if isinstance(eval_env.action_space, gym.spaces.Dict):
        eval_env = FlattenActionSpaceWrapper(eval_env)

    train_env = ManiSkillVectorEnv(train_env, args.num_envs, record_metrics=False)
    eval_env = ManiSkillVectorEnv(eval_env, args.eval_num_envs, record_metrics=False)

    obs_shape = train_env.single_observation_space.shape
    act_shape = train_env.single_action_space.shape
    act_low = train_env.single_action_space.low
    act_high = train_env.single_action_space.high

    obs_dim = int(np.prod(obs_shape))
    act_dim = int(np.prod(act_shape))

    # --- TensorBoard Initialization ---
    # run_name = f"{args.env_id}_seed{args.seed}_{int(time.time())}"
    run_name = f"{args.env_id}_seed{args.seed}_{time.strftime('%Y%m%d-%H%M%S')}"

    writer = None
    if not args.evaluate:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(os.path.join(args.log_dir, run_name))
        print("[Train Mode] Logging to:", os.path.join(args.log_dir, run_name))
    else:
        print("[Eval Mode] No training. Just inference.")

    # --- Build networks ---
    actor = Actor(obs_dim, act_dim, act_low, act_high, device=device).to(device)
    critic1 = Critic(obs_dim, act_dim).to(device)
    critic2 = Critic(obs_dim, act_dim).to(device)
    target_critic1 = Critic(obs_dim, act_dim).to(device)
    target_critic2 = Critic(obs_dim, act_dim).to(device)

    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())

    actor_optim = optim.Adam(actor.parameters(), lr=args.learning_rate)
    critic_optim = optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=args.learning_rate)

    if args.autotune:
        target_entropy = -float(act_dim)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = optim.Adam([log_alpha], lr=args.learning_rate)
        alpha = log_alpha.exp().detach()
    else:
        alpha = torch.tensor(args.alpha, device=device)

    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        actor.load_state_dict(ckpt["actor"])
        critic1.load_state_dict(ckpt["critic1"])
        critic2.load_state_dict(ckpt["critic2"])
        target_critic1.load_state_dict(ckpt["critic1"])
        target_critic2.load_state_dict(ckpt["critic2"])
        if args.autotune and "log_alpha" in ckpt:
            log_alpha.data.copy_(ckpt["log_alpha"])

    rb = ReplayBuffer(obs_shape, act_shape, args.buffer_size, args.num_envs, device)
    obs, _ = train_env.reset(seed=args.seed)

    # --- Initialize an array to record cumulative reward for each environment ---
    ep_rewards = np.zeros(args.num_envs, dtype=np.float32)

    # --- If evaluating ---
    if args.evaluate:
        print("Starting evaluation ...")
        eval_obs, _ = eval_env.reset(seed=args.seed)
        for _ in range(args.eval_steps):
            with torch.no_grad():
                act = actor.get_deterministic_action(eval_obs.to(device))
            eval_obs, rew, done, trunc, info = eval_env.step(act)
        print("Evaluation done.")
        eval_env.close()
        train_env.close()
        return

    # --- Training loop ---
    global_step = 0
    max_steps = args.total_timesteps
    pbar = tqdm.tqdm(total=max_steps, desc="Training SAC")

    while global_step < max_steps:
        steps_this_iter = 0
        while steps_this_iter < args.train_freq:
            if global_step < args.start_steps:
                action = torch.tensor(train_env.action_space.sample(), dtype=torch.float32, device=device)
            else:
                with torch.no_grad():
                    a, _ = actor.get_action(obs.to(device))
                action = a

            next_obs, rew, done, trunc, info = train_env.step(action)
            rb.add(obs, next_obs, action, rew, done)
            obs = next_obs

            # --- Accumulate rewards ---
            current_rew = rew.detach().cpu().numpy()  # shape: (num_envs,)
            ep_rewards += current_rew

            # --- Check each environment for episode completion ---
            # Note: use done, not reward
            done_np = done.detach().cpu().numpy() if torch.is_tensor(done) else np.array(done)
            for i in range(args.num_envs):
                if done_np[i]:
                    if writer:
                        writer.add_scalar("train/episode_return", ep_rewards[i], global_step)
                    print(f"Episode finished in env {i}: reward = {ep_rewards[i]:.2f} (global_step={global_step})")
                    ep_rewards[i] = 0.0

            steps_this_iter += 1
            global_step += 1
            pbar.update(1)
            if global_step >= max_steps:
                break

        # --- Record the average ongoing episode reward for the current iteration ---
        avg_reward = float(np.mean(ep_rewards))
        if writer:
            writer.add_scalar("train/avg_episode_reward", avg_reward, global_step)
        print(f"[Step {global_step}] Current average episode reward: {avg_reward:.2f}")

        # --- Update the model ---
        for _update in range(args.updates_per_iter):
            b_obs, b_act, b_rew, b_nobs, b_done = rb.sample(args.batch_size)
            with torch.no_grad():
                next_act, next_logp = actor.get_action(b_nobs)
                q1_next = target_critic1(b_nobs, next_act)
                q2_next = target_critic2(b_nobs, next_act)
                q_next = torch.min(q1_next, q2_next) - alpha * next_logp
                q_target = b_rew + args.gamma * (1.0 - b_done) * q_next

            q1_val = critic1(b_obs, b_act)
            q2_val = critic2(b_obs, b_act)
            q1_loss = F.mse_loss(q1_val, q_target)
            q2_loss = F.mse_loss(q2_val, q_target)
            c_loss = q1_loss + q2_loss

            critic_optim.zero_grad()
            c_loss.backward()
            critic_optim.step()

            new_act, logp_pi = actor.get_action(b_obs)
            q1_pi = critic1(b_obs, new_act)
            q2_pi = critic2(b_obs, new_act)
            q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha * logp_pi - q_pi).mean()

            actor_optim.zero_grad()
            actor_loss.backward()
            actor_optim.step()

            if args.autotune:
                alpha_loss = (-(log_alpha.exp() * (logp_pi + target_entropy).detach())).mean()
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()
                alpha = log_alpha.exp().detach()
            else:
                alpha_loss = torch.tensor(0.).to(device)

            with torch.no_grad():
                for param, tparam in zip(critic1.parameters(), target_critic1.parameters()):
                    tparam.data.lerp_(param.data, args.tau)
                for param, tparam in zip(critic2.parameters(), target_critic2.parameters()):
                    tparam.data.lerp_(param.data, args.tau)

        if writer:
            writer.add_scalar("loss/critic", c_loss.item(), global_step)
            writer.add_scalar("loss/actor", actor_loss.item(), global_step)
            writer.add_scalar("alpha/value", alpha.item(), global_step)

        if (global_step % args.eval_interval == 0) and (global_step < max_steps):
            actor.eval()
            eobs, _ = eval_env.reset()
            e_ret = 0.0
            for _evalstep in range(args.eval_steps):
                with torch.no_grad():
                    dact = actor.get_deterministic_action(eobs.to(device))
                eobs, er, edone, etrunc, einfo = eval_env.step(dact)
                e_ret += er.sum().item()
            if writer:
                writer.add_scalar("eval/return", e_ret/args.eval_num_envs, global_step)
            actor.train()

    print("Training done. Saving final checkpoint ...")
    final_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(final_dir, exist_ok=True)
    ckpt_path = os.path.join(final_dir, "final_model.pt")
    ckpt_data = {
        "actor": actor.state_dict(),
        "critic1": critic1.state_dict(),
        "critic2": critic2.state_dict()
    }
    if args.autotune:
        ckpt_data["log_alpha"] = log_alpha
    torch.save(ckpt_data, ckpt_path)
    print(f"Model saved to: {ckpt_path}")

    if writer:
        writer.close()
    train_env.close()
    eval_env.close()
