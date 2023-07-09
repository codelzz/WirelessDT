from torch.optim.lr_scheduler import MultiplicativeLR

from RL.agent.fusion_model import FusionModel
from RL.agent.fusion_model_v2 import FusionModelv2
from RL.agent.fusion_model_v3 import FusionModelv3
from RL.agent.fusion_model_v4 import FusionModelv4
from RL.agent.fusion_model_v5 import FusionModelv5
from RL.agent.fusion_model_v6 import FusionModelv6
from RL.agent.fusion_model_v7 import FusionModelv7
from RL.agent.fusion_model_v8 import FusionModelv8
from RL.agent.fusion_model_v9 import FusionModelv9
from RL.agent.fusion_model_v9_no_wifi import FusionModelv9_No_WIFI
from RL.agent.fusion_model_v9_no_imu import FusionModelv9_No_IMU
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
import gymnasium as gym
import RL
import time
from matplotlib import pyplot as plt
import logging

class Rlfuse_ppo:
    def __init__(self, env, logger, fill_value=0.01, lr=0.005, decay_rate=0.000009, load_weight=False, train=True):
        self.env = env
        self.load_weight = load_weight
        # self.obs_dim = env.observation_space["TXs"].shape[0]
        self.act_dim = env.action_space.n
        self.device = "cuda:0"

        # ALG STEP 1
        # Initialize actor and critic networks

        self.actor = FusionModelv3(wifi_feature_size=128, imu_feature_size=64, visual_feature_size=64, num_classes=10).to(self.device)
        self.critic = FusionModelv3(wifi_feature_size=128, imu_feature_size=64, visual_feature_size=64, num_classes=10).to(self.device)

        # self.actor = FusionModelv5(wifi_feature_size=32, imu_feature_size=32, visual_feature_size=64, lstm_hidden_size=64, max_pedestrian_detections=10, num_classes=10).to(self.device)
        # self.critic = FusionModelv5(wifi_feature_size=32, imu_feature_size=32, visual_feature_size=64, lstm_hidden_size=64, max_pedestrian_detections=10, num_classes=10).to(self.device)
        if self.load_weight:
            self.actor.load_state_dict(torch.load('./fuse_ppo_actor.pth'))
            self.critic.load_state_dict(torch.load('./fuse_ppo_critic.pth'))

        if train:
            self.actor.train()
            self.critic.train()
        else:
            self.actor.eval()
            self.critic.eval()

        self._init_hyperparameters(lr=lr)

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.init_cov_var = torch.full(size=(self.act_dim,), fill_value=fill_value)

        # Create the covariance matrix
        self.decay_rate = decay_rate
        self.min_cov_var = torch.full(size=(self.act_dim,), fill_value=0.01)
        self.cov_mat = torch.diag(self.init_cov_var).to(self.device)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        lmbda1 = lambda step: 0.5
        self.actor_scheduler = MultiplicativeLR(self.actor_optim, lr_lambda=lmbda1)
        lmbda2 = lambda step: 0.8
        self.critic_scheduler = MultiplicativeLR(self.critic_optim, lr_lambda=lmbda2)

        self.save_rate = 512
        # self.timestep_length = 50

        self.training_step = 0

        self.logger = logger
        logger.info('Setting up logger...')


    def _init_hyperparameters(self, lr):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 512  # timesteps per batch
        self.max_timesteps_per_episode = 256 # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = lr

    def update_cov_mat(self):
        cov_var = self.init_cov_var * np.exp(-self.decay_rate * self.training_step)
        cov_var = np.maximum(cov_var, self.min_cov_var)
        self.cov_mat = torch.diag(cov_var).to(self.device)
        self.logger.debug(f"Current cov_var: {cov_var}")
        # return cov_var

    def evaluate_action(self, obs):
        with torch.no_grad():
            wifi_name_lists, wifi_rssi_lists, imu_list, vis_list = self.process_obs(obs)
            obs = (wifi_name_lists, wifi_rssi_lists, imu_list, vis_list)
            # action, _ = self.get_action(obs)
            mean, _ = self.actor(obs)
            return mean.tolist()

    def get_action(self, obs):
        mean, _ = self.actor(obs)
        # Create our Multivariate Normal Distribution
        # self.update_cov_mat()
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.cpu().detach().numpy(), log_prob.cpu().detach()

    def process_obs(self, obs):
        wifi_name_lists, wifi_rssi_lists, imu_list, vis_list = obs
        imu_list = torch.from_numpy(imu_list).float().unsqueeze(0).to(self.device)
        vis_list = torch.from_numpy(vis_list).float().unsqueeze(0).to(self.device)
        wifi_name_lists = torch.from_numpy(wifi_name_lists).unsqueeze(0).to(self.device)
        wifi_rssi_lists = torch.from_numpy(wifi_rssi_lists).unsqueeze(0).to(self.device)
        return wifi_name_lists, wifi_rssi_lists, imu_list, vis_list

    # def compute_rtgs(self, batch_rews):
    #     # The rewards-to-go (rtg) per episode per batch to return.
    #     # The shape will be (num timesteps per episode)
    #     batch_rtgs = []
    #     # Iterate through each episode backwards to maintain same order
    #     # in batch_rtgs
    #     for ep_rews in reversed(batch_rews):
    #         discounted_reward = 0  # The discounted reward so far
    #         for rew in reversed(ep_rews):
    #             discounted_reward = rew + discounted_reward * self.gamma
    #             batch_rtgs.insert(0, discounted_reward)
    #     # Convert the rewards-to-go into a tensor
    #     batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
    #     return batch_rtgs

    def compute_rtgs(self, batch_rews):
        # batch_rews shape: (1, num_episodes, num_timesteps)
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (1, num_episodes, num_timesteps)
        batch_rtgs = []
        # Iterate through each batch
        for batch_rew in batch_rews:
            batch_rtgs_per_batch = []
            # Iterate through each episode
            for ep_rews in batch_rew:
                ep_rtgs = []
                discounted_reward = 0  # The discounted reward so far
                # Iterate through each timestep in the episode
                for rew in reversed(ep_rews):
                    discounted_reward = rew + discounted_reward * self.gamma
                    ep_rtgs.insert(0, discounted_reward)
                batch_rtgs_per_batch.append(ep_rtgs)
            batch_rtgs.append(batch_rtgs_per_batch)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float).to(self.device)
        return batch_rtgs

    def rollout(self):
        # Batch data
        batch_obs_imu = []  # batch observations
        batch_obs_vis = []
        batch_obs_wifi_name = []
        batch_obs_wifi_rssi = []

        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        batch_msp = []

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs, info = self.env.reset()
            terminated = False
            truncated = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                wifi_name_lists, wifi_rssi_lists, imu_list, vis_list = self.process_obs(obs)
                batch_obs_imu.append(imu_list)
                batch_obs_vis.append(vis_list)
                batch_obs_wifi_name.append(wifi_name_lists)
                batch_obs_wifi_rssi.append(wifi_rssi_lists)

                obs = (wifi_name_lists, wifi_rssi_lists, imu_list, vis_list)
                action, log_prob = self.get_action(obs)
                # obs, rew, done, _ = self.env.step(action)
                obs, rew, terminated, truncated, _ = self.env.step(action)
                prob, reward = rew
                matching_success_percentage = prob

                # Collect reward, action, and log prob
                ep_rews.append(reward)
                batch_msp.append(matching_success_percentage)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if terminated or truncated:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

            # Reshape data as tensors in the shape specified before returning
            # batch_obs_imu = np.vstack(batch_obs_imu)
            batch_obs_imu = torch.cat(batch_obs_imu, dim=0).to(self.device)
            # batch_obs_imu = torch.tensor(batch_obs_imu, dtype=torch.float).to(self.device)
            # batch_obs_vis = np.vstack(batch_obs_vis)
            batch_obs_vis = torch.cat(batch_obs_vis, dim=0).to(self.device)
            # batch_obs_vis = torch.tensor(batch_obs_vis, dtype=torch.float).to(self.device)
            # batch_obs_wifi_name = np.vstack(batch_obs_wifi_name)
            batch_obs_wifi_name = torch.cat(batch_obs_wifi_name, dim=0).to(self.device)
            # batch_obs_wifi_name = torch.tensor(batch_obs_wifi_name, dtype=torch.int32).to(self.device)
            # batch_obs_wifi_rssi = np.vstack(batch_obs_wifi_rssi)
            batch_obs_wifi_rssi = torch.cat(batch_obs_wifi_rssi, dim=0).to(self.device)
            # batch_obs_wifi_rssi = torch.tensor(batch_obs_wifi_rssi, dtype=torch.int32).to(self.device)
            batch_obs = (batch_obs_wifi_name, batch_obs_wifi_rssi, batch_obs_imu, batch_obs_vis)

            batch_acts = np.concatenate(batch_acts, axis=0)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float).to(self.device)

            batch_log_probs = np.concatenate(batch_log_probs, axis=0)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float).to(self.device)
            # ALG STEP #4
            batch_rtgs = self.compute_rtgs(batch_rews)
            # Return the batch data
            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_msp

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        _, V = self.critic(batch_obs)
        V = V.squeeze()
        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean, _ = self.actor(batch_obs)
        # self.update_cov_mat()
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs

    def learn(self, total_timesteps):
        batch_rtgs_list = []
        timesteps = []
        batch_msp_means = []
        t_so_far = 0  # Timesteps simulated so far
        last_lr_decrease = 0
        while t_so_far < total_timesteps:
            # print("Learning at timestep ", t_so_far)
            self.logger.info(f"Learnilng at timestep: {t_so_far}")
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_msp = self.rollout()

            # save the batch average matching success rate (percentage)
            batch_msp_mean = sum(batch_msp) / len(batch_msp)
            batch_msp_means.append(batch_msp_mean)

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            batch_rtgs_copy = torch.clone(batch_rtgs).cpu().detach().numpy()
            batch_rtgs = batch_rtgs[0]

            rtg_sum = np.sum(batch_rtgs_copy)
            # print(f"rtg_sum: {rtg_sum}; batch_msp_mean: {batch_msp_mean}")
            self.logger.info(f"rtg_sum: {rtg_sum}; batch_msp_mean: {batch_msp_mean}")
            self.logger.info(f"Current actor learning rate: {self.actor_optim.param_groups[0]['lr']}, critic learning rate: {self.critic_optim.param_groups[0]['lr']}")
            batch_rtgs_list.append(rtg_sum)
            timesteps.append(t_so_far)
            self.training_step = t_so_far
            # Calculate V_phi and pi_theta(a_t | s_t)
            V, curr_log_probs = self.evaluate(batch_obs, batch_acts)
            # ALG STEP 5
            # Calculate advantage
            A_k = batch_rtgs - V.detach()

            # Normalize advantages
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.n_updates_per_iteration):
                # epoch code
                # Calculate pi_theta(a_t | s_t)
                V, curr_log_probs = self.evaluate(batch_obs, batch_acts)

                # Calculate ratios
                ratios = torch.exp(curr_log_probs - batch_log_probs)

                # Calculate surrogate losses
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                # Calculate gradients and perform backward propagation for actor
                # network
                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                # Calculate gradients and perform backward propagation for critic network
                self.critic_optim.zero_grad()
                critic_loss.backward()
                self.critic_optim.step()

            if t_so_far - last_lr_decrease >= 50000:
                self.actor_scheduler.step()
                self.critic_scheduler.step()
                last_lr_decrease = t_so_far

            # Save our model if it's time
            if t_so_far % self.save_rate == 0:
                torch.save(self.actor.state_dict(), './checkpoints_rl/fuse_ppo_actor_'+str(t_so_far)+'.pth')
                torch.save(self.critic.state_dict(), './checkpoints_rl/fuse_ppo_critic_'+str(t_so_far)+'.pth')
        torch.save(self.actor.state_dict(), './checkpoints_rl/fuse_ppo_actor_' + str(t_so_far) + '.pth')
        torch.save(self.critic.state_dict(), './checkpoints_rl/fuse_ppo_critic_' + str(t_so_far) + '.pth')
        # plt.plot(timesteps, batch_rtgs_list)
        # plt.xlabel('timestep')
        # plt.ylabel('rtgs')
        # plt.savefig(f'learning_timestep_{total_timesteps}.png')
        #
        # plt.plot(timesteps, batch_msp_means)
        # plt.xlabel('timestep')
        # plt.ylabel('average matching successful rate')
        # plt.savefig(f'msp_{total_timesteps}.png')

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()

        # Plot y1 on the first figure
        ax1.plot(timesteps, batch_rtgs_list)
        # Plot y2 on the second figure
        ax2.plot(timesteps, batch_msp_means)

        # Customize the first plot
        ax1.set_title('batch reward to go')
        ax1.set_xlabel('timestep')
        ax1.set_ylabel('reward')

        # Customize the second plot
        ax2.set_title('batch matching success rate')
        ax2.set_xlabel('timestep')
        ax2.set_ylabel('success rate')

        plt.savefig(f'learning_timestep_{total_timesteps}.png')