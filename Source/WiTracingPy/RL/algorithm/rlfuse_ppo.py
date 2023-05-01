from RL.agent.fusion_model import FusionModel
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
import gymnasium as gym
import RL
import time
from matplotlib import pyplot as plt

class Rlfuse_ppo:
    def __init__(self, env, load_weight=False):
        self.env = env
        self.load_weight = load_weight
        # self.obs_dim = env.observation_space["TXs"].shape[0]
        self.act_dim = env.action_space.n

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FusionModel(imu_feature_size=32, visual_feature_size=32, lstm_hidden_size=64, max_pedestrian_detections=10, num_classes=10)
        self.critic = FusionModel(imu_feature_size=32, visual_feature_size=32, lstm_hidden_size=64, max_pedestrian_detections=10, num_classes=1)
        if self.load_weight:
            self.actor.load_state_dict(torch.load('./fuse_ppo_actor.pth'))
            self.actor.eval()
            self.critic.load_state_dict(torch.load('./fuse_ppo_critic.pth'))
            self.critic.eval()

        self._init_hyperparameters()

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.01)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.save_rate = 500
        self.timestep_length = 50

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 480  # timesteps per batch
        self.max_timesteps_per_episode = 160  # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = 0.005

    def evaluate_action(self, obs):
        with torch.no_grad():
            imu_list, vis_list = self.process_obs(obs)
            obs = (imu_list, vis_list)
            action, _ = self.get_action(obs)
            return action

    def get_action(self, obs):
        mean = self.actor(obs)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return action.detach().numpy(), log_prob.detach()

    def process_obs(self, obs):
        imu_list, vis_list = obs
        imu_list = torch.from_numpy(imu_list).float().unsqueeze(0)
        vis_list = torch.from_numpy(vis_list).float().unsqueeze(0)
        return imu_list, vis_list

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
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def rollout(self):
        # Batch data
        batch_obs_imu = []  # batch observations
        batch_obs_vis = []
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
                imu_list, vis_list = self.process_obs(obs)
                batch_obs_imu.append(imu_list)
                batch_obs_vis.append(vis_list)
                obs = (imu_list, vis_list)
                action, log_prob = self.get_action(obs)
                # obs, rew, done, _ = self.env.step(action)
                obs, rew, terminated, truncated, _ = self.env.step(action)

                matching_success_percentage = rew.count(1) / len(rew)

                # Collect reward, action, and log prob
                ep_rews.append(rew)
                batch_msp.append(matching_success_percentage)
                batch_acts.append(action)
                batch_log_probs.append(log_prob)
                if terminated or truncated:
                    break

            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

            # Reshape data as tensors in the shape specified before returning
            batch_obs_imu = np.vstack(batch_obs_imu)
            batch_obs_imu = torch.tensor(batch_obs_imu, dtype=torch.float)
            batch_obs_vis = np.vstack(batch_obs_vis)
            batch_obs_vis = torch.tensor(batch_obs_vis, dtype=torch.float)
            batch_obs = (batch_obs_imu, batch_obs_vis)

            batch_acts = np.concatenate(batch_acts, axis=1)
            batch_acts = torch.tensor(batch_acts, dtype=torch.float)

            batch_log_probs = np.concatenate(batch_log_probs, axis=1)
            batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            # ALG STEP #4
            batch_rtgs = self.compute_rtgs(batch_rews)
            # Return the batch data
            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_msp

    def evaluate(self, batch_obs, batch_acts):
        # Query critic network for a value V for each obs in batch_obs.
        V = self.critic(batch_obs).squeeze()

        # Calculate the log probabilities of batch actions using most
        # recent actor network.
        # This segment of code is similar to that in get_action()
        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        log_probs = dist.log_prob(batch_acts)
        # Return predicted values V and log probs log_probs
        return V, log_probs

    def learn(self, total_timesteps):
        batch_rtgs_list = []
        timesteps = []
        batch_msp_means = []
        t_so_far = 0  # Timesteps simulated so far
        while t_so_far < total_timesteps:
            print("Learning at timestep ", t_so_far)
            # ALG STEP 3
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_msp = self.rollout()

            # save the batch average matching success rate (percentage)
            batch_msp_mean = sum(batch_msp) / len(batch_msp)
            batch_msp_means.append(batch_msp_mean)

            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            batch_rtgs_copy = torch.clone(batch_rtgs).detach().numpy()
            batch_rtgs = batch_rtgs[0].T

            rtg_sum = np.sum(batch_rtgs_copy)
            print(rtg_sum)
            batch_rtgs_list.append(rtg_sum)
            timesteps.append(t_so_far)
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