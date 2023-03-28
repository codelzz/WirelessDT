from network import FeedForwardNN
from torch.distributions import MultivariateNormal
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
import gymnasium as gym
import RL
import time
from matplotlib import pyplot as plt
from sklearn.preprocessing import normalize


class OFFLINE_SUPERVISED:
    def __init__(self, env):
        # Extract environment information
        self.env = env
        self.obs_dim = env.observation_space["TXs"].shape[0]
        self.act_dim = env.action_space.shape[0]

        self.load_weight = True

        # ALG STEP 1
        # Initialize actor and critic networks
        self.actor = FeedForwardNN(self.obs_dim, self.act_dim)
        self.critic = FeedForwardNN(self.obs_dim, 1)
        if self.load_weight:
            self.actor.load_state_dict(torch.load('./ppo_actor.pth'))
            self.actor.eval()
            self.critic.load_state_dict(torch.load('./ppo_critic.pth'))
            self.critic.eval()

        self._init_hyperparameters()

        # Create our variable for the matrix.
        # Note that I chose 0.5 for stdev arbitrarily.
        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)

        # Create the covariance matrix
        self.cov_mat = torch.diag(self.cov_var)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def get_action(self, obs, hidden):
        # Query the actor network for a mean action.
        # Same thing as calling self.actor.forward(obs)
        mean = self.actor(obs, hidden)
        # Create our Multivariate Normal Distribution
        dist = MultivariateNormal(mean, self.cov_mat)
        # Sample an action from the distribution and get its log prob
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # Return the sampled action and the log prob of that action
        # Note that I'm calling detach() since the action and log_prob
        # are tensors with computation graphs, so I want to get rid
        # of the graph and just convert the action to numpy array.
        # log prob as tensor is fine. Our computation graph will
        # start later down the line.
        return action.detach().numpy(), log_prob.detach(), hidden

    def _init_hyperparameters(self):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 480  # timesteps per batch
        self.max_timesteps_per_episode = 160  # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = 0.0001

    def compute_rtgs(self, batch_rews):
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        batch_rtgs = []
        # Iterate through each episode backwards to maintain same order
        # in batch_rtgs
        for ep_rews in reversed(batch_rews):
            discounted_reward = 0  # The discounted reward so far
            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                batch_rtgs.insert(0, discounted_reward)
        # Convert the rewards-to-go into a tensor
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def process_obs(self, obs):
        last_tx = obs["Last_TXs"]
        tx = obs["TXs"]
        return np.append([last_tx], [tx], axis=0)

    def rollout(self):
        # Batch data
        batch_obs = []  # batch observations
        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch
        batch_labels = []

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            # Rewards this episode
            ep_rews = []
            obs = self.env.reset()
            obs = obs[0]
            terminated = False
            truncated = False
            for ep_t in range(self.max_timesteps_per_episode):
                # Increment timesteps ran this batch so far
                t += 1
                # Collect observation
                obs = self.process_obs(obs)
                obs = np.expand_dims(obs, axis=0)
                batch_obs.append(obs)

                # action, log_prob, hidden = self.get_action(obs, hidden)
                # obs, rew, done, _ = self.env.step(action)
                action = np.array([[0.0, 0.0]])
                obs, rew, terminated, truncated, info = self.env.step(action)

                label = info["Movement"]
                batch_labels.append(label)
                # Collect reward, action, and log prob
                ep_rews.append(rew)
                # batch_acts.append(action)
                # batch_log_probs.append(log_prob)
                if terminated or truncated:
                    break
            # Collect episodic length and rewards
            batch_lens.append(ep_t + 1)  # plus 1 because timestep starts at 0
            batch_rews.append(ep_rews)

            # Reshape data as tensors in the shape specified before returning
            batch_obs = np.vstack(batch_obs)
            batch_obs = torch.tensor(batch_obs, dtype=torch.float)
            # batch_acts = np.vstack(batch_acts)
            # batch_acts = torch.tensor(batch_acts, dtype=torch.float)
            # batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
            # ALG STEP #4
            batch_rtgs = self.compute_rtgs(batch_rews)
            # Return the batch data
            return batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_labels

    def learn(self, total_timesteps):
        batch_rtgs_list = []
        train_losses = []
        t_so_far = 0  # Timesteps simulated so far
        epoch = 0
        # h = self.actor.init_hidden(self.max_timesteps_per_episode)
        self.actor.train()
        while t_so_far < total_timesteps:
            print("Learning at epoch ", epoch)
            # Collect batch of training data and label
            batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_labels= self.rollout()
            batch_labels = np.array(batch_labels)
            normed_label = normalize(batch_labels, axis=1, norm='l1')
            normed_label = torch.tensor(normed_label, dtype=torch.float)

            self.actor_optim.zero_grad()
            output = self.actor(batch_obs)
            loss = self.criterion(output, normed_label)
            train_losses.append(loss.item())
            loss.backward()
            # nn.utils.clip_grad_norm_(self.actor.parameters(), 5)
            self.actor_optim.step()
            # Calculate how many timesteps we collected this batch
            t_so_far += np.sum(batch_lens)
            epoch += 1
            print("Loss: {:.6f}...".format(loss.item()))

        torch.save(self.actor.state_dict(), './ppo_actor.pth')
        torch.save(self.critic.state_dict(), './ppo_critic.pth')
        plt.plot(train_losses)
        plt.savefig(f'learning_timestep_{total_timesteps}.png')

    def evaluate(self, total_timesteps):
        with torch.no_grad():
            batch_rtgs_list = []
            train_losses = []
            t_so_far = 0  # Timesteps simulated so far
            epoch = 0
            self.actor.eval()
            while t_so_far < total_timesteps:
                print("Evaluating at epoch ", epoch)
                # Collect batch of training data and label
                batch_obs, batch_acts, batch_log_probs, batch_rtgs, batch_lens, batch_labels = self.rollout()
                batch_labels = np.array(batch_labels)
                normed_label = normalize(batch_labels, axis=1, norm='l1')
                normed_label = torch.tensor(normed_label, dtype=torch.float)

                self.actor.zero_grad()
                output = self.actor(batch_obs)
                loss = self.criterion(output, normed_label)
                train_losses.append(loss.item())
                # Calculate how many timesteps we collected this batch
                t_so_far += np.sum(batch_lens)
                epoch += 1
                print("Loss: {:.6f}...".format(loss.item()))

            torch.save(self.actor.state_dict(), './ppo_actor.pth')
            torch.save(self.critic.state_dict(), './ppo_critic.pth')
            plt.plot(train_losses)
            plt.savefig(f'evaluate_timestep_{total_timesteps}.png')


if __name__ == "__main__":
    env = gym.make('RL/RLTrackOffline-v0')
    wrapped_env = gym.wrappers.RecordEpisodeStatistics(env, 50)  # Records episode-reward
    # obs_space_dims = env.observation_space["TXs"].shape[0]
    # action_space_dims = env.action_space.shape[0]
    # print(obs_space_dims)
    # print(action_space_dims)

    agent = OFFLINE_SUPERVISED(env)
    # agent.learn(1500000)
    agent.evaluate(500)
