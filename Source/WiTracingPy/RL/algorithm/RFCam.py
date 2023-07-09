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
import pandas as pd
import torch
from torch.optim import Adam
from torch import nn
import numpy as np
import gymnasium as gym
import RL
import time
from matplotlib import pyplot as plt
import logging
import os
import random

from RL.agent.RFCam_wifi import RFCam_wifi
from RL.agent.RFCam_visual import RFCam_visual
from RL.agent.RFCam_imu import RFCam_imu

import cv2

H = np.array([[ 6.33747653e+03, -2.02050995e+03, -4.96471329e+03],
       [-4.42442719e+01,  3.31291282e+03, -2.91777206e+03],
       [-8.23393812e-07,  1.15105347e+00,  1.00000000e+00]])
CAM_X = -1773.490637
CAM_Y = 280.782365
CAM_COORDS = np.array([CAM_X, CAM_Y])
class RFCam:

    def __init__(self, env, lr=0.00002, load_weight=False, train=True):
        self.env = env
        self.load_weight = load_weight
        self.act_dim = env.action_space.n
        self.device = "cuda:0"

        # self.model = RFCam_wifi(wifi_feature_size=64,  num_classes=10).to(self.device)
        self.model = RFCam_visual(visual_feature_size=64, num_classes=10).to(self.device)
        # self.model = RFCam_imu(imu_feature_size=64, num_classes=10).to(self.device)

        self.model_wifi_distance = RFCam_wifi(wifi_feature_size=64,  num_classes=10).to(self.device)
        self.model_wifi_angle = RFCam_wifi(wifi_feature_size=64, num_classes=10).to(self.device)
        self.model_wifi_action = RFCam_wifi(wifi_feature_size=64, num_classes=10).to(self.device)
        self.model_visual = RFCam_visual(visual_feature_size=64, num_classes=10).to(self.device)
        self.model_imu = RFCam_imu(imu_feature_size=64, num_classes=10).to(self.device)

        if train:
            self.model.train()
        else:
            self.model.eval()

        if self.load_weight:
            self.model.load_state_dict(torch.load('./RFCam_abla_visual_action_1.pth'))

            self.model_wifi_distance.load_state_dict(torch.load('./RFCam_abla_wifi_distance.pth'))
            self.model_wifi_angle.load_state_dict(torch.load('./RFCam_abla_wifi_angle.pth'))
            self.model_wifi_action.load_state_dict(torch.load('./RFCam_abla_wifi_action.pth'))

            self.model_visual.load_state_dict(torch.load('./RFCam_abla_visual_action.pth'))

            self.model_imu.load_state_dict(torch.load('./RFCam_abla_imu_action.pth'))

        self._init_hyperparameters(lr=lr)

        self.distance_loss = nn.L1Loss(reduction='mean')
        self.angle_loss = nn.L1Loss(reduction='mean')
        self.action_loss = nn.L1Loss(reduction='mean')
        # self.action_loss = nn.BCELoss()

        self.model_optim = Adam(self.model.parameters(), lr=self.lr)



    def _init_hyperparameters(self, lr):
        # Default values for hyperparameters, will need to change later.
        self.timesteps_per_batch = 256  # timesteps per batch
        self.max_timesteps_per_episode = 64 # timesteps per episode
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2  # As recommended by the paper
        self.lr = lr

    def process_obs(self, wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list):
        # wifi_name_lists, wifi_rssi_lists, imu_list, vis_list = obs
        imu_list = torch.from_numpy(imu_list).float().unsqueeze(0).to(self.device)
        vis_list = torch.from_numpy(vis_list).float().unsqueeze(0).to(self.device)
        wifi_name_lists = torch.from_numpy(wifi_name_lists).unsqueeze(0).to(self.device)

        gt_distances, gt_angles, gt_actions = label_list
        gt_distances = torch.Tensor(gt_distances).float().unsqueeze(0).to(self.device)
        gt_angles = torch.Tensor(gt_angles).float().unsqueeze(0).to(self.device)
        gt_actions = torch.Tensor(gt_actions).unsqueeze(0).to(self.device)
        labels = gt_distances, gt_angles, gt_actions
        wifi_rssi_lists = torch.from_numpy(wifi_rssi_lists).unsqueeze(0).to(self.device)
        # label_list = torch.from_numpy(label_list).unsqueeze(0).to(self.device)
        return wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels

    def process_obs_evaluate(self, wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list):
        # wifi_name_lists, wifi_rssi_lists, imu_list, vis_list = obs
        imu_list = torch.from_numpy(imu_list).float().unsqueeze(0).to(self.device)
        vis_list = torch.from_numpy(vis_list).float().unsqueeze(0).to(self.device)
        vis_list = torch.permute(vis_list, (2, 1, 0, 3)).squeeze()
        wifi_name_lists = torch.from_numpy(wifi_name_lists).unsqueeze(0).to(self.device)

        gt_distances, gt_angles, gt_actions = label_list
        gt_distances = torch.Tensor(gt_distances).float().unsqueeze(0).to(self.device)
        gt_angles = torch.Tensor(gt_angles).float().unsqueeze(0).to(self.device)
        gt_actions = torch.Tensor(gt_actions).unsqueeze(0).to(self.device)
        labels = gt_distances, gt_angles, gt_actions
        wifi_rssi_lists = torch.from_numpy(wifi_rssi_lists).unsqueeze(0).to(self.device)
        # label_list = torch.from_numpy(label_list).unsqueeze(0).to(self.device)
        return wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels

    def rollout(self):
        # Batch data
        batch_obs_imu = []  # batch observations
        batch_obs_vis = []
        batch_obs_wifi_name = []
        batch_obs_wifi_rssi = []
        batch_label = []

        batch_gt_distance = []
        batch_gt_angle = []
        batch_gt_action = []

        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        batch_msp = []

        # Number of timesteps run so far this batch
        t = 0
        while t < self.timesteps_per_batch:
            curr_time = random.randint(0, 40000)
            # print(curr_time)
            # get a observation containing wifi observation and ground truth distance, angle and action
            wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list, gt_label = self.env.get_obs(curr_time)
            # Collect observation
            wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels = self.process_obs(wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, gt_label)
            gt_distances, gt_angles, gt_actions = labels

            batch_obs_imu.append(imu_list)
            batch_obs_vis.append(vis_list)
            batch_obs_wifi_name.append(wifi_name_lists)
            batch_obs_wifi_rssi.append(wifi_rssi_lists)
            batch_label.append(label_list)

            batch_gt_distance.append(gt_distances)
            batch_gt_angle.append(gt_angles)
            batch_gt_action.append(gt_actions)
            t += 1


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

        batch_gt_distance = torch.cat(batch_gt_distance, dim=0).to(self.device)
        batch_gt_angle = torch.cat(batch_gt_angle, dim=0).to(self.device)
        batch_gt_action = torch.cat(batch_gt_action, dim=0).to(self.device)

        return batch_obs, batch_gt_distance, batch_gt_angle, batch_gt_action

    def rollout_eval(self, curr_time):
        # Batch data
        batch_obs_imu = []  # batch observations
        batch_obs_vis = []
        batch_obs_wifi_name = []
        batch_obs_wifi_rssi = []
        batch_label = []

        batch_gt_distance = []
        batch_gt_angle = []
        batch_gt_action = []

        batch_acts = []  # batch actions
        batch_log_probs = []  # log probs of each action
        batch_rews = []  # batch rewards
        batch_rtgs = []  # batch rewards-to-go
        batch_lens = []  # episodic lengths in batch

        batch_msp = []

        # Number of timesteps run so far this batch
        t = 0
        while t < 1:
            # curr_time = random.randint(0, 40000)
            # print(curr_time)
            # get a observation containing wifi observation and ground truth distance, angle and action
            wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list, gt_label = self.env.get_obs(curr_time)
            # Collect observation
            wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels = self.process_obs(wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, gt_label)
            gt_distances, gt_angles, gt_actions = labels

            batch_obs_imu.append(imu_list)
            batch_obs_vis.append(vis_list)
            batch_obs_wifi_name.append(wifi_name_lists)
            batch_obs_wifi_rssi.append(wifi_rssi_lists)
            batch_label.append(label_list)

            batch_gt_distance.append(gt_distances)
            batch_gt_angle.append(gt_angles)
            batch_gt_action.append(gt_actions)
            t += 1


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

        batch_gt_distance = torch.cat(batch_gt_distance, dim=0).to(self.device)
        batch_gt_angle = torch.cat(batch_gt_angle, dim=0).to(self.device)
        batch_gt_action = torch.cat(batch_gt_action, dim=0).to(self.device)

        return batch_obs, batch_gt_distance, batch_gt_angle, batch_gt_action

    def learn(self, total_timesteps):
        t_so_far = 0  # Timesteps simulated so far
        losses = []
        losses_valid = []
        while t_so_far < total_timesteps:
            print("Training at epoch: " + str(t_so_far))
            # print("Learning at timestep ", t_so_far)
            # ALG STEP 3
            batch_obs, batch_gt_distance, batch_gt_angle, batch_gt_action = self.rollout()

            predicted_distance, predicted_angle, predicted_action = self.model(batch_obs)

            # m = nn.Sigmoid()

            dist_loss = self.distance_loss(predicted_distance, batch_gt_distance)
            angle_loss = self.angle_loss(predicted_angle, batch_gt_angle)
            action_loss = self.action_loss(predicted_action, batch_gt_action)

            loss = action_loss
            # loss = 5 * dist_loss + 5 * angle_loss + action_loss

            print("Loss: " + str(loss.item()))
            losses.append(loss.item())


            # Calculate gradients and perform backward propagation for actor
            # network
            self.model_optim.zero_grad()
            loss.backward(retain_graph=False)
            self.model_optim.step()

            # validate
            if t_so_far % 5 == 0:
                batch_obs, batch_gt_distance, batch_gt_angle, batch_gt_action = self.rollout()
                predicted_distance, predicted_angle, predicted_action = self.model(batch_obs)
                dist_loss = self.distance_loss(predicted_distance, batch_gt_distance)
                angle_loss = self.angle_loss(predicted_angle, batch_gt_angle)
                action_loss = self.action_loss(predicted_action, batch_gt_action)

                loss = action_loss
                # loss = 5 * dist_loss + 5 * angle_loss + action_loss
                print("Validate Loss: " + str(loss.item()))
                losses_valid.append(loss.item())

            t_so_far += 1

        torch.save(self.model.state_dict(), 'RFCam_abla_visual_action_1.pth')
        return losses, losses_valid

    def learn_imu(self, total_timesteps):
        t_so_far = 0  # Timesteps simulated so far
        losses = []
        losses_valid = []
        while t_so_far < total_timesteps:
            print("Training at epoch: " + str(t_so_far))
            # print("Learning at timestep ", t_so_far)
            # ALG STEP 3
            batch_obs, batch_gt_distance, batch_gt_angle, batch_gt_action = self.rollout()

            predicted_action = self.model(batch_obs)

            # m = nn.Sigmoid()
            action_loss = self.action_loss(predicted_action, batch_gt_action)

            loss = action_loss

            print("Loss: " + str(loss.item()))
            losses.append(loss.item())

            # Calculate gradients and perform backward propagation for actor
            # network
            self.model_optim.zero_grad()
            loss.backward(retain_graph=False)
            self.model_optim.step()

            # validate
            if t_so_far % 5 == 0:
                batch_obs, batch_gt_distance, batch_gt_angle, batch_gt_action = self.rollout()
                predicted_action = self.model(batch_obs)
                action_loss = self.action_loss(predicted_action, batch_gt_action)

                loss = action_loss
                print("Validate Loss: " + str(loss.item()))
                losses_valid.append(loss.item())

            t_so_far += 1

        torch.save(self.model.state_dict(), 'RFCam_abla_imu_action.pth')
        return losses, losses_valid
    def evaluate(self, curr_time):
        # get a observation containing wifi observation and ground truth distance, angle and action
        wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list, gt_label = self.env.get_obs(curr_time)
        # Collect observation
        wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels = self.process_obs(wifi_name_lists,
                                                                                        wifi_rssi_lists, vis_list,
                                                                                        imu_list, gt_label)
        gt_distances, gt_angles, gt_actions = labels

        obs = (wifi_name_lists, wifi_rssi_lists, imu_list, vis_list)
        predicted_distance, predicted_angle, predicted_action = self.model(obs)


        return predicted_distance, predicted_angle, predicted_action

    def evaluate_wifi(self, curr_time):
        m = nn.Sigmoid()
        # get a observation containing wifi observation and ground truth distance, angle and action
        # wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list, gt_label = self.env.get_obs(curr_time)
        # # Collect observation
        # wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels = self.process_obs(wifi_name_lists,
        #                                                                                 wifi_rssi_lists, vis_list,
        #                                                                                 imu_list, gt_label)
        # gt_distances, gt_angles, gt_actions = labels
        #
        # obs = (wifi_name_lists, wifi_rssi_lists, imu_list, vis_list)
        # predicted_distance, _, _ = self.model_wifi_distance(obs)
        # _, predicted_angle, _ = self.model_wifi_angle(obs)
        # _, _, predicted_action = self.model_wifi_action(obs)

        batch_obs, batch_gt_distance, batch_gt_angle, batch_gt_action = self.rollout_eval(curr_time)

        predicted_distance, _, _ = self.model_wifi_distance(batch_obs)
        _, predicted_angle, _ = self.model_wifi_angle(batch_obs)

        # dist_loss = self.distance_loss(predicted_distance, batch_gt_distance)
        # angle_loss = self.angle_loss(predicted_angle, batch_gt_angle)
        # print(dist_loss)

        return predicted_distance, predicted_angle, _

    def evaluate_visual(self, curr_time):
        # m = nn.Sigmoid()
        # get a observation containing wifi observation and ground truth distance, angle and action
        wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list, gt_label = self.env.get_obs(curr_time)
        # Collect observation
        wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels = self.process_obs_evaluate(wifi_name_lists,
                                                                                        wifi_rssi_lists, vis_list,
                                                                                        imu_list, gt_label)
        gt_distances, gt_angles, gt_actions = labels

        obs = (wifi_name_lists, wifi_rssi_lists, imu_list, vis_list)
        _, _, predicted_action = self.model_visual(obs)
        predicted_action = predicted_action

        vis_list = vis_list.detach().cpu().numpy()
        calculated_distances = []
        calculated_angles = []
        # batch, length, coord = vis_list.shape
        for batch in vis_list:
            angles = batch[:, 0]
            # Reshape the points into required format for cv2.perspectiveTransform
            reshaped_points = batch.reshape(1, -1, 2)

            # Apply the perspective transformation
            transformed = cv2.perspectiveTransform(reshaped_points, H)

            # Reshape the transformed points back into original format
            reshaped_transformed = transformed.reshape(-1, 2)

            distances = np.sqrt(np.sum((reshaped_transformed - CAM_COORDS) ** 2, axis=1)) / 1000

            calculated_distances.append(distances)
            calculated_angles.append(angles)
        calculated_distances = np.array(calculated_distances)
        calculated_angles = np.array(calculated_angles)


        return calculated_distances, calculated_angles, predicted_action

    def evaluate_imu(self, curr_time):
        # get a observation containing wifi observation and ground truth distance, angle and action
        wifi_name_lists, wifi_rssi_lists, vis_list, imu_list, label_list, gt_label = self.env.get_obs(curr_time)
        # Collect observation
        wifi_name_lists, wifi_rssi_lists, imu_list, vis_list, labels = self.process_obs(wifi_name_lists,
                                                                                        wifi_rssi_lists, vis_list,
                                                                                        imu_list, gt_label)
        gt_distances, gt_angles, gt_actions = labels

        obs = (wifi_name_lists, wifi_rssi_lists, imu_list, vis_list)
        predicted_action = self.model_imu(obs)

        action_loss = self.action_loss(predicted_action, gt_actions)
        print(action_loss)

        return predicted_action

def match(curr_time, timelength):
    os.chdir('../..')
    # wifi_df = pd.read_csv('../Services/data/jew_3AI_abla/raw.csv')
    # cam_df = pd.read_csv('../Services/data/jew_3AI_abla/cam_raw.csv')
    # imu_df = pd.read_csv('../Services/data/jew_3AI_abla/imu_raw.csv')

    wifi_df = pd.read_csv('../Services/data/jew_train_new/raw.csv')
    cam_df = pd.read_csv('../Services/data/jew_train_new/cam_raw.csv')
    imu_df = pd.read_csv('../Services/data/jew_train_new/imu_raw.csv')

    env = gym.make('RL/RLfuse-abla')
    env.load_data(wifi_df, cam_df, imu_df, timelength=timelength)
    agent_eval = RFCam(env, load_weight=True, train=False)

    dis_list = []
    ang_list = []

    def count_zeros(list1, list2):
        count = 0
        for x, y in zip(list1, list2):
            if x == 0 or y == 0:
                count += 1
        return count

    for cur in curr_time:
        wifi_distance, wifi_angle, wifi_action = agent_eval.evaluate_wifi(cur)
        visual_distance, visual_angle, visual_action = agent_eval.evaluate_visual(cur)

        wifi_distance = wifi_distance.detach().cpu().numpy()
        wifi_angle = wifi_angle.detach().cpu().numpy()
        d_distance = np.sum(np.abs(wifi_distance - visual_distance), axis=1)
        a_distance = np.sum(np.abs(wifi_angle - visual_angle), axis=1)

        min_dist_index = np.argmin(d_distance)
        min_angle_index = np.argmin(a_distance)

        dis_list.append(min_dist_index)
        ang_list.append(min_angle_index)

        print(min_dist_index, min_angle_index)

    print(count_zeros(dis_list, ang_list)/100)


        # imu_action = agent_eval.evaluate_imu(curr_time)


def match_imu(curr_time, timelength):
    os.chdir('../..')
    # wifi_df = pd.read_csv('../Services/data/jew_3AI_abla/raw.csv')
    # cam_df = pd.read_csv('../Services/data/jew_3AI_abla/cam_raw.csv')
    # imu_df = pd.read_csv('../Services/data/jew_3AI_abla/imu_raw.csv')

    wifi_df = pd.read_csv('../Services/data/jew_train_new/raw.csv')
    cam_df = pd.read_csv('../Services/data/jew_train_new/cam_raw.csv')
    imu_df = pd.read_csv('../Services/data/jew_train_new/imu_raw.csv')

    env = gym.make('RL/RLfuse-abla')
    env.load_data(wifi_df, cam_df, imu_df, timelength=timelength)
    agent_eval = RFCam(env, load_weight=True, train=False)

    # dis_list = []
    # ang_list = []
    action_list = []

    def count_zeros(list1):
        count = 0
        for x in list1:
            if x == 0:
                count += 1
        return count

    for cur in curr_time:
        # wifi_distance, wifi_angle, wifi_action = agent_eval.evaluate_wifi(cur)
        _, _, visual_action = agent_eval.evaluate_visual(cur)
        imu_action = agent_eval.evaluate_imu(cur)

        visual_action = visual_action.detach().cpu().numpy()
        imu_action = imu_action.detach().cpu().numpy()

        act_distance = np.sum(np.abs(visual_action - imu_action), axis=1)

        min_action_index = np.argmin(act_distance)
        action_list.append(min_action_index)

        # wifi_distance = wifi_distance.detach().cpu().numpy()
        # wifi_angle = wifi_angle.detach().cpu().numpy()
        # d_distance = np.sum(np.abs(wifi_distance - visual_distance), axis=1)
        # a_distance = np.sum(np.abs(wifi_angle - visual_angle), axis=1)

        # min_dist_index = np.argmin(d_distance)
        # min_angle_index = np.argmin(a_distance)

        # dis_list.append(min_dist_index)
        # ang_list.append(min_angle_index)

        print(min_action_index)

    print(count_zeros(action_list) / 100)

def train(train_imu=False):
    os.chdir('../..')
    wifi_df = pd.read_csv('../Services/data/jew_train_new/raw.csv')
    cam_df = pd.read_csv('../Services/data/jew_train_new/cam_raw.csv')
    imu_df = pd.read_csv('../Services/data/jew_train_new/imu_raw.csv')

    env = gym.make('RL/RLfuse-abla')
    env.load_data(wifi_df, cam_df, imu_df, timelength=150)
    agent_eval = RFCam(env, load_weight=True)

    if not train_imu:
        losses, losses_valid = agent_eval.learn(400)
    else:
        losses, losses_valid = agent_eval.learn_imu(200)

    # Plot training losses
    plt.plot(losses, label='Training loss')

    # Plot validation losses
    plt.plot(losses_valid, label='Validation loss')

    # Add legend to clarify which line is for training loss and which is for validation loss
    plt.legend()

    # Add title and labels for clarity
    plt.title('Training and Validation Losses Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    # Display the plot
    plt.show()


if __name__ == '__main__':
    train(train_imu=False)

    # random_time_indexes = [random.randint(0, 10000) for _ in range(100)]
    # match(random_time_indexes, timelength=10)

    # random_time_indexes = [random.randint(0, 10000) for _ in range(10)]
    # match_imu(random_time_indexes, timelength=150)


