import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import Normalize
from matplotlib.cm import get_cmap


def printnorm(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    if isinstance(output, tuple):
        # print('output size:', [o.size() for o in output])
        # print('output norm:', [o.norm() for o in output])
        feature_map = output[0].cpu()
        sum_tensor = torch.sum(feature_map, dim=-1)  # Sum
        mean_tensor = torch.mean(feature_map, dim=-1)  # Average
        # Assuming result_tensor is either sum_tensor or mean_tensor
        result_tensor = sum_tensor.squeeze()  # Remove the batch dimension
        result = result_tensor.numpy().reshape(-1, 1).T
        result = np.repeat(result, repeats=20, axis=0)
        plt.figure(figsize=(10, 5))
        plt.imshow(result, cmap='viridis')  # Replace histogram with colormap
        plt.colorbar()  # Add a colorbar to the plot
        plt.grid(False)
        plt.show()

    else:
        print('output size:', output.size())
        print('output norm:', output.norm())


def printnorm_visual(self, input, output):
    # input is a tuple of packed inputs
    # output is a Tensor. output.data is the Tensor we are interested
    print('Inside ' + self.__class__.__name__ + ' forward')
    print('')
    print('input: ', type(input))
    print('input[0]: ', type(input[0]))
    print('output: ', type(output))
    print('')
    print('input size:', input[0].size())
    if isinstance(output, tuple):
        # print('output size:', [o.size() for o in output])
        # print('output norm:', [o.norm() for o in output])
        feature_map = output[0].cpu()
        for i in range(feature_map.shape[0]):  # Loop through each item in the batch
            sum_tensor = torch.sum(feature_map[i], dim=-1)  # Sum
            mean_tensor = torch.mean(feature_map[i], dim=-1)  # Average
            result_tensor = mean_tensor.squeeze()  # Remove the batch dimension
            result = result_tensor.numpy().reshape(-1, 1).T
            result = np.repeat(result, repeats=20, axis=0)
            plt.figure(figsize=(10, 5))
            plt.imshow(result, cmap='viridis')  # Replace histogram with colormap
            plt.colorbar()  # Add a colorbar to the plot
            plt.grid(False)
            plt.title(f'Tracklet {i}')  # Add a title to each plot
            plt.show()

    else:
        print('output size:', output.size())
        print('output norm:', output.norm())
