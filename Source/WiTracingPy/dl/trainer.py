import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import time
import pandas as pd
import numpy as np
import tensorflow as tf

import settings
from udpthread.runnable import Runnable
from preprocessing.dataset import DatasetGenerator
from nn.deepAR import create_model