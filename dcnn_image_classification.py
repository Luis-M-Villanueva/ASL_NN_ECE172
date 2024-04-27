#%%
# from the top! let's load datasets
#using https://keras.io/examples/vision/image_classification_from_scratch/
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten, Dropout
# dense = fully connected layers
import matplotlib.pyplot as plt

r'''
Folder setup:
C:\Users\amaan_r7vd8kf\AppData\Local\Programs\Microsoft VS Code\ASL_ECE172_Project
	contains:
		Actual_ASL_Dataset 	- this is the raw data
		working_dataset 	- subfolders are here too, but minimized for testing.
'''

print(f"CWD: ", os.getcwd())		# Amaan's
									# C:\Users\amaan_r7vd8kf\AppData\Local\Programs\Microsoft VS Code

dir_dataset = os.path.join(os.getcwd(), 'ASL_ECE172_Project', 'working_dataset')
image_size = (200,200)
batch_size = 64				# reduce batch size if laggy

print("Running image_dataset_from_directory()...")
train_data, test_data = tf.keras.utils.image_dataset_from_directory(
	dir_dataset,
	labels = 'inferred',
	validation_split = 0.2,
	subset = 'both',
	seed = 172,
	image_size = image_size,
	batch_size = batch_size
)