#%%
# loading dataset
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation
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
	color_mode = "grayscale",
	subset = 'both',
	seed = 172,
	image_size = image_size,
	batch_size = batch_size
)

print("Sampling output...")
#%matplotlib qt # type: ignore
# use %matplotlib inline for interactive window input
# be sure to have pylance and pip install PyQt5

plt.figure(figsize=(10, 10))
for images, labels in test_data.take(1):
	for i in range(9):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(np.array(images[i]).astype("uint8"))
		plt.title(int(labels[i]))
		plt.axis("off")

n_classes = len(train_data.class_names)
print("Number of classes: ", n_classes)

# %% creating model

'''
conv > max > conv > max >conv > max > full conn > softmax
'''

print("Making sequential model structure...")
model = Sequential([
	Rescaling(scale = 1./255),	# changes rgb channels from 0 to 255 to 0 to 1.
	# be sure to input 1 channel into input shape
	Conv2D(filters = 8, kernel_size = (20, 20), activation = 'relu'),
	MaxPool2D(pool_size = (2,2)),
	# we don't specify 8 feature maps bc it happens to all; and its based off the prev layer
	Conv2D(filters = 16, kernel_size = (10,10), activation = 'relu'),
	MaxPool2D(pool_size = (2,2)),
	Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu'),
	MaxPool2D(pool_size = (2,2)),
	Flatten(),
	Dense(units = n_classes),
	Activation(activation = "softmax")
	# i have it set to # of classes rn # print(len(train_data.class_names))
])
print("Done making model structure.")
