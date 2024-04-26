#%%
# from the top! let's load datasets
#using https://keras.io/examples/vision/image_classification_from_scratch/
import os
import numpy as np
import keras
from keras import layers
from keras import Sequential				# read up on how to use this
from tensorflow import data as tf_data
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
imageSize = (200, 200)	# images are 200x200
batchSize = 200			# not sure how much it should be but bigger batch size = more mem usage, but is more
						#	efficient. used for fwd and bkwd pass

print("Running image_dataset_from_directory()...")
train_data, test_data = keras.utils.image_dataset_from_directory(
	dir_dataset,
	labels = 'inferred',
	validation_split = 0.2,
	subset = 'both',
	seed = 172,
	image_size = imageSize,
	batch_size = batchSize
)
#%%
# sampling an output

# ignoring errors on the following line. use it to output a new window for the plot
%matplotlib qt # type: ignore
# use %matplotlib inline for interactive window input

plt.figure(figsize=(10, 10))
for images, labels in test_data.take(1):
	for i in range(9):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(np.array(images[i]).astype("uint8"))
		plt.title(int(labels[i]))
		plt.axis("off")
# %%
# let's make the model or some shit
'''
https://keras.io/examples/vision/image_classification_from_scratch/
	and https://github.com/keras-team/keras-io/blob/master/examples/vision/image_classification_from_scratch.py

checking with liam's code too: https://github.com/liamgoss/neuralNetworks/blob/master/Machine%20Learning/TensorFlow%20Tutorials/Convolutional%20Neural%20Network%20Models/Image%20Classification.py

here's some of what i've been up to 6:42 4/25
https://chat.openai.com/share/840b2ec2-7f90-4c70-ac3c-05b16cdc0691
	scroll to bottom

'''
# model = Sequential()