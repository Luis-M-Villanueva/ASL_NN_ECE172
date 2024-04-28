#%% 
#Load Libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation
# dense = fully connected layers
from tensorflow.keras.utils import load_img,img_to_array
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import ResNet50 
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

print("Libraries Loaded")
#%%Running initial setup... setting up dataset...
print(f"CWD: ", os.getcwd())		# Amaan's
									# C:\Users\amaan_r7vd8kf\AppData\Local\Programs\Microsoft VS Code

dir_dataset = os.path.join(os.getcwd(), 'working_dataset') # DIRECTOY IS SUBJECT TO CHANGE FOR DIFFERENT USERS
image_size = (200,200)


categories = [folder for folder in os.listdir(dir_dataset) if os.path.join(dir_dataset, folder)]

images = []
labels = []

for index, category in enumerate(categories):
	cat_path = os.path.join(dir_dataset, category)		# full path
	image_files = [file for file in os.listdir(cat_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

	for image in image_files:
		img_path = os.path.join(cat_path, image)
		img_data = img.load_img(img_path, target_size=image_size, color_mode='grayscale')	# add color_mode = grayscale if needed
		img_array = img.img_to_array(img_data)				
		
		images.append(img_array)							
		labels.append(index)								
															
	
# convert to np arrays
images = np.array(images)
labels = np.array(labels)

print('Size of images array:', images.shape)
print('Size of labels array:', labels.shape)

n_classes = len(np.unique(labels))		# labels contains each label for each category too
print("Number of classes: ", n_classes)


#%%
#ResNet50

rnn = Sequential()

model = ResNet50(
        include_top = False,
        weights= None,
		input_tensor = None,
        input_shape = (200,200,1),
		pooling = 'avg',
		classes=5,
		classifier_activation="softmax"
)

rnn.add(model)
rnn.add(Flatten())
rnn.add(Dense(512,activation='relu'))
rnn.add(Dense(5,activation='softmax'))
rnn.compile(optimizer=Adam(learning_rate=.001),
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
	metrics=['accuracy'])



print("Resnet Model Created")

#%%
#Split Data & fit
test_to_train = 0.2 
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_to_train, random_state=172, stratify=labels)

history = rnn.fit(
	x_train,
	y_train,
	epochs = 10,
	validation_data=(x_test, y_test),
	#callbacks = [earlyStopping]
)

rnn.summary()
# %%
# plotting section
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(5, 5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Over Epochs')
plt.legend()
plt.show()
