#%%
# start
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as img
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPool2D, Flatten, Dropout, Dense, Activation
# dense = fully connected layers
from tensorflow.keras.optimizers import Adam, SGD
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

tf.config.list_physical_devices("GPU")
# %%


print("Running initial setup... setting up dataset...")

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
# batch_size = 64				# reduce batch size if laggy

print("Running image_dataset_from_directory()...")

# categories is a list of "folders"
#	consider a  "folder" for each folder in the dir_dataset
#	add it to the list if it exists (via checking to see if the main directory + folder in question is valid)
categories = [folder for folder in os.listdir(dir_dataset) if os.path.join(dir_dataset, folder)]

images = []
labels = []

# i need to study for loops in python this is cbrazy.
#	index and category are tuple returned from enumrate func
for index, category in enumerate(categories):
	cat_path = os.path.join(dir_dataset, category)		# full path
	image_files = [file for file in os.listdir(cat_path) if file.endswith(('.jpg', '.png', '.jpeg'))]

	for image in image_files:
		img_path = os.path.join(cat_path, image)
		img_data = img.load_img(img_path, target_size=image_size, color_mode='grayscale')	# add color_mode = grayscale if needed
		img_array = img.img_to_array(img_data)				# convert to array
		
		images.append(img_array)							# add it to large array
		labels.append(index)								# might want to change to index idk
															# 	yeah use index bc data needs to be ints
	
# convert to np arrays
images = np.array(images)
labels = np.array(labels)

print('Size of images array:', images.shape)
print('Size of labels array:', labels.shape)

# old code that didn't work
'''train_data, test_data = tf.keras.utils.image_dataset_from_directory(
	dir_dataset,
	labels = 'inferred',
	validation_split = 0.2,
	#color_mode = "grayscale",
	subset = 'both',
	seed = 172,
	image_size = image_size,
	batch_size = batch_size
)'''


#%%
# sampling
print("Sampling output...")

#%matplotlib qt # type: ignore
# use %matplotlib inline for interactive window input
# be sure to have pylance and pip install PyQt5
#	update: no longer working for me either

plt.figure(figsize=(7, 10))
rand_samples = np.random.choice(len(images), size=9, replace=False)		# get random sample indices, no dups

for i, index in enumerate(rand_samples):
	axes = plt.subplot(3, 3, i + 1)	# subplot for each img
	plt.imshow(images[index].astype("uint8"))
	plt.title(labels[index])
	plt.axis("off")

n_classes = len(np.unique(labels))		# labels contains each label for each category too
print("Number of classes: ", n_classes)

# %%
# creating model

# conv > max > conv > max >conv > max > full conn (dense) + softmax

print("Making sequential model structure...")
'''model = Sequential([
	Rescaling(1./255, input_shape = (200, 200, 1)),	# changes rgb channels from 0 to 255 to 0 to 1.
		# be sure to input 1 channel into input shape
	Conv2D(filters = 8, kernel_size = (20, 20), activation = 'relu'),
	MaxPool2D(pool_size = (2,2)),
		# we don't specify 8 feature maps bc it happens to all; and its based off the prev layer
	Conv2D(filters = 16, kernel_size = (10,10), activation = 'relu'),
	MaxPool2D(pool_size = (2,2)),
	Conv2D(filters = 32, kernel_size = (5,5), activation = 'relu'),
	MaxPool2D(pool_size = (2,2)),
	Flatten(),
	Dense(units = n_classes, activation = "softmax"),
	#Activation(activation = "softmax")
		# i have it set to # of classes rn # print(len(train_data.class_names))
])'''

model = Sequential([
	Rescaling(1./255, input_shape=(200, 200, 1)),		# still gives warning on input shape but whatev
	Conv2D(filters = 8, kernel_size = (20,20), activation='relu'),
	MaxPool2D(pool_size=(2, 2)),
	Conv2D(filters = 16, kernel_size = (10,10), activation='relu'),
	MaxPool2D(pool_size=(2, 2)),
	Conv2D(filters = 32, kernel_size = (5,5), activation='relu'),
	MaxPool2D(pool_size=(2, 2)),
	Flatten(),
	Dense(units = n_classes, activation='softmax'),
])

print("Done making model structure.")

# %%
# model compile section

print("Compiling model...")
# compile the model with the optimizer. no idea what any of this shit is rn
# optimizer = Adam(learning_rate=0.001)
# optimizer = SGD(learning_rate=0.001, momentum=0.9)
optimizer = 'adam'

model.compile(optimizer=optimizer,
	loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
	metrics=['accuracy'])

print("Finished compiling.")

model.summary()

# %%
# splitting data
test_to_train = 0.2  # 20 testing  80 training

# stratify maintains category proportion across train and test
x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=test_to_train, random_state=172, stratify=labels)

print('Size of x_train:', x_train.shape)
print('Size of y_train:', y_train.shape)
print('Size of x_test:', x_test.shape)
print('Size of y_test:', y_test.shape)

# %%
# fitting section
from tensorflow.keras.callbacks import EarlyStopping

epochs = 20
batch_size = 128

print("Epochs: ", epochs)
print("Batch size: ", batch_size)

# stop if val_loss plateaus
#earlyStopping = EarlyStopping(monitor="val_loss", patience = 3)

history = model.fit(
	x_train,
	y_train,
	epochs = epochs,
	batch_size = batch_size,
	validation_data=(x_test, y_test),
	#callbacks = [earlyStopping]
)

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

#%%
# evaluation
val_loss, val_acc = model.evaluate(x_test, y_test)

print(f'Validation accuracy: {val_acc:.4f}')

randomList = np.random.choice(len(x_test), 9, replace=False)

#%%
# generate figures
plt.figure(figsize = (5, 5))

for i, index in enumerate(randomList):
	image = x_test[index]

	yhat = model.predict(np.asarray([image]))
	prediction = np.argmax(yhat)
	actual = y_test[index]
	# recall: n rows, n cols, index of current subplot (matlab isn't zero indexed)
	axes = plt.subplot(3, 3, i+1)
	plt.imshow(image.squeeze(), cmap='gray')
	plt.title(f'Pred: {categories[prediction]}? ({categories[actual]})')
	plt.axis("off")
plt.show()

# enumerate over categories and print each class name with its corresponding integer
halfway = len(categories) // 2

for i, class_name in enumerate(categories):
    if i == halfway:
        print()
    # print the class name and corresponding integer
    print(f'({i}: {class_name}),', end=' ')
print()

# %%
# confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Step 2: Calculate predictions on the test set
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 3: Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Step 4: Plot the confusion matrix
plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

# diagonal elements are correct predictions
# off-diagonal elements are misclassifications
# %%
# if you want to save model

# import joblib
# joblib.dump(model, '29x1k_20-epoch_128-batch_adam.joblib')
# or
# model.save('29x1k_20-epoch_128-batch_adam')

# %%
# testing it out with a custom image

if (os.path.exists(os.path.join(os.getcwd(), 'ASL_ECE172_Project', 'test_images'))):
	test_images_path = os.path.join(os.getcwd(), 'ASL_ECE172_Project', 'test_images')

	# Load, preprocess, and make predictions on the test images
	test_images = os.listdir(test_images_path)

	# Prepare to display the images in a grid
	plt.figure(figsize=(5, 5))

	# Iterate over test images
	for i, image_file in enumerate(test_images):
		# Load and preprocess the image
		img_path = os.path.join(test_images_path, image_file)
		img_data = img.load_img(img_path, target_size=image_size, color_mode='grayscale')
		img_array = img.img_to_array(img_data) / 255.0  # Rescale pixel values (0 to 1)
		
		# Add a batch dimension to the image array
		img_array = np.expand_dims(img_array, axis=0)
		
		# Make prediction
		yhat = model.predict(img_array)
		prediction = np.argmax(yhat)
		
		# Plot the image and its prediction
		plt.subplot(2, 2, i + 1)  # Adjust grid size as needed (e.g., 2x2)
		plt.imshow(img_data, cmap='gray')
		plt.title(f'Pred: {categories[prediction]}.')
		plt.axis('off')

	# Show the images with predictions
	plt.show()
# %%
