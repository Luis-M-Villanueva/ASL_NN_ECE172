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

tf.config.list_physical_devices("GPU")
print("Libraries Loaded")
#%%Running initial setup... setting up dataset...
print(f"CWD: ", os.getcwd())		# Amaan's
									# C:\Users\amaan_r7vd8kf\AppData\Local\Programs\Microsoft VS Code

dir_dataset = os.path.join(os.getcwd(), 'ASL_ECE172_Project', 'working_dataset')
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
		#classifier_activation="softmax"
)

# new code, make or break - speed is life?
# for layer in model.layers:
#	layer.trainable=False

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
	batch_size = 32,
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

#%%
# evaluation
val_loss, val_acc = rnn.evaluate(x_test, y_test)

print(f'Validation accuracy: {val_acc:.4f}')

randomList = np.random.choice(len(x_test), 9, replace=False)

#%%
# generate figures
plt.figure(figsize = (5, 5))

for i, index in enumerate(randomList):
	image = x_test[index]

	yhat = rnn.predict(np.asarray([image]))
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
y_pred = rnn.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# Step 3: Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_classes)

# Step 4: Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Reds', fmt='g')
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
plt.title('Confusion Matrix')
plt.show()

# diagonal elements are correct predictions
# off-diagonal elements are misclassifications