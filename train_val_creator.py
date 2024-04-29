import os
import random
import shutil


dir_dataset = os.path.join(os.getcwd(), 'working_dataset') 

images = []
labels = []
val_dir = os.path.join(dir_dataset,'val')
train_dir = os.path.join(dir_dataset,'train')

if not os.path.exists(train_dir) and not os.path.exists(val_dir):
	os.makedirs(val_dir) 
	os.makedirs(train_dir)
	
for subfolder in os.listdir(dir_dataset):
	subfolder_path = os.path.join(dir_dataset, subfolder)
	if os.path.isdir(subfolder_path):
		images = os.listdir(subfolder_path)
		image_files = [file for file in os.listdir(subfolder_path) if file.endswith(('.jpg', '.png', '.jpeg'))]
		for i, image in enumerate(image_files):
			if i % 5 == 0:
				dest_dir = os.path.join(val_dir,subfolder)
			else:
				dest_dir = os.path.join(train_dir,subfolder)
			if not os.path.exists(dest_dir):
				os.makedirs(dest_dir)
			shutil.copy2(os.path.join(subfolder_path,image),dest_dir)
	
								