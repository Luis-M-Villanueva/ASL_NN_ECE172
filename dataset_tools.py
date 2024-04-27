import os
import shutil

print(f"CWD: ", os.getcwd())

# Define the source directory containing the subfolders
source_dir = os.path.join(os.getcwd(), 'ASL_ECE172_Project', 'Actual_ASL_Dataset')
print(source_dir)

# Define the destination directory for the working dataset
dest_dir = os.path.join(os.getcwd(), 'ASL_ECE172_Project', 'working_dataset')
print(dest_dir)

# Iterate through each subfolder in the source directory
for subfolder in os.listdir(source_dir):
	subfolder_path = os.path.join(source_dir, subfolder)
	
	# Check if the path is a directory (skip files)
	if os.path.isdir(subfolder_path):
		# List the images in the subfolder
		images = os.listdir(subfolder_path)

		print(f"Processing subfolder: {subfolder}, Total images found: {len(images)}")
		
		# Create the destination subfolder if it doesn't exist
		dest_subfolder_path = os.path.join(dest_dir, subfolder)
		os.makedirs(dest_subfolder_path, exist_ok=True)
		
		# Copy every third image up to a total of 1000 images
		counter = 0
		# Iterate through images, selecting every third one
		for i in range(0, len(images), 3):
			if counter >= 1000:
				break
			
			# Get the source image path
			src_img_path = os.path.join(subfolder_path, images[i])
			
			# Get the destination image path in the respective subfolder
			dest_img_path = os.path.join(dest_subfolder_path, images[i])
			
			# Copy the image
			shutil.copy2(src_img_path, dest_img_path)
			
			# Increment the counter
			counter += 1

print("Done.")
