#%% 
#Load Libraries
from ultralytics import YOLO
import matplotlib.pyplot as plt
import os
import pandas as pd

print("Libraries Loaded")
#%%
#YoloV8 

model = YOLO('yolov8n-cls.pt')

results = model.train(data=os.path.join(os.getcwd(),'asl_dataset'),epochs=10,batch=50,imgsz=200)
print("Training Complete")

metrics = model.val(data=os.path.join(os.getcwd(),'asl_dataset'),epochs=10,batch=50,imgsz=200)
print("Validation Complete")
#%%
#Plot

results_path= os.path.join(os.getcwd(),'runs','classify','train3','results.csv')

results = pd.read_csv(results_path)

plt.figure()
plt.plot(results['                  epoch'], results['             train/loss'], label='train loss')
plt.plot(results['                  epoch'], results['               val/loss'], label='val loss', c='red')
plt.grid()
plt.title('Loss vs epochs for YoLoV8')
plt.ylabel('loss')
plt.xlabel('epochs')
plt.legend()


plt.figure()
plt.plot(results['                  epoch'], results['  metrics/accuracy_top1'] * 100)
plt.grid()
plt.title('Validation accuracy vs epochs for YoLoV8')
plt.ylabel('accuracy (%)')
plt.xlabel('epochs')

plt.show()


# %%
