import csv
import random

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
random.seed(27)
sns.set()

def preprocess_image(image):
    # Removes portions of the image that doesn't have relevant information of the track
    image = image[64:140, 0:320, :]

    return image

def read_image(filepath, preprocess=True):
    image = cv2.imread(filepath)
    
    if preprocess:
        image = preprocess_image(image)
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image
offset_left_camera = 0.2
offset_right_camera = -0.2
lines = []
with open("./driving_log.csv") as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)
        
images = []
steers = []
for line in lines:
    for i in range(3):
        filepath = (line[i]).strip()
        # print(filepath)
        # continue
        image = read_image(filepath)
        
        # center
        if i == 0:
            steer = float(line[3])
            steers.append(steer)
            images.append(image)
        # left
        elif i == 1:
            steer = float(line[3])
            steers.append(steer + offset_left_camera)
            images.append(image)
        # right
        elif i == 2:
            steer = float(line[3])
            steers.append(steer - offset_right_camera)
            images.append(image)
indexes = [100, 200, 350, 400, 500, 600, 725, 815]
fig, axes = plt.subplots(len(indexes), 3, figsize=(20, 20))

#import sys
#sys.exit(0)
print(f'total lines in csv: {len(lines)}')

for i in range(len(indexes)):
    
    idx = indexes[i]
    
    print(f'Reading line no.: {idx}')
    print(f'No. of columns in line {idx}: {len(lines[idx])}')

    
    filepath_l = (lines[idx][1]).strip()
    image_l = read_image(filepath_l)
    
    filepath_c =(lines[idx][0]).strip()
    image_c = read_image(filepath_c)
    
    filepath_r = (lines[idx][2]).strip()
    image_r = read_image(filepath_r)

    axes[i, 0].set_title("Left Camera (Steering: {:.2f})".format(steers[idx] + offset_left_camera))
    axes[i, 0].imshow(image_l)
    axes[i, 1].set_title("Center Camera (Steering: {:.2f})".format(steers[idx]))
    axes[i, 1].imshow(image_c)
    axes[i, 2].set_title("Right Camera (Steering: {:.2f})".format(steers[idx] + offset_right_camera))
    axes[i, 2].imshow(image_r)
plt.show()
unique, counts = np.unique(np.array(steers), return_counts=True)
fig = plt.figure(figsize=(15,10))
plt.title("Data distribution")
plt.hist(steers, bins=len(unique))
plt.show()
steerings = {}
for i in range(len(unique)):
    steerings[unique[i]] = counts[i]
freq = {k: v for k, v in sorted(steerings.items(), key=lambda item: item[1])}
freq = pd.DataFrame(freq.items(), columns=['steering', 'count'])
freq = freq.sort_values(by=['count'], ascending=False)
freq.head(10)
augmented_images, augmented_steers = [], []

for image, steer in zip(images, steers):
    augmented_images.append(image)
    augmented_steers.append(steer)
    augmented_images.append(cv2.flip(image, 1))
    augmented_steers.append(steer*-1.0)
unique, counts = np.unique(np.array(augmented_steers), return_counts=True)
fig = plt.figure(figsize=(15, 10))
plt.title("Data distribution after augmentation")
plt.hist(augmented_steers, bins=len(unique))
plt.show()
augmented_images, augmented_steers = shuffle(augmented_images, augmented_steers, random_state=27)
X_train, X_val, y_train, y_val = train_test_split(augmented_images,
                                                  augmented_steers,
                                                  test_size=0.2,
                                                  random_state=27)
np.save("X_train", X_train)
np.save("y_train", y_train)
np.save("X_val", X_val)
np.save("y_val", y_val)