import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt
import pickle

#Define the paths, categories and size of the image files

DIRECTORY = r'C:\Users\mlobo\Desktop\ML photos\train'
CATEGORIES = ['cat', 'dog'] #Where 0 = cat and 1 = dog
IMG_SIZE = 100

#Convert the images into numeric data
data = []
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category) #picks the path for the two folders (cat and dog)
    label = CATEGORIES.index(category) #defines the labels acording to the folders
    for img in os.listdir(path): #each image 
        img_path = os.path.join(path, img) #path of the image
        arr = cv2.imread(img_path) #reads the images as an array 
        new_arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE)) #changes the size of every image to be the same
        # plt.imshow(new_arr)
        # break
        data.append([new_arr, label]) #adds the data to an object 
len(data)

#Randomize the data and divide it into features and labels 
random.shuffle(data) #randomizes the data 

x = []
y = []
for feaures, labels in data: #separates the features and labels
    x.append(feaures)
    y.append(labels)
    
x = np.array(x) #to conver them into arrays 
y = np.array(y)

#Save the data as files
pickle.dump(x, open('x.pkl','wb')) #wb: write in binary
pickle.dump(y, open('y.pkl','wb')) 

#Load the data from the files 
X = pickle.load(open('x.pkl', 'rb')) #rb: read in binary 
Y = pickle.load(open('y.pkl', 'rb')) 