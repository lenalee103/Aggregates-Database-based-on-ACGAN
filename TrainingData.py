import numpy as np
import os
from matplotlib import pyplot as plt
from PIL import Image
import random
import pickle

path = os.path.join(os.getcwd(),'stones')

training_data = []

def create_training_data():
    for img in os.listdir(path):
        try:
           if 'tif' in img:
               im = Image.open(os.path.join(path,img))
               im1 = im.resize((56,56))
               img_array = np.array(im1)
               class_num = str(img).split('_')[0]
               if class_num == '4.75mm': class_nume = 0
               elif class_num == '9.5mm': class_nume = 1
               elif class_num == '13.2mm' : class_nume = 2
               elif class_num == '16mm' : class_nume = 3
               elif class_num == '4.75mm1000': class_nume = 4
               elif class_num == '9.5mm1000': class_nume = 5
               elif class_num == '13.2mm1000' : class_nume = 6
               elif class_num == '16mm1000' : class_nume = 7
               else: class_nume = 8
               training_data.append([img_array, class_nume])
        except Exception as e:
            pass


create_training_data()
print(training_data[:5])


random.shuffle(training_data)

X = [] #features
y = [] #labels


for features, label in training_data:
    X.append(features)
    y.append(label)


# Creating the files containing all the information about your model
pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)


