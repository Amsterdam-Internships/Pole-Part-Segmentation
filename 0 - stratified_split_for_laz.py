from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import laspy
#from plyfile import PlyData
import numpy as np
import pandas as pd
import os
import shutil

# Introduce the path to the .ply files
os.chdir(r'/home/sara/Desktop/Internship/labelled_pole_dataset-20220117T100957Z-001')
wd = os.getcwd()
# Extract the name of the .laz files and store in a list
file_list = os.listdir()
laz_files=[]
# Choosing .ply files among windows files and store the names in a list
for i in range(len(file_list)):
    if file_list[i].endswith('.laz'):
        laz_files.append(file_list[i])
# Existence matrix generation : which classes exist in each .ply file (1x12 array)? (binary)
exs = np.zeros((len(laz_files), 12), dtype=int)
parts=[]
for i in range(len(laz_files)):
    cloud = laspy.read(laz_files[i])
    j=0
    #print(cloud['vertex']['label'])
    for m in range(12):
        bn = m in cloud.label
        if bn:
            exs[i,m] = 1
            j+=1
        else:
            exs[i,m] = 0
        parts.append(j)
        # print(exs[i])
# Preparing the data to be splited (turn lists to arrays)
X = np.array(laz_files)
y = np.array(exs)
# print(max(parts))


# Setting the split parameters
msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=36)
# Storing the split result as an array
for train_index, test_index in msss.split(X, y):
    print("TRAIN:", train_index,'\n', 'test1', test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

# Name of the chosen files for each split
# print('X_train:\n',X_train, '\n',type(X_test))
# print('X_test:\n',X_test,'\n',type(X_test))

# Clean and Create a directory to save training files
shutil.rmtree(r'/home/sara/Desktop/Internship/ply_files_new/train_ply')
os.mkdir(r'/home/sara/Desktop/Internship/ply_files_new/train_ply')
shutil.rmtree(r'/home/sara/Desktop/Internship/ply_files_new/test_ply')
os.mkdir(r'/home/sara/Desktop/Internship/ply_files_new/test_ply')
shutil.rmtree(r'/home/sara/Desktop/Internship/ply_files_new/val_ply')
os.mkdir(r'/home/sara/Desktop/Internship/ply_files_new/val_ply')

# Separate the training files and store it in a separate folder
for name in X_train:
    shutil.copy(name, 'train_ply')
# Split the test data into test and validation set
exs2=[]
for name in X_test:
    exs2.append(exs[laz_files.index(name)])
X_test=np.array(X_test)
y2=np.array(exs2)
msss2 = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=45)
for test2_index, validation_index in msss2.split(X_test, y2):
    print("TEST:", test2_index,'\n', "Validation:", validation_index)
    Final_test, validation = X_test[test2_index], X_test[validation_index]
    y_test2, y_validation = y2[test2_index], y2[validation_index]
# Move the files into test and validation folders
for name in Final_test:
    shutil.copy(name, 'test_ply')
for name in validation:
    shutil.copy(name, 'val_ply')
