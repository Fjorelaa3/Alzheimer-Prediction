import os
import pickle
import numpy as np
import pandas as pd
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, build_data_generator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

#%% Read the input data
inputs_dir_path = os.path.abspath('C:/Users/drici/Downloads/MLProject-2/MLProject-2/inputs')
results_dir_path = os.path.abspath('C:/Users/drici/Downloads/MLProject-2/MLProject-2/results')

images_dir_path = os.path.join(inputs_dir_path, 'images')
print("Directory containing images:", images_dir_path)
labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')

labels = get_labels(labels_csv_path)
images = read_all_images(images_dir_path, True)
print("Number of images loaded:", len(images))
print("Images DataFrame columns:", images.columns)
print("Labels DataFrame columns:", labels.columns)
data = build_input_df(images, labels)

#%% Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict): # means all the images have the same scale
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

# Split the data into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

kernels = ['linear','poly', 'rbf', 'sigmoid']

# Function to train and evaluate SVM with different kernels
def train_and_evaluate_svm(kernels, train_data, test_data):
    results = {}
    for kernel in kernels:
        classifier = SVC(kernel=kernel, random_state=123)
        X_train = np.array([img.flatten() for img in train_data['image']])
        y_train = train_data['label'].astype(int)
        X_test = np.array([img.flatten() for img in test_data['image']])
        y_test = test_data['label'].astype(int)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[kernel] = accuracy
        print(f"Accuracy for {kernel} kernel: {accuracy}")
    return results

# Evaluate and print accuracies for each kernel type
train_and_evaluate_svm(kernels, data_train, data_test)

def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def train_and_evaluate_classifier(classifier, train_data, test_data):
    X_train = np.array([img.flatten() for img in train_data['image']])
    y_train = train_data['label'].astype(int)
    
    X_test = np.array([img.flatten() for img in test_data['image']])
    y_test = test_data['label'].astype(int)
    
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    return accuracy, conf_matrix


# SVM
svm_classifier = SVC(kernel='linear', random_state=123)
svm_accuracy, svm_conf_matrix = train_and_evaluate_classifier(svm_classifier, data_train, data_test)
print("SVM Accuracy:", svm_accuracy)
plot_confusion_matrix(svm_conf_matrix, 'SVM Confusion Matrix')