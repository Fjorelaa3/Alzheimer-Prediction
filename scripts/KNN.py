import os
import pickle
import numpy as np
import pandas as pd
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, build_data_generator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_score, recall_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as plt

print(np.__version__)

# Read the input data
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

# Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict):
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

# Split the data into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

# Modify the provided KNN code to run for neighbors from 1 to 7 and print the accuracies for each
neighbors_list = [1, 2, 3, 4, 5, 6, 7]

#Helper function to train and evaluate the KNN model
def train_and_evaluate_knn(neighbors_list, train_data, test_data):
    results = {}
    for n_neighbors in neighbors_list:
        classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        X_train = np.array([img.flatten() for img in train_data['image']])
        y_train = train_data['label'].astype(int)
        X_test = np.array([img.flatten() for img in test_data['image']])
        y_test = test_data['label'].astype(int)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[n_neighbors] = accuracy
        print(f"Accuracy for {n_neighbors} neighbors: {accuracy}")
    return results

# Evaluate and print accuracies for each number of neighbors
train_and_evaluate_knn(neighbors_list, data_train, data_test)

# Define plot_confusion_matrix function
def plot_confusion_matrix_and_roc(conf_matrix, y_true, y_score, title):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title(title + '\nConfusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()

# Define train_and_evaluate_classifier function
def train_and_evaluate_classifier(classifier, train_data, test_data):
    X_train = np.array([img.flatten() for img in train_data['image']])
    y_train = train_data['label'].astype(int)
    X_test = np.array([img.flatten() for img in test_data['image']])
    y_test = test_data['label'].astype(int)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    y_score = classifier.predict_proba(X_test)[:, 1]  # Get the predicted probabilities for class 1
    return accuracy, precision, recall, conf_matrix, classification_rep, y_test, y_score

# KNN best model
knn_classifier = KNeighborsClassifier(n_neighbors=1)
knn_accuracy, knn_precision, knn_recall, knn_conf_matrix, knn_classification_rep, knn_y_test, knn_y_score = train_and_evaluate_classifier(knn_classifier, data_train, data_test)
print("KNN Accuracy:", knn_accuracy)
print("KNN Precision:", knn_precision)
print("KNN Recall:", knn_recall)
print("KNN Classification Report:\n", knn_classification_rep)
plot_confusion_matrix_and_roc(knn_conf_matrix, knn_y_test, knn_y_score, 'KNN')
