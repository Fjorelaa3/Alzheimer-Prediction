import os
import pickle
import numpy as np
import pandas as pd
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report

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
if isinstance(sizes, dict):  # means all the images have the same scale
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')


# Split the data into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

estimators = [37, 50,100, 200, 250, 300, 500, 1000]

# Function to train and evaluate SVM with different kernels
def train_and_evaluate_rf(estimators, train_data, test_data):
    results = {}
    for estimator in estimators:
        classifier = RandomForestClassifier(n_estimators=estimator, random_state=123)
        X_train = np.array([img.flatten() for img in train_data['image']])
        y_train = train_data['label'].astype(int)
        X_test = np.array([img.flatten() for img in test_data['image']])
        y_test = test_data['label'].astype(int)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        results[estimator] = accuracy
        print(f"Accuracy for {estimator} estimators: {accuracy}")
    return results

# Evaluate and print accuracies for each kernel type
train_and_evaluate_rf(estimators, data_train, data_test)

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
    y_proba = classifier.predict_proba(X_test)[:, 1]  # For ROC and Precision-Recall

    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    return X_test, y_test, accuracy, conf_matrix, fpr, tpr, roc_auc, precision, recall, pr_auc

# Random Forest best model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=123)
X_test, y_test, rf_accuracy, rf_conf_matrix, fpr, tpr, roc_auc, precision, recall, pr_auc = train_and_evaluate_classifier(rf_classifier, data_train, data_test)
y_pred_rf = rf_classifier.predict(X_test)
print("Random Forest Accuracy:", rf_accuracy)
plot_confusion_matrix(rf_conf_matrix, 'Random Forest Confusion Matrix')

# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Plot Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.show()

# Classification Report
rf_classification_report = classification_report(y_test, y_pred_rf)
print("Random Forest Classification Report:")
print(rf_classification_report)