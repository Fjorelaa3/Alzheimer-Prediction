import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df
from sklearn.metrics import classification_report

# Paths
inputs_dir_path = os.path.abspath('C:/Users/drici/Downloads/MLProject-2/MLProject-2/inputs')
images_dir_path = os.path.join(inputs_dir_path, 'images')
labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')

# Load data
labels = get_labels(labels_csv_path)
images = read_all_images(images_dir_path, use_pickle=True)
data = build_input_df(images, labels)

# Check image sizes
sizes = check_image_sizes(images)
if not isinstance(sizes, dict):
    print('Images have different sizes, they need to be uniform for GBM.')
else:
    # Prepare data
    data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

    estimators = [50, 100, 200, 300, 400, 700]

# Function to train and evaluate SVM with different kernels
    def train_and_evaluate_xgb(estimators, train_data, test_data):
        results = {}
        for estimator in estimators:
            classifier = XGBClassifier(n_estimators=estimator, use_label_encoder=False, eval_metric='mlogloss')
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
    train_and_evaluate_xgb(estimators, data_train, data_test)

    
    X_train = np.array([img.flatten() for img in data_train['image']])
    y_train = data_train['label'].astype(int)
    
    X_test = np.array([img.flatten() for img in data_test['image']])
    y_test = data_test['label'].astype(int)

    # Initialize and train the XGBoost classifier
    xgb_classifier = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss')
    
    xgb_classifier.fit(X_train, y_train)

    # Predict probabilities for ROC curve and Precision-Recall
    y_proba = xgb_classifier.predict_proba(X_test)[:, 1]  
    
    # Predictions and evaluation
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Plotting the confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Labels')
    plt.xlabel('Predicted Labels')
    plt.show()

    print(f'Accuracy: {accuracy}')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    # Precision-Recall Curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (area = {pr_auc:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.show()

    y_pred = xgb_classifier.predict(X_test)

# Generate classification report
xgb_classification_report = classification_report(y_test, y_pred)
print("XGBoost Classification Report:")
print(xgb_classification_report)
