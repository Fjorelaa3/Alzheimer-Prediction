import os, pickle
import numpy as np
import pandas as pd
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, grayscale_conversion, build_data_generator
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.models import load_model
from visualization import plot_input_data_histogram, plot_image_from_array, plot_nn_loss, plot_nn_accuracy

# Read the input data
inputs_dir_path = os.path.abspath('C:\\Users\\Perdorues\\Desktop\\diploma\\dataset\\NN\\inputs')
results_dir_path = os.path.abspath('C:\\Users\\Perdorues\\Desktop\\diploma\\dataset\\NN\\results')

images_dir_path = os.path.join(inputs_dir_path, 'images')
labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')

labels = get_labels(labels_csv_path)
images = read_all_images(images_dir_path, True)
data = build_input_df(images, labels)

plot_input_data_histogram(data, os.path.join(results_dir_path, 'input_data_histogram.png'))

# Test a grayscale conversion
plot_image_from_array(grayscale_conversion(data.iloc[0]['image']))

# Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict):  # means all the images have the same scale
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

batch_size = 32
epochs = 100

model_path = os.path.join(results_dir_path, 'model.h5')
model_history_path = os.path.join(results_dir_path, 'model_history.pcl')
recreate_model = True



# Perform KMeans clustering
num_clusters = 5  # You can adjust this number as needed
kmeans = KMeans(n_clusters=num_clusters, random_state=123)
image_features = np.array([img.flatten() for img in data['image']])
cluster_labels = kmeans.fit_predict(image_features)
data['cluster'] = cluster_labels

# Split the data into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

datagen = build_data_generator()

generator_train = datagen.flow_from_dataframe(
    dataframe=data_train,
    x_col="filepath",
    y_col="label",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'
)

generator_test = datagen.flow_from_dataframe(
    dataframe=data_test,
    x_col="filepath",
    y_col="label",
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='binary'
)

# CNN model
if not recreate_model and os.path.isfile(model_path):  # it means the model exists
    model = load_model(model_path)
    with open(model_history_path, 'rb') as file:
        history = pickle.load(file)
else:
    # Define the CNN model
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_height, image_width, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(
        generator_train,
        steps_per_epoch=generator_train.samples // batch_size,
        epochs=epochs,
        validation_data=generator_test,
        validation_steps=generator_test.samples // batch_size
    )
    history = history.history
    # Save the training history to a file
    with open(model_history_path, 'wb') as file:
        pickle.dump(history, file)

    # Save the model
    model.save(model_path)

# Evaluate the model
print('Evaluating model...')
test_loss, test_acc = model.evaluate(generator_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Plot some graphs
# Plot loss
loss_plot_path = os.path.join(results_dir_path, 'nn_loss.png')
plot_nn_loss(history, loss_plot_path)

# Plot accuracy
accuracy_plot_path = os.path.join(results_dir_path, 'nn_accuracy.png')
plot_nn_accuracy(history, accuracy_plot_path)
# Plot confusion matrix, ROC curve, and precision-recall curve
def plot_confusion_matrix(conf_matrix, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

def plot_roc_curve(fpr, tpr, roc_auc):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_precision_recall_curve(precision, recall):
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2, label='Precision-Recall curve')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_confusion_matrix(conf_matrix, 'Hybrid Model Confusion Matrix')
plot_roc_curve(fpr, tpr, roc_auc)
plot_precision_recall_curve(precision, recall)

# Print classification report
print("Hybrid Model Classification Report:\n", report)