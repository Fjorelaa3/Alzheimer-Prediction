import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import numpy as np

# Path to the directory where your 'images inputs' folder is located
base_dir = 'C:/Users/drici/Downloads/MLProject-2/MLProject-2/inputs' 
source_dir = os.path.join(base_dir, 'images')

# Create train and test directories with class subdirectories
train_dir = os.path.join(base_dir, 'data/train')
test_dir = os.path.join(base_dir, 'data/test')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

train_class0_dir = os.path.join(train_dir, 'class0')
train_class1_dir = os.path.join(train_dir, 'class1')
test_class0_dir = os.path.join(test_dir, 'class0')
test_class1_dir = os.path.join(test_dir, 'class1')
os.makedirs(train_class0_dir, exist_ok=True)
os.makedirs(train_class1_dir, exist_ok=True)
os.makedirs(test_class0_dir, exist_ok=True)
os.makedirs(test_class1_dir, exist_ok=True)

# Get all images from the source directory
all_images = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f))]

# Split the dataset into training and testing sets (80% train, 20% test)
train_files, test_files = train_test_split(all_images, test_size=0.2, random_state=42)

# Function to copy files to the new directory based on the filename
def copy_files(files, train_or_test_dir):
    for file in files:
        if 'nonDem' in file:
            shutil.copy(os.path.join(source_dir, file), os.path.join(train_or_test_dir, 'class0'))
        else:
            shutil.copy(os.path.join(source_dir, file), os.path.join(train_or_test_dir, 'class1'))

# Copy the split files into their respective directories
copy_files(train_files, train_dir)
copy_files(test_files, test_dir)


train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,  
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True, 
        fill_mode='nearest',
        validation_split=0.2)

train_generator = train_datagen.flow_from_directory(
        train_dir,  # This is the target directory
        target_size=(176, 208),  # All images will be resized to 176x208
        batch_size=32,
        class_mode='binary',  # Since we use categorical_crossentropy loss, we need categorical labels
        subset='training')

validation_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(176, 208),
        batch_size=32,
        class_mode='binary',
        subset='validation')

model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (176, 208, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation = 'relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation = 'relu'),
        layers.Dense(1, activation = 'sigmoid')
    ])


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    epochs=100,
    validation_data=validation_generator
    )

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,  # This is the target directory for test data
        target_size=(176, 208),
        batch_size=32,
        class_mode='binary',
        shuffle=False)  # Important for later comparison in evaluation

test_loss, test_acc = model.evaluate(test_generator, steps=50)  # steps should cover all samples
print("Test accuracy:", test_acc)
print("Test loss:", test_loss)

predictions = model.predict(test_generator, steps=50)  # Adjust steps to cover all test samples
predicted_classes = predictions.argmax(axis=1)

# Function to plot the training and validation accuracy and loss at each epoch
def plot_training_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

# Call this function with your training history
plot_training_history(history)


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], '.2f' if normalize else 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def plot_roc_curve(test_labels, test_predictions):
    fpr, tpr, _ = roc_curve(test_labels, test_predictions)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# Predict classes to use in confusion matrix and ROC curve
test_generator.reset()
predictions = model.predict(test_generator, steps=int(np.ceil(test_generator.samples / test_generator.batch_size)))
predicted_classes = (predictions > 0.5).astype(int)
true_classes = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(true_classes, predicted_classes)
plot_confusion_matrix(cm, classes=['Class0', 'Class1'])

# Classification Report
print(classification_report(true_classes, predicted_classes, target_names=['Class0', 'Class1']))

# ROC Curve
if len(np.unique(true_classes)) == 2:
    plot_roc_curve(true_classes, predictions)

