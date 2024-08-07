import os
import pickle
from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, build_data_generator
from tensorflow.keras import layers, models, Input
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNet

from sklearn.model_selection import train_test_split
from visualization import plot_input_data_histogram, plot_image_from_array, plot_MobileNet_loss, plot_MobileNet_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define paths
inputs_dir_path = os.path.abspath('C:\\Users\\HP\\Desktop\\MLproject\\inputs')
results_dir_path = os.path.abspath('C:\\Users\\HP\\Desktop\\MLproject\\results')
model_path = os.path.join(results_dir_path, 'mobilenet_model.h5')  # Change model name to mobilenet_model.h5
model_history_path = os.path.join(results_dir_path, 'mobilenet_model_history.pcl')  # Change model history name to mobilenet_model_history.pcl

# Read data
labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')
labels = get_labels(labels_csv_path)
images_dir_path = os.path.join(inputs_dir_path, 'images')
images = read_all_images(images_dir_path, True)
data = build_input_df(images, labels)

# Plot input data histogram
plot_input_data_histogram(data, os.path.join(results_dir_path, 'input_data_histogram.png'))

# Define some variables
sizes = check_image_sizes(images)
if isinstance(sizes, dict):  # means all the images have the same scale
    image_width = sizes['width']
    image_height = sizes['height']
else:
    print('Please check the image sizes, because they need to be equal!')

batch_size = 32
epochs = 100

# Split the data into train and test
data_train, data_test = train_test_split(data, test_size=0.2, random_state=123)

# Define data generators
datagen_train = ImageDataGenerator(rescale=1./255)  # You might need to adjust other parameters
datagen_test = ImageDataGenerator(rescale=1./255)   # Based on your data preprocessing needs

generator_train = datagen_train.flow_from_dataframe(
    dataframe=data_train,
    x_col="filepath",
    y_col="label",
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

generator_test = datagen_test.flow_from_dataframe(
    dataframe=data_test,
    x_col="filepath",
    y_col="label",
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='binary'
)

# Define recreate_model
recreate_model = True

# MobileNet model
if not recreate_model and os.path.isfile(model_path):  # it means the model exists
    model = load_model(model_path)
    with open(model_history_path, 'rb') as file:
        history = pickle.load(file)
else:
    # Define the MobileNet model
    base_model = MobileNet(input_shape=(image_height, image_width, 3), include_top=False, weights='imagenet')
    for layer in base_model.layers:
        layer.trainable = False
    
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(128, activation='relu')(x)
    predictions = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)
    
    # Compile the model
    model.compile(optimizer=Adam(),
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

# Plot loss
loss_plot_path = os.path.join(results_dir_path, 'mobilenet_loss.png')  # Change loss plot name to mobilenet_loss.png
plot_MobileNet_loss(history, loss_plot_path)

# Plot accuracy
accuracy_plot_path = os.path.join(results_dir_path, 'mobilenet_accuracy.png')  # Change accuracy plot name to mobilenet_accuracy.png
plot_MobileNet_accuracy(history, accuracy_plot_path)
from sklearn.metrics import classification_report

# Evaluate the model
print('Evaluating model...')
test_loss, test_acc = model.evaluate(generator_test, verbose=2)
print('\nTest accuracy:', test_acc)

# Generate predictions
print('Generating predictions...')
y_pred = model.predict(generator_test)
y_pred_classes = (y_pred > 0.5).astype("int32")

# Get true labels
y_true = generator_test.classes

# Calculate precision, recall, and classification report
print('Calculating precision, recall, and classification report...')
classification_rep = classification_report(y_true, y_pred_classes)
print('Classification Report:\n', classification_rep)
