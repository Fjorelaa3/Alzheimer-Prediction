import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.style import context
# import os, pickle
# from helpers import get_labels, read_all_images, check_image_sizes, build_input_df, build_data_generator

def plot_input_data_histogram(data, save_path = None):
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        label_counts = data['label'].value_counts()
        # ax.hist(data['label'])
        ax.bar(label_counts.index, label_counts.values)
        ax.set_title("Input data histogram")
        ax.set_xlabel("Diagnosis")
        ax.set_ylabel("Occurrence")
        # ax.grid(visible = True, which = 'both')
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)
    
def plot_image_from_array(data, save_path = None):
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.imshow(data, cmap = 'gray')
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)
            
        
def plot_nn_loss(data, save_path = None):
    loss_train = data['loss']
    loss_test = data['val_loss']
    epochs = np.array(range(len(loss_train))) + 1
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.plot(epochs, loss_train, label = 'train loss')
        ax.plot(epochs, loss_test, label = 'validation loss')
        ax.set_title("Neural network loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(visible = True, which = 'both')
        ax.legend()
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)
    
def plot_nn_accuracy(data, save_path = None):
    accuracy_train = data['accuracy']
    accuracy_test = data['val_accuracy']
    epochs = np.array(range(len(accuracy_train))) + 1
    with context('default'):
        fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
        ax.plot(epochs, accuracy_train, label = 'train accuracy')
        ax.plot(epochs, accuracy_test, label = 'validation accuracy')
        ax.set_title("Neural network accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy")
        ax.grid(visible = True, which = 'both')
        ax.legend()
        plt.show()
        if save_path is not None:
            fig.savefig(save_path)


#%% Read the inpud data
# inputs_dir_path = os.path.abspath('C:\Users\drici\Downloads\MLProject-2\MLProject-2\inputs')
# results_dir_path = os.path.abspath('C:\Users\drici\Downloads\MLProject-2\MLProject-2\results')

# images_dir_path = os.path.join(inputs_dir_path, 'images')
# labels_csv_path = os.path.join(inputs_dir_path, 'labels.csv')

# labels = get_labels(labels_csv_path)
# images = read_all_images(images_dir_path, True)
# data = build_input_df(images, labels)

# plot_input_data_histogram(data, os.path.join(results_dir_path, 'input_data_histogram.png'))



# #%% Define some variables
# sizes = check_image_sizes(images)
# if isinstance(sizes, dict): # means all the images have the same scale
#     image_width = sizes['width']
#     image_height = sizes['height']
# else:
#     print('Please check the image sizes, because they need to be equal!')

# batch_size = 32
# epochs = 100

# model_path = os.path.join(results_dir_path, 'modelML.h5')
# model_history_path = os.path.join(results_dir_path, 'model_historyML.pcl')
# recreate_model = True

# #%% Build train and test data

# # Split the data into train and test
# data_train, data_test = train_test_split(data, test_size = 0.2, random_state = 123)

# datagen = build_data_generator()

# generator_train = datagen.flow_from_dataframe(
#     dataframe = data_train,
#     x_col = "filepath",
#     y_col = "label",
#     target_size = (image_width, image_height),
#     batch_size = batch_size,
#     class_mode = 'binary'
# )

# generator_test = datagen.flow_from_dataframe(
#     dataframe = data_test,
#     x_col = "filepath",
#     y_col = "label",
#     target_size = (image_width, image_height),
#     batch_size = batch_size,
#     class_mode = 'binary'
# )

# #%% NN
# if not recreate_model and os.path.isfile(model_path): # it means the model exists
#     model = load_model(model_path)
#     with open(model_history_path, 'rb') as file:
#         history = pickle.load(file)
# else:    
#     # Define the CNN model
#     model = models.Sequential([
#         layers.Conv2D(32, (3, 3), activation = 'relu', input_shape = (image_height, image_width, 3)),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(64, (3, 3), activation = 'relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Conv2D(128, (3, 3), activation = 'relu'),
#         layers.MaxPooling2D((2, 2)),
#         layers.Flatten(),
#         layers.Dense(128, activation = 'relu'),
#         layers.Dense(1, activation = 'sigmoid')
#     ])
    
#     # Compile the model
#     model.compile(optimizer = 'adam',
#                   loss = 'binary_crossentropy',
#                   metrics = ['accuracy'])
    
#     # Train the model
#     history = model.fit(
#         generator_train,
#         steps_per_epoch = generator_train.samples // batch_size,
#         epochs = epochs,
#         validation_data = generator_test,
#         validation_steps = generator_test.samples // batch_size
#     )
#     history = history.history
#     # Save the training history to a file
#     with open(model_history_path, 'wb') as file:
#         pickle.dump(history, file)
    
#     # Save the model
#     model.save(model_path)

# # Evaluate the model
# print('Evaluating model...')
# test_loss, test_acc = model.evaluate(generator_test, verbose = 2)
# print('\nTest accuracy:', test_acc)

# #%% Plot some graphs

# # Plot loss
# loss_plot_path = os.path.join(results_dir_path, 'nn_loss.png')
# plot_nn_loss(history, loss_plot_path)

# # Plot accuracy
# accuracy_plot_path = os.path.join(results_dir_path, 'nn_accuracy.png')
# plot_nn_accuracy(history, accuracy_plot_path)