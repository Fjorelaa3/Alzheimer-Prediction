import numpy as np, pandas as pd, os, tensorflow as tf, pickle, cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from skimage import io
from skimage.transform import resize
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Function to get the labels
def get_labels(labels_path):
    labels = pd.read_csv(labels_path)
    labels.rename(columns = {'id_code': 'filename', 'diagnosis': 'label'}, inplace = True)
    return labels

# Function to norm images
def norm_image(image):
    image = image.astype('float32') / 255.0  # Normalizing pixel values between 0 and 1
    return image  # Normalizing pixel values between 0 and 1


# Function to translate the number to sickness level
def num_to_diagnosis(val):
    array = ['no_dr', 'mild', 'moderate', 'severe', 'proliferate']
    if val >= 0 and val < 4:
        return array[int(val)]
    else:
        print('Val has to be between 0 and 3')

# Function to read all the images
def read_all_images(image_dir_path, use_pickle = True):
    pickle_file = os.path.join(image_dir_path, 'all_images.pcl')
    if use_pickle and os.path.isfile(pickle_file):
        with open(pickle_file, 'rb') as file:
            all_images = pickle.load(file)
            return all_images
    all_images = []
    for filename in os.listdir(image_dir_path):
        if filename.endswith(".jpg"):
            image = io.imread(os.path.join(image_dir_path, filename))
            all_images.append({
                    'filepath': os.path.join(image_dir_path, filename),
                    'filename': filename.split('.')[0],
                    'image': image
                })
    all_images = pd.DataFrame.from_dict(all_images)
    with open(pickle_file, 'wb') as file:
        pickle.dump(all_images, file)
    return all_images

# Function to check the image sizes
def check_image_sizes(images):
    sizes = []
    for i, row in images.iterrows():
        sizes.append(row['image'].shape)
    sizes = set(sizes)
    if len(sizes) == 1:
        print('All images have the same size: ', sizes)
        sizes = list(sizes)[0]
        width = sizes[0]
        height = sizes[1]
        return {'width': width, 'height': height}
    else:
        print('Images have different sizes: ', sizes)
        return sizes


# Function to build input
def build_input_df(images, labels):
    data = pd.merge(images, labels, on = 'filename', how = 'inner')
    data.loc[data['label'] >=1, 'label'] = 1         
    data['label'] = data['label'].astype(str)
    return data    



def build_data_generator():
    datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 20,
        width_shift_range = 0.1,
        height_shift_range = 0.1,
        shear_range = 0.1,
        zoom_range = 0.1,
        horizontal_flip = True,
        fill_mode = 'nearest',
    )
    return datagen

    
    