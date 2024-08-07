import os
import csv

# Set the paths for the train and test folders
train_folder = 'C:\\Users\\Perdorues\\Desktop\\Alzheimer_s\\train'
test_folder = 'C:\\Users\\Perdorues\\Desktop\\Alzheimer_s\\test'

# Define the class labels
class_labels = {
    'NonDemented': 0,
    'ModerateDemented': 1,
    'MildDemented': 2,
    'VeryMildDemented': 3
}

# Create the labels.csv file
with open('labels.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_id', 'diagnosis'])  # Write the header row

    # Process the train folder
for class_name, label in class_labels.items():
          class_folder = os.path.join(train_folder, class_name)
for image_file in os.listdir(class_folder):
            if image_file.endswith('.jpg'):
                 image_id = os.path.splitext(image_file)[0]
            writer.writerow([image_id, label])
                
#with open('labels2.csv', 'w', newline='') as csvfile:
    #writer = csv.writer(csvfile)
    #writer.writerow(['image_id', 'diagnosis'])  # Write the header row


    # Process the test folder
    #for class_name, label in class_labels.items():
     #  class_folder = os.path.join(test_folder, class_name)
    #for image_file in os.listdir(class_folder):
            #if image_file.endswith('.jpg'):
               # image_id = os.path.splitext(image_file)[0]
               # writer.writerow([image_id, label])

print("labels.csv file has been generated successfully.")