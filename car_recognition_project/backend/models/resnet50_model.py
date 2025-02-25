from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
import PIL
import numpy as np
import pandas as pd
import cv2
from scipy.io import loadmat
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
import os
import shutil
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

annot = loadmat(r'dataset\cars_annos.mat')
print(annot.keys())

# Check structure of annotations
print(type(annot['annotations']))
print(annot['annotations'].shape)

# Check structure of class_names
print(type(annot['class_names']))
print(annot['class_names'].shape)
print(annot['class_names'])

#Extracting the class names
class_names = [c for c in annot['class_names'][0]]

#Extracting the image paths and corresponding labels
annotations = annot['annotations']
image_paths = []
labels = []
label_names = []
bbox_x1 = []
bbox_y1 = []    
bbox_x2 = []
bbox_y2 = []

# Extract the image paths and labels
for i in range(annotations.shape[1]):  # Iterate over all images
    image_path = annotations[0][i][0][0].replace('car_ims/', '')
    image_paths.append(image_path)
    
    # Extract the labels
    labels.append(annotations[0][i][5][0][0])
    label_names.append(class_names[labels[-1]-1])
    
# Display some example images
for i in range(5):
    print(image_paths[i], labels[i], label_names[i])

# Create a DataFrame with the image paths and labels
df = pd.DataFrame({
    'image_paths': image_paths, 
    'labels': labels, 
    'label_names': label_names,
})

# Display the first rows of the DataFrame
print(df.head())

#save the dataframe to a csv file
df.to_csv(r'dataset/cars_annos.csv', index=False)

# Load the DataFrame from the CSV file
df = pd.read_csv(r'dataset/cars_annos.csv')

# Define the split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Split the data into training, validation and test sets
df_train, df_temp = train_test_split(df, test_size=(val_ratio + test_ratio), stratify=df['labels'], random_state=42)

df_val, df_test = train_test_split(df_temp, test_size=test_ratio/(val_ratio + test_ratio), stratify=df_temp['labels'], random_state=42)

# Check the split sizes
print('Train size:', df_train.shape[0])
print('Validation size:', df_val.shape[0])
print('Test size:', df_test.shape[0])

# Save the split dataframes to CSV files
df_train.to_csv(r'dataset/cars_train.csv', index=False)
df_val.to_csv(r'dataset/cars_val.csv', index=False)
df_test.to_csv(r'dataset/cars_test.csv', index=False)

# Create directories for the training, validation and test sets
os.makedirs(r'dataset/train', exist_ok=True)
os.makedirs(r'dataset/val', exist_ok=True)
os.makedirs(r'dataset/test', exist_ok=True)

# Move the images to the corresponding directories
def move_images(df, folder):
    for _, row in df.iterrows():
        src = 'dataset/images/' + row['image_paths']
        dst = os.path.join(folder, os.path.basename(src))
        shutil.copyfile(src, dst)

# Move images
move_images(df_train, "dataset/train")
move_images(df_val, "dataset/val")
move_images(df_test, "dataset/test")

print("Images successfully split into train, val, and test folders!")


#Define preprocessing function
def load_and_preprocess_image(image_paths, labels):
    image = tf.io.read_file(image_paths)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, labels

# Convert Pandas DataFrame to TensorFlow dataset
def create_tf_dataset(csv_file, folder, batch_size=32):
    df = pd.read_csv(csv_file)
    image_paths = [os.path.join(folder, path) for path in df['image_paths'].values]
    labels = df['labels'].values
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    dataset = dataset.map(load_and_preprocess_image).batch(batch_size).shuffle(1000).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Create train, validation, and test datasets
train_ds = create_tf_dataset(r'dataset\cars_train.csv', 'dataset/train')
val_ds = create_tf_dataset(r'dataset\cars_val.csv', 'dataset/val')
test_ds = create_tf_dataset(r'dataset\cars_test.csv', 'dataset/test')

print("Datasets ready for training!")

# Load the ResNet50 model
model = ResNet50(weights='imagenet', include_top=False)

# Freeze the weights of the pre-trained layers
for layer in model.layers:
    layer.trainable = False

# Add a new classification head based on the number of classes in the dataset
num_classes = 197
x = model.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create the final model
model = tf.keras.models.Model(inputs=model.input, outputs=predictions)

# View the model summary
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=5, restore_best_weights=True)

# Fit the model
history = model.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping])


