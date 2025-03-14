from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input, decode_predictions
import PIL
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import os
import shutil
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.layers import Rescaling
from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.model_selection import train_test_split

#Define preprocessing function
def load_and_preprocess_image(image_paths, labels):
    image = tf.io.read_file(image_paths)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.keras.applications.resnet50.preprocess_input(image)
    return image, labels


# Convert Pandas DataFrame to TensorFlow dataset
def create_tf_dataset(csv_file, folder, batch_size=64):
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

print("Datasets created successfully!")


#Load the VGG19 model
vggmodel = VGG19(weights='imagenet', include_top=False)

#Freeze the weights of the pretrained layers
for layer in vggmodel.layers:
    layer.trainable = False

# Add a new classification head based on the number of classes in the dataset
num_classes = 197
x = vggmodel.output
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Create the final model
vggmodel = tf.keras.models.Model(inputs=vggmodel.input, outputs=predictions)

# View the model summary
vggmodel.summary()

# Compile the model
vggmodel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', mode = 'min', verbose=1, patience=5, restore_best_weights=True)

# Fit the model
history = vggmodel.fit(train_ds, validation_data=val_ds, epochs=50, callbacks=[early_stopping])

# Evaluate the model on the test dataset
test_loss, test_accuracy = vggmodel.evaluate(test_ds)

#Evaluate the model on the val dataset
val_loss, val_accuracy = vggmodel.evaluate(val_ds)

# Print the evaluation results
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Val Loss: {val_loss}")
print(f"Val Accuracy: {val_accuracy}")

#Save the model
vggmodel.save('car_recognition_project/backend/models/vgg19_model.h5')
vggmodel.save('car_recognition_project/backend/models/vgg19_model.keras')