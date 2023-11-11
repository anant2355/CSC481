import os
import cv2
import glob
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# list the flower types 
root=os.path.join(os.getcwd(),'flower_images/')
dirlist = [ item for item in os.listdir(root) if os.path.isdir(os.path.join(root, item)) ]

# list of image paths and the respictive label
image_paths = []
class_labels = []
for dir_ in dirlist:
    for filename in glob.glob(os.path.join(os.path.join(root,dir_),'*.jpg')):
        image_paths.append(filename)
        class_labels.append(dir_)

# label the flowers 
label_to_index = {'Lotus': 0, 'Tulip': 1, 'Orchid': 2, 'Lilly': 3, 'Sunflower': 4}

orginal_images = []
enhanced_images = []
canny_edge_images = []
morphology_opening_images = []
morphology_closing_images = []
labels = []

for path, label in zip(image_paths, class_labels):
    
    # read the image
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # resize the image
    image = cv2.resize(image, (128, 128))
    
    # Contrast enhancement of the image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Edge detection and convert image to grey scale
    greyscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny_edge_image = cv2.Canny(image, 100, 200)
        
    # Mathematical morphology
    kernel = np.ones((5,5),np.uint8)
    mean_intensity = np.mean(image)
    _, binary_image = cv2.threshold(greyscale_image, mean_intensity, 255, cv2.THRESH_BINARY)
    morphology_opening_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    morphology_closing_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # append the images to the respective list
    orginal_images.append(image)
    enhanced_images.append(enhanced_image)
    canny_edge_images.append(canny_edge_image)
    morphology_opening_images.append(morphology_opening_image)
    morphology_closing_images.append(morphology_closing_image)
    labels.append(label_to_index[label])
    
# convert the lists to numpy arrays
orginal_images = np.array(orginal_images, dtype='float32')
enhanced_images = np.array(enhanced_images, dtype='float32')
canny_edge_images = np.array(canny_edge_images, dtype='float32')
canny_edge_images = np.expand_dims(canny_edge_images, axis=-1)
morphology_opening_images = np.array(morphology_opening_images, dtype='float32')
morphology_opening_images = np.expand_dims(morphology_opening_images, axis=-1)
morphology_closing_images = np.array(morphology_closing_images, dtype='float32')
morphology_closing_images = np.expand_dims(morphology_closing_images, axis=-1)


# normalize the images
orginal_images /= 255.0
enhanced_images /= 255.0
canny_edge_images /= 255.0
morphology_opening_images /= 255.0
morphology_closing_images /= 255.0

labels = np.array(labels)
labels = to_categorical(labels, num_classes=len(label_to_index))


num_classes = len(dirlist)
total_class = 1000
num_train = 750
num_test = 250


# train test split the images , 750 images of each type to training and 250 images to testing 
def train_test_split(X_data):
    X_train,y_train,X_test,y_test = [],[],[],[]
    for i in range(num_classes):
        start = i * total_class
        train_index = start + num_train
        test_index = train_index + num_test

        X_train.extend(X_data[start:train_index])
        y_train.extend(labels[start:train_index])

        X_test.extend(X_data[train_index:test_index])
        y_test.extend(labels[train_index:test_index])
    return np.array(X_train),np.array(y_train),np.array(X_test),np.array(y_test)
    
X_train_original, y_train_original, X_test_original, y_test_original = train_test_split(orginal_images)
X_train_enhanced, y_train_enhanced, X_test_enhanced, y_test_enhanced = train_test_split(enhanced_images)
X_train_canny_edge, y_train_canny_edge, X_test_canny_edge, y_test_canny_edge = train_test_split(canny_edge_images)
X_train_morphology_opening, y_train_morphology_opening, X_test_morphology_opening, y_test_morphology_opening = train_test_split(morphology_opening_images)
X_train_morphology_closing, y_train_morphology_closing, X_test_morphology_closing, y_test_morphology_closing = train_test_split(morphology_closing_images)


# CNN model
def model_training(X_train, y_train, X_test, y_test, n= 3):
    model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, n)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(label_to_index), activation='softmax') ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=10, validation_split=0.1)

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    return test_loss, test_accuracy

# get the accuracy for each of the image type
start = time.time()
test_loss_original, test_accuracy_original = model_training(X_train_original, y_train_original, X_test_original, y_test_original)
print(f'Time: {time.time() - start} seconds taken for 10 epochs - original images')

start = time.time()
test_loss_enhanced, test_accuracy_enhanced = model_training(X_train_enhanced, y_train_enhanced, X_test_enhanced, y_test_enhanced)
print(f'Time: {time.time() - start} seconds taken for 10 epochs - enhanced images')

start = time.time()
test_loss_canny_edge, test_accuracy_canny_edge = model_training(X_train_canny_edge, y_train_canny_edge, X_test_canny_edge, y_test_canny_edge, n=1)
print(f'Time: {time.time() - start} seconds taken for 10 epochs - canny edge images')

start = time.time()
test_loss_morphology_opening, test_accuracy_morphology_opening = model_training(X_train_morphology_opening, y_train_morphology_opening, X_test_morphology_opening, y_test_morphology_opening, n=1)
print(f'Time: {time.time() - start} seconds taken for 10 epochs - morphology opening images')

start = time.time()
test_loss_morphology_closing, test_accuracy_morphology_closing = model_training(X_train_morphology_closing, y_train_morphology_closing, X_test_morphology_closing, y_test_morphology_closing, n=1)
print(f'Time: {time.time() - start} seconds taken for 10 epochs - morphology closing images')

print(f"The accuracy on the test set for the original images is {test_accuracy_original}")
print(f"The accuracy on the test set for the enhanced images is {test_accuracy_enhanced}")
print(f"The accuracy on the test set for the canny edge images is {test_accuracy_canny_edge}")
print(f"The accuracy on the test set for the morphology opening images is {test_accuracy_morphology_opening}")
print(f"The accuracy on the test set for the morphology closing images is {test_accuracy_morphology_closing}")