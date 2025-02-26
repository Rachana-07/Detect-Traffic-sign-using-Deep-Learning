# Import required libraries
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Load a preprocessed images and the corresponding labels
image, labels = np.load('batch.npy', allow_pickle=True).tolist()

# Hyperparameters
input_size = image.shape[1]  # Dimension of input image
num_classes = labels['classifier_head'].shape[1]  # Number of classes
DROPOUT_FACTOR = 0.2  # Dropout probability

# Visualize one example preprocessed image
plt.imshow(image[2])

# Create a sequential model - a linear stack of layers
model = keras.Sequential()

# Feature extractor
model.add(keras.layers.Conv2D(16, kernel_size=3, activation='relu', input_shape=(input_size, input_size, 3)))
model.add(keras.layers.AveragePooling2D(2, 2))
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu'))
model.add(keras.layers.AveragePooling2D(2, 2))
model.add(keras.layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(keras.layers.Dropout(DROPOUT_FACTOR))
model.add(keras.layers.AveragePooling2D(2, 2))

# Model adaptor
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(64, activation='relu'))

# Classifier head
model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dense(num_classes, activation='softmax', name='classifier_head'))

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(image, labels['classifier_head'], epochs=20)

# Print the training accuracy
accuracy = history.history['accuracy'][-1]
print(f"Training accuracy: {accuracy}")
