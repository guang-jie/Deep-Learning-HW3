import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import tensorflow.python.profiler as profiler
import time



# Load MNIST dataset
(train_dataset, test_dataset), dataset_info = tfds.load('mnist', split=['train', 'test'], shuffle_files=True, with_info=True, as_supervised=True)

# Split train_dataset into train and valid datasets
train_size = dataset_info.splits['train'].num_examples
validation_ratio = 0.2
validation_size = int(train_size * validation_ratio)

valid_dataset = train_dataset.take(validation_size)
train_dataset = train_dataset.skip(validation_size)
print(len(train_dataset), len(valid_dataset), len(test_dataset))

# Preprocess the data
def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    label = tf.one_hot(label, 10)
    return image, label

train_dataset = train_dataset.map(preprocess)
valid_dataset = valid_dataset.map(preprocess)
test_dataset = test_dataset.map(preprocess)

# Set batch size and buffer size for shuffling
batch_size = 32
buffer_size = 10000

# Shuffle and batch the data
train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size)
valid_dataset = valid_dataset.batch(batch_size)
test_dataset = test_dataset.batch(batch_size)

# Define LeNet-5 model
def LeNet5(input_shape, num_classes):
    model = tf.keras.Sequential()
    
    # Convolutional layers
    model.add(layers.Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    
    # Flatten the output from previous layer
    model.add(layers.Flatten())
    
    # Fully connected layers
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# Create LeNet-5 model
input_shape = (28, 28, 1)  # Input shape of MNIST images
num_classes = 10  # Number of output classes (digits 0-9)
model = LeNet5(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
epochs = 10

train_accuracy_list = []
valid_accuracy_list = []

for epoch in range(epochs):
    # Train the model
    model.fit(train_dataset, epochs=1)

    # Evaluate on training data
    train_loss, train_accuracy = model.evaluate(train_dataset)
    
    # Evaluate on validation data
    valid_loss, valid_accuracy = model.evaluate(valid_dataset)
    print(f"Epoch {epoch+1}, Valid Loss: {valid_loss:.4f}, Valid Accuracy: {valid_accuracy:.4f}")
    
    # Append accuracy to lists
    train_accuracy_list.append(train_accuracy * 100)
    valid_accuracy_list.append(valid_accuracy * 100)
    break

# Evaluate the model on test data
start_time = time.time()
test_loss, test_accuracy = model.evaluate(test_dataset)
inference_time = time.time() - start_time
print('Test Loss:', test_loss)
print('Test Accuracy:', test_accuracy)
print('inference_time:', inference_time)

# 輸出模型的參數量和層次結構
model.summary()

# 計算模型的空間複雜度
total_params = model.count_params()
memory_size = total_params * 4  # 假設每個參數使用4個位元組的記憶體

# 輸出結果
print("Total Parameters:", total_params)
print("Memory Size (bytes):", memory_size)
print("Memory Size (MB):", memory_size / (1024 * 1024))

'''
# Profile the model and calculate FLOPs
profile = tf.profiler.profile(
    tf.get_default_graph(),
    options=tf.profiler.ProfileOptionBuilder.float_operation())

flops = profile.total_float_ops
print("Total FLOPs:", flops)
'''

# Plot accuracy curve
plt.plot(range(1, epochs+1), train_accuracy_list, label='Train')
plt.plot(range(1, epochs+1), valid_accuracy_list, label='Valid')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()
