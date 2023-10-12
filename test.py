import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img

test_dataset_path = "test/"

labels = ["A", "B", "C","D","E","F","G","H","I","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]

model = load_model("Models\keras_model.h5")
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

image_height = 224
image_width = 224

test_images = []
test_target_labels = []

for label in labels:
    label_path = os.path.join(test_dataset_path, label).replace("\\", "/")
    image_files = os.listdir(label_path)
    
    for image_file in image_files:
        image_path = os.path.join(label_path, image_file).replace("\\", "/")
        image = load_img(image_path, target_size=(image_height, image_width))
        image = img_to_array(image)
        image = image / 255.0  
        test_images.append(image)
        test_target_labels.append(labels.index(label))

test_images = np.array(test_images)
test_target_labels = np.array(test_target_labels)

test_loss, test_accuracy = model.evaluate(test_images, test_target_labels)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

predicted_labels = model.predict(test_images)
predicted_labels = np.argmax(predicted_labels, axis=1)

# 1. Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(test_target_labels, predicted_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# 2. Accuracy per Class
from sklearn.metrics import classification_report

class_report = classification_report(test_target_labels, predicted_labels, target_names=labels)
print("Classification Report:")
print(class_report)

# 3. Sample Image Visualization with Predicted and True Labels
num_samples_to_visualize = 5
indices = np.random.choice(len(test_images), num_samples_to_visualize, replace=False)

plt.figure(figsize=(12, 8))
for i, idx in enumerate(indices):
    plt.subplot(1, num_samples_to_visualize, i+1)
    plt.imshow(test_images[idx])
    plt.title(f"True: {labels[test_target_labels[idx]]}, Pred: {labels[predicted_labels[idx]]}")
    plt.axis('off')
plt.suptitle("Sample Image Visualization with Predicted and True Labels")
plt.show()

# 4. Bar Chart for Correct and Incorrect Predictions
correct_predictions = (predicted_labels == test_target_labels)
correct_counts = np.sum(correct_predictions)
incorrect_counts = len(test_images) - correct_counts

plt.figure(figsize=(6, 4))
plt.bar(["Correct", "Incorrect"], [correct_counts, incorrect_counts], color=['green', 'red'])
plt.xlabel("Prediction")
plt.ylabel("Count")
plt.title("Correct vs. Incorrect Predictions")
plt.show()

# 5. Accuracy Distribution across Classes
class_accuracies = [np.mean(correct_predictions[test_target_labels == i]) for i in range(len(labels))]

plt.figure(figsize=(6, 4))
plt.bar(labels, class_accuracies, color='orange')
plt.xlabel("Class")
plt.ylabel("Accuracy")
plt.title("Accuracy Distribution across Classes")
plt.ylim(0, 1)
plt.show()
