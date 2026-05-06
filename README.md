# 👕 Fashion MNIST Image Classifier

**Deep Learning | Multi-Class Classification | TensorFlow/Keras | Computer Vision**

A fully connected neural network built with TensorFlow/Keras to classify grayscale clothing images from the classic Fashion MNIST dataset into 10 categories.

---

## 🚀 Features

- **End-to-End Pipeline**: Load data → preprocess → train → evaluate → predict
- **Data Normalization**: Pixel value scaling for stable and fast convergence
- **Simple yet Effective Architecture**: Flatten + Dense layers demonstrating core NN concepts
- **Multi-Class Prediction**: Softmax output with probability analysis

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| Framework | TensorFlow 2.x / Keras |
| Dataset | Keras Built-in `fashion_mnist` |
| Computing | NumPy |
| Visualization | Matplotlib |

---

## 🧠 Model Architecture
Input (28×28)
↓
Flatten → 784 neurons
↓
Dense(128) → ReLU
↓
Dense(10) → Softmax

- **Total Parameters**: ~101,770 (lightweight and fast to train)
- **Optimizer**: Adam
- **Loss**: Sparse Categorical Crossentropy
- **Metric**: Accuracy

---

## 📊 Dataset

- **Source**: [Zalando Research - Fashion MNIST](https://www.kaggle.com/datasets/zalando-research/fashionmnist)
- **Size**: 60,000 train + 10,000 test images
- **Format**: 28×28 grayscale (0=black, 255=white)
- **Classes**: T-shirt/top, Trouser, Pullover, Dress, Coat, Sandal, Shirt, Sneaker, Bag, Ankle boot

---

## 🔧 Preprocessing

1. **Normalization**: `image / 255.0` → scales to [0, 1]
2. **Flattening**: 2D (28,28) → 1D (784) for dense layer input

---

## 📈 Training

```python
model.fit(train_images, train_labels, epochs=10)
🎯 Prediction Example
predictions = model.predict(test_images)
predicted_class = np.argmax(predictions[20])  # Highest probability
true_class = test_labels[20]
💡 Key Learnings
| Concept                         | Application                          |
| ------------------------------- | ------------------------------------ |
| Normalization                   | Stable gradient flow                 |
| ReLU Activation                 | Non-linear pattern learning          |
| Softmax                         | Multi-class probability distribution |
| Sparse Categorical Crossentropy | Efficient integer-label training     |
🔮 Future Improvements
[ ] Add Dropout layers to reduce overfitting
[ ] Implement CNN (Conv2D) for better spatial feature extraction
[ ] Add data augmentation (rotation, shift) for robustness
[ ] Visualize confusion matrix for per-class performance
[ ] Plot training curves (accuracy/loss over epochs)
