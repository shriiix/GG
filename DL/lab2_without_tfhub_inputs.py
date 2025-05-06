import tensorflow as tf
from mlxtend.plotting import plot_confusion_matrix
from sklearn import metrics
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from tqdm.notebook import tqdm
import warnings
warnings.filterwarnings("ignore")

# ---

vocab_size = 10000
max_len = 200
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.imdb.load_data(num_words=vocab_size)

# ---

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# ---

x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

# ---

x_train.shape, y_train.shape, x_test.shape, y_test.shape

# ---

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 128, input_length=max_len),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# ---

tf.keras.utils.plot_model(model, show_shapes=True)

# ---

model.summary()

# ---

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ---

history = model.fit(x_train, y_train, batch_size=128, epochs=5, validation_data=(x_test, y_test))

# ---

pd.DataFrame(history.history).plot(figsize=(10,7))
plt.title("Model Metrics")
plt.show()

# ---

loss, accuracy = model.evaluate(x_test, y_test)
print("Test Accuracy:", accuracy)

# ---

y_pred = model.predict(x_test)

# ---

y_pred

# ---

y_pred = y_pred.flatten()

# ---

y_pred

# ---

y_pred = (y_pred > 0.5).astype(int)

# ---

print(metrics.classification_report(y_test, y_pred))

# ---

cm = metrics.confusion_matrix(y_test, y_pred)
plot_confusion_matrix(cm, class_names=['Negative', 'Positive'])
plt.title("Confusion Matrix")
plt.show()

# ---



# ---


#  Lab 2 (Without TensorFlow Hub Embedding)
# 🔧 Purpose:
# Similar goal as original Lab 2 but does not use a pre-trained embedding.

# Likely uses a simpler text vectorization (e.g., tokenizer, bag-of-words, or custom embedding layer).

# 🧰 Key Differences:
# No tensorflow_hub.

# Probably includes Tokenizer, TextVectorization, or manually created word embeddings.

# Why do this version?
# To learn how to handle raw text without relying on pre-trained models.
