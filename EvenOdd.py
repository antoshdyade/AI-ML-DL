import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Use last digit as input
X = np.array([i % 10 for i in range(1000)])  # Last digit of number
y = np.array([i % 2 for i in range(1000)])   # 0 = even, 1 = odd

# Normalize input
X = X / 10.0

# Build the model
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X, y, epochs=20, batch_size=32)

# Predict function
def predict_even_odd(n):
    last_digit = (n % 10) / 10.0
    pred = model.predict(np.array([[last_digit]]))[0][0]
    result = "Even" if pred < 0.5 else "Odd"
    print(f"{n} is predicted as: {result} (Confidence: {pred:.2f})")

# Test
predict_even_odd(33)
predict_even_odd(42)
predict_even_odd(77)
predict_even_odd(100)
