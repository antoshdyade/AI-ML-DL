import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. Prepare 2D transformed data
X = np.array([[x, x if x % 2 == 0 else -x] for x in range(1000)])
y = np.array([x % 2 for x in range(1000)])  # 0 = even, 1 = odd

# 2. Normalize (optional)
X = X / 1000.0

# 3. Build model
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(12, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train
model.fit(X, y, epochs=50, batch_size=32, verbose=2)

# 5. Prediction helper
def predict_even_odd(n):
    x_input = np.array([[n, n if n % 2 == 0 else -n]]) / 1000.0
    pred = model.predict(x_input)[0][0]
    print(f"{n} → {'Even' if pred < 0.5 else 'Odd'} (Confidence: {pred:.2f})")

# 6. Test
predict_even_odd(33)
predict_even_odd(42)
predict_even_odd(77)
