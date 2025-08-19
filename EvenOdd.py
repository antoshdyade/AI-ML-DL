import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1. Prepare training data (numbers 0 to 999)
X = np.array(range(1000))            # Input numbers
y = X % 2                            # Labels: 0 for even, 1 for odd

# 2. Build the model
model = keras.Sequential([
    layers.Input(shape=(1,)),        # One input number
    layers.Dense(10, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Output: probability (0 to 1)
])

# 3. Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 4. Train the model
model.fit(X, y, epochs=10, batch_size=32)

# 5. Test the model
def predict_even_odd(n):
    pred = model.predict(np.array([[n]]))[0][0]
    result = "Even" if pred < 0.5 else "Odd"
    print(f"{n} is predicted as: {result} (Confidence: {pred:.2f})")

# Try with custom numbers
predict_even_odd(42)
predict_even_odd(77)
predict_even_odd(1234)
