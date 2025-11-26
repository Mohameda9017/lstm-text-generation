"""
Character-Level Text Generation using LSTMs
-------------------------------------------

This project trains a character-level LSTM model to generate 
Shakespeare-style text. The model learns to predict the next 
character based on a sliding window of 40 characters.

Steps:
1. Prepare and tokenize text at the character level
2. Build a stacked LSTM sequence model
3. Train the model to predict next characters
4. Generate new text using temperature-based sampling
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
import numpy as np

# -----------------------------
# 1. Load and prepare text data
# -----------------------------

text = """To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die, to sleep—
No more—and by a sleep to say we end
The heart-ache and the thousand natural shocks
That flesh is heir to. ’Tis a consummation
Devoutly to be wish’d. To die, to sleep—
To sleep, perchance to dream—ay, there’s the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause."""

# Get all unique characters in the text
unique_characters = sorted(list(set(text)))

# Map characters ↔ integers
char_to_idx = {c: i for i, c in enumerate(unique_characters)}
idx_to_char = {i: c for c, i in char_to_idx.items()}

# Encode text as integers
encoded_text = [char_to_idx[c] for c in text]

# Create input sequences and labels using sliding windows
sequence_length = 40
X, y = [], []

for i in range(len(encoded_text) - sequence_length):
    X.append(encoded_text[i : i + sequence_length])   # 40-character input
    y.append(encoded_text[i + sequence_length])       # Next character

X = np.array(X)
y = tf.keras.utils.to_categorical(y, num_classes=len(unique_characters))


# -----------------------------
# 2. Build the LSTM model
# -----------------------------

vocab_size = len(unique_characters)
embedding_dim = 256
lstm_units = 512

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=sequence_length),
    LSTM(lstm_units, return_sequences=True),
    LSTM(lstm_units),
    Dense(vocab_size, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)


# ---------------------------------------
# 3. Text generation with temperature
# ---------------------------------------

def generate_text(seed_text, length=200, temperature=1.0):
    """
    Generates text by repeatedly predicting the next character.
    Temperature controls randomness:
      - low temperature → safer predictions
      - high temperature → more creative/chaotic
    """
    result = seed_text.lower()

    for _ in range(length):
        seed = result[-sequence_length:]
        seed_ids = [char_to_idx.get(c, 0) for c in seed]

        # Pad if seed is shorter than sequence_length
        if len(seed_ids) < sequence_length:
            seed_ids = [0] * (sequence_length - len(seed_ids)) + seed_ids

        preds = model.predict(np.array([seed_ids]), verbose=0)[0]

        # Apply temperature scaling
        preds = np.log(preds + 1e-8) / temperature
        preds = np.exp(preds) / np.sum(np.exp(preds))

        # Sample next character
        next_id = np.random.choice(len(preds), p=preds)
        result += idx_to_char[next_id]

    return result


# -----------------------------
# 4. Train and generate output
# -----------------------------

model.fit(X, y, epochs=5)
print(generate_text("to be", length=200, temperature=0.8))

