# Character-Level Text Generation with LSTMs

This project implements a **character-level text generation model** using a stacked LSTM architecture.  
The model learns from a sample of Shakespeare text and generates new text one character at a time, mimicking the style and structure of the original writing.

The project demonstrates the core ideas of sequence modeling:
- Encoding text at the character level
- Using embeddings to represent characters
- Applying stacked LSTM layers to learn sequential patterns
- Sampling next characters using temperature-based generation

---

## Project Overview

The goal of this project is to train an LSTM to predict the **next character** given the previous 40 characters.  
Once trained, the model can generate new Shakespeare-like text.

This is a minimal and educational example of:
- Recurrent Neural Networks (RNNs)
- Long Short-Term Memory networks (LSTMs)
- Sequence-to-sequence learning
- Temperature-controlled text generation

---

## How It Works

### **1. Character Tokenization**
The text is processed at the character level:
- All unique characters are extracted
- Each character is mapped to an integer (char → index)
- The entire text is encoded into a sequence of integers

### **2. Sliding Window Dataset**
Using a window of 40 characters:
- **X** contains sequences of 40 characters
- **y** contains the next character after each sequence

This teaches the model:  
> “Given these 40 characters, what comes next?”

### **3. LSTM Model Architecture**
The model uses:
- An **Embedding layer** (character → dense vector)
- **Two stacked LSTM layers**  
  - The first learns low-level patterns  
  - The second learns deeper structural patterns  
- A **Dense softmax output layer** to predict the next character

### **4. Training**
The model is trained using:
- Loss: `categorical_crossentropy`
- Optimizer: `adam`
- Recommended epochs: 3–10

### **5. Temperature-Based Generation**
The model generates new text by sampling characters with a configurable **temperature**:
- Low temperature → predictable, repetitive text  
- Medium temperature → realistic, balanced output  
- High temperature → creative, chaotic text  

