# Bigram Language Model for Name Generation

A **character-level bigram language model** implemented in Python using **PyTorch**, designed to generate names by learning the probabilities of consecutive character pairs. This project also evaluates the model performance using **negative log-likelihood** on a test dataset.


## Table of Contents

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Dataset](#dataset)  
4. [Installation](#installation)  
5. [Usage](#usage)  
6. [Model Details](#model-details)  
7. [Evaluation](#evaluation)  
8. [Technologies & Tools](#technologies--tools)  
9. [Contributing](#contributing)  
10. [License](#license)  


## Project Overview

This project focuses on building a **simple probabilistic model** that learns character relationships in words (names in this case) and generates new sequences. Unlike neural networks, bigram models rely purely on **co-occurrence statistics** of adjacent characters. The model can:  

- Generate realistic-looking names by sampling characters based on learned probabilities.  
- Evaluate how well it models the test data using **negative log-likelihood (NLL)**.  

This project provides a foundational understanding of **language modeling**, **sequence generation**, and **probabilistic NLP**.


## Features

- Character-level **encoding and decoding**  
- **Bigram probability matrix** construction and normalization  
- Random **sequence generation** from learned probabilities  
- **Evaluation** on test data using negative log-likelihood  
- Handles **start `<S>` and end `<E>` tokens** to mark sequence boundaries  


## Dataset

- The model trains on a **list of names** stored in `names.txt`.  
- Data is split into **training** (default 90%) and **testing** (10%).  
- Each word is prepended with `<S>` and appended with `<E>` for proper sequence modeling.


## Installation

1. **Clone the repository**
  
```bash
git clone https://github.com/your-username/bigram-name-generator.git
cd bigram-name-generator
```

2. **Install dependencies**

```bash
pip install torch
```

3. **Add your dataset**

- Place names.txt in the root directory.
- Ensure each name is separated by a newline or space.

## Usage

```bash
import torch
from bigram_model import BigramModel

file_path = "names.txt"
train_ratio = 0.9

# Initialize and train the model
b = BigramModel(file_path, train_ratio)

# Generate negative log-likelihood for evaluation
nll = b.model_evaluation()
print(f"Negative Log Likelihood = {nll:.4f}")

```
- Generated names are printed during initialization.
- The model_evaluation() method returns the negative log-likelihood of the test data.

## Model Details

**1. Encoding & Decoding**

- Converts characters to integer indices (stoi) and back (itos).

**2. Count Dictionary**

- Counts occurrences of all character pairs in the training data.

**3. Normalized Bigram Probability Matrix**

- Converts counts into probabilities for sampling next characters.

**4. Sequence Generation**

- Uses multinomial sampling to generate new sequences based on learned bigram probabilities.

## Technologies & Tools

- **Programming Language:** Python
- **Library:** PyTorch
- **Concepts:** Natural Language Processing (NLP), Probabilistic Models, Sequence Modeling

## Contributing

Contributions are welcome! You can:
- Improve the model (e.g., trigram or neural network extensions)
- Add new datasets for testing
- Enhance sequence generation or evaluation methods

Please open an issue or submit a pull request.

## Example Output
Generated names (sample)

```bash
sar
efaylyla
jeei
lee
giyssie
corar
citay
veliaurl
kican
on
```

Negative Log-Likelihood on Test Data

```bash
Negative Log Likelihood = 5.1134
```

