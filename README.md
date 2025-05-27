# 🔍 CLIP-Based Image Retrieval

## 📌 Overview

This project implements a deep learning-based image retrieval system that returns visually similar images from a gallery, given a query image. It is based on a modified CLIP ViT-L/14 backbone and uses cosine similarity between learned embeddings for retrieval.

---

## 🚀 Key Features

- **Powerful Backbone**: [OpenCLIP ViT-L/14](https://github.com/mlfoundations/open_clip)
- **Advanced Pooling**: Generalized Mean (GeM) pooling to capture discriminative features
- **Optimized Loss Function**: Circle Loss with margin 0.4 and scale 80
- **Mining Strategy**: Adaptive mining of hard patterns after epoch 5
- **Efficient Optimization**: Layer-wise Learning Rate Decay (LLRD) with cosine scheduler
- **Advanced Retrieval Techniques**:
  - Test-Time Augmentation (TTA)
  - Re-ranking with k-reciprocal encoding
  - Query Expansion (QE)

---

## 🧠 Technical Details

### Model Architecture

The system uses a multi-stage architecture:

1. **Backbone**: Vision Transformer (ViT-L/14) with CLIP pre-training provides robust semantic representations
2. **Feature Processing**: GeM pooling aggregates spatial features into a compact representation
3. **Embedding Head**: Linear projection to 512-dimensional embedding space with BatchNorm and Dropout
4. **L2 Normalization**: Final normalization for cosine similarity-based retrieval

### Training Strategy

The training process leverages several advanced techniques:

1. **Loss Function**: Circle Loss with margin 0.4 and scale 80 for effective separation in the embedding space
2. **Mining Strategy**: No mining in early epochs, then MultiSimilarity miner after epoch 5
3. **Batch Sampling**: P-K sampling (P=32, K=4) for balanced batch composition
4. **Optimization**: AdamW with Layer-wise Learning Rate Decay (LLRD)
5. **Learning Rate Schedule**: Cosine decay with 15% warmup period
6. **Regularization**: Weight decay 0.05, Dropout 0.1, and extensive data augmentation

### Advanced Retrieval Pipeline

For optimal retrieval results, the advanced pipeline includes:

1. **Test-Time Augmentation (TTA)**: Averaging embeddings from original and horizontally flipped images
2. **Re-ranking**: Using k-reciprocal encoding to refine initial rankings by considering gallery-to-gallery relationships
3. **Query Expansion**: Averaging the query with its top retrieved results to create an expanded query

---

## 🛠 Installation

1. Clone the repository:

```bash
git clone https://github.com/stanghee/clip-image-retrieval.git
cd clip-image-retrieval
```

2. Install dependencies:

```bash
pip install -r requirements_txt.txt
```

3. Prepare the dataset:

The dataset should have the following structure:
```
your_dataset/
├── training/                     # Training images, with subfolders for each class
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/                         # Test images
    ├── query/                    # Query images (flat structure)
    └── gallery/                  # Gallery images (flat structure)
```

---

## 🗃️ Dataset

The model was trained and evaluated on the following datasets:

- Cifar100
- Food101

---

## 🧠 How It Works

### 1. Training Pipeline (`src/training/pipeline.py`)
- Loads and preprocesses the training data
- Creates model, loss function, and optimizer
- Trains for the specified number of epochs
- Evaluates on validation data
- Saves the best model

### 2. Inference Pipeline (`src/inference/pipeline.py`)
- Loads the trained model
- Extracts embeddings from query and gallery images
- Performs similarity search (with optional advanced techniques)
- Generates the submission file

### 3. Model Architecture (`src/training/model.py`)
- Uses Vision Transformer backbone with CLIP pre-training
- Implements GeM pooling and embedding head
- Provides methods for feature extraction

### 4. Data Handling (`src/data/loader.py`, `src/data/datasets.py`)
- Manages dataset loading and preprocessing
- Implements custom samplers for balanced batches
- Applies data augmentation for improved generalization

---

## 👥 Contributors

- [@stanghee](https://github.com/stanghee) 
- [@lorenzoattolico](https://github.com/lorenzoattolico) 
- [@MolteniF](https://github.com/MolteniF)

---

## 📄 License

This project is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
