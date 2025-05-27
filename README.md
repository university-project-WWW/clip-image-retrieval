# 🔍 CLIP-Based Image Retrieval

## 📌 Overview

This project implements a deep learning-based image retrieval system that returns visually similar images from a gallery, given a query image. It is based on a modified CLIP ViT-L/14 backbone and uses cosine similarity between learned embeddings for retrieval.

---

## 🧠 Model Architecture

The model is based on OpenCLIP ViT-L/14 and is customized as follows:

- **Backbone**: [OpenCLIP ViT-L/14](https://github.com/mlfoundations/open_clip)
- **Grid Size**: 16×16 patches (14×14 patch size)
- **Pooling**: Generalized Mean Pooling (GeM)
- **Projection Head**:
  - `Linear(1024 → 512)` without bias
  - `BatchNorm1d`
  - `Dropout(p=0.1)`
- **Normalization**: Final L2 normalization of the embedding
- **Embedding Dimension**: 512

This architecture enables compact and discriminative image embeddings suitable for nearest-neighbor search.

---

## 🛠️ Inference Pipeline

1. **Extract embeddings** for both query and gallery images using the model.
2. **Compute cosine similarity** between query and gallery features.
3. **Return top-K (e.g., K=10) most similar images** for each query.
4. **Save the output** in a JSON file formatted as:

```json
[
  {
    "filename": "query_001.jpg",
    "samples": [
      "gallery_023.jpg",
      "gallery_074.jpg",
    ]
  }
]
```

## 🗃️ Dataset

The model was trained and evaluated on the following datasets:

- Cifar100
- Food101

## 🚀 Installation

```bash
git clone https://github.com/stanghee/image-retrieval-model.git
cd image-retrieval-model
pip install -r requirements.txt
```

## 👥 Contributors

- [@stanghee](https://github.com/stanghee) 
- [@lorenzoattolico](https://github.com/lorenzoattolico) 
- [@MolteniF](https://github.com/MolteniF)

## 📄 License

This project is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).
[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
