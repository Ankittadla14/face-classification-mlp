Face Recognition using MLPs (Baseline vs DeepNN)

This repository explores face recognition using Multi-Layer Perceptrons (MLPs) implemented in PyTorch.
It compares a baseline shallow MLP with a deeper network (BN + Dropout) to evaluate improvements in accuracy and generalization.

📂 Repository Structure
.
├── facenn_baseline.ipynb   # Baseline MLP (simpler architecture)
├── facenn_deepnn.ipynb     # Deeper MLP with BN + Dropout (recommended)
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignore unnecessary files
└── README_face.md          # Project overview (this file)

📊 Dataset

Dataset used: face_all.pickle (features + labels)

Size: >25MB (not committed due to GitHub limits).

Download it from Google Drive: 📥 Download face_all.pickle

After downloading, place it in the project root folder (same directory as the notebooks):

project-root/
    facenn_baseline.ipynb
    facenn_deepnn.ipynb
    face_all.pickle   <-- place here

⚙️ Installation

Create a virtual environment and install dependencies:

pip install -r requirements.txt

🚀 Running the Notebooks

Run facenn_baseline.ipynb → trains a 2-layer MLP

Run facenn_deepnn.ipynb → trains a deeper MLP with BN + Dropout

Both notebooks load face_all.pickle automatically if present in the same folder.

📝 Results

Baseline MLP: ~80–85% accuracy

Deep MLP (BN + Dropout): ~88–90% accuracy

📌 Notes

.pt model files (saved weights) are not included. Train the notebooks to generate them.

The dataset (face_all.pickle) must be manually downloaded before running.
