# Face Classification — Baseline MLP vs Deep MLP (PyTorch)

This repo compares two fully‑connected neural networks on a face/non‑face dataset (`face_all.pickle`):
- **`facenn_baseline.py`** — simpler MLP baseline.
- **`facenn_deepnn.py`** — deeper MLP with BatchNorm/Dropout and a stronger training recipe.

## Repository Structure
```
.
├── facenn_baseline.py       # Baseline MLP (simpler architecture)
├── facenn_deepnn.py         # Deeper MLP with BN + Dropout (recommended)
├── data/
│   └── face_all.pickle      # Not tracked; place the dataset here
├── README.md
├── requirements.txt
└── .gitignore
```
> `face_all.pickle` is **not** committed. Place it at `./data/face_all.pickle`.

## 📂 Dataset
The dataset `face_all.pickle` is required to run the experiments. Due to size limits, it is not in this repo.

**Download here:** [Google Drive – face_all.pickle](https://drive.google.com/file/d/13_uo4CoAWUsX4LvGAg3tDECRYEZ3TSIn/view?usp=sharing)

After downloading, put it in:
```
./data/face_all.pickle
```

**Optional (auto‑download with gdown):**
```bash
pip install gdown
gdown --fuzzy "https://drive.google.com/file/d/13_uo4CoAWUsX4LvGAg3tDECRYEZ3TSIn/view?usp=sharing" -O data/face_all.pickle
```

## Quickstart
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### Run baseline
```bash
python facenn_baseline.py
```

### Run deep MLP (recommended)
```bash
python facenn_deepnn.py
```

## Methods (high level)
- **Baseline:** 2376 → 256 → 256 → 2, ReLU, SGD.
- **Deep MLP:** 2376 → 512 → 256 → 64 → 2, ReLU, BatchNorm (after first two layers), Dropout (0.30/0.40), AdamW, label smoothing, LR scheduler, early stopping (optional in code comments).

## Results (fill after running)
| Model | Test Accuracy | Notes |
|---|---|---|
| Baseline MLP (`facenn_baseline.py`) | XX% | Simpler network |
| Deep MLP (`facenn_deepnn.py`) | YY% | BN + Dropout + AdamW |

## Notes
- If you observe overfitting, increase dropout (e.g., 0.35/0.45). If underfitting, reduce by 0.05.
- Batch size ≥ 64 is recommended for stable BatchNorm statistics.

## License
Add an MIT `LICENSE` file if you plan to open‑source this work.
