import torch
from classifier.config import NUM_CLASSES

# ----------------------------
# Config
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMB_DIM = 100
EMB_DIM_OHE = 4

# MAX_LEN = 10938
MAX_LEN = 12000
D_MODEL = 128   # D_MODEL = 256
NHEAD = 4       # NHEAD = 8
FF_DIM = 512    # FF_DIM = 1024
NUM_LAYERS = 2  # NUM_LAYERS = 3
CHUNK_SIZE = 512
DROPOUT = 0.1

BATCH_SIZE = 32  # 1
EPOCHS = 100     # 100
LR = 3e-4

