import torch

DATA_PATH = "../../data"
TRAIN_IMAGE_PATH = "../../data/train/train_images"
TRAIN_QUESTION_PATH = "../../data/train/train_questions.json"
TRAIN_ANNOTATION_PATH = "../../data/train/train_annotations.json"
VAL_IMAGE_PATH = "../../data/val/val_images"
VAL_QUESTION_PATH = "../../data/val/val_questions.json"
VAL_ANNOTATION_PATH = "../../data/val/val_annotations.json"
RESIZE_PATH = "../../data/resize_image"
TOP = 1000
IMAGE_SIZE = 224

MODEL = "CLIP"
DATA_DIR = "../../data"
MODEL_DIR = "../../ckpt"
MAXL = 30
NUM_WORKER = 24
BATCH_SIZE = 150
QUESTION_FILE = "question_vocabs.txt"
ANNOTATION_FILE = "annotation_vocabs.txt"
EPOCH = 10
INPUT_SIZE = 1024
EMBED = 300
HIDDEN_SIZE = 512
NUM_HIDDEN = 2
NUM_ATTENTION = 2
NUM_CHANNEL = 512
LR = 0.001
STEP_SIZE = 10
GAMMA = 0.1
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
