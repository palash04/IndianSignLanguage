# Hyper-parameters
NUMBER_OF_VIDEOS_PER_GESTURE = 50  # 50
NUMBER_OF_TEST_VIDEOS_PER_GESTURE = 10
FRAMES_PER_VIDEO = 20  # 20
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
EPOCHS = 20
split_data = False  # true: split train val from train data; false: use provided val data directly
BATCH_SIZE = 16
NUM_WORKERS = 4
LEARNING_RATE = 1e-5
OPTIMIZER_NAME = 'adam'
BIDIRECTIONAL = True  # matters only if model name is bilstm
TRAIN_VAL_SPLIT = 0.75
HIDDEN_SIZE = 128
MODEL_NAME = 'transformer'  # transformer or bilstm or lstm
