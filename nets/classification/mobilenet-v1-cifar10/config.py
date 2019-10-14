# dataset related params
DATASET_NAME = 'cifar-10'
DATASET_DIR = '../../../datasets/cifar-10/raw_data'
DATASET_SIZE = 60000
DATASET_SPLIT = [0.7, 0.15, 0.15] # TRAIN, VAL, TEST 
NUM_CLASSES  = 10
BATCH_SIZE   = 32

TRAIN_SIZE = int(DATASET_SIZE*DATASET_SPLIT[0])
VAL_SIZE   = int(DATASET_SIZE*DATASET_SPLIT[1])
TEST_SIZE  = int(DATASET_SIZE*DATASET_SPLIT[2])
TRAIN_SPLIT = TRAIN_SIZE
VAL_SPLIT   = TRAIN_SIZE+VAL_SIZE
TEST_SPLIT  = DATASET_SIZE

# model related params
IMG_SHAPE = [128, 128, 3]
DEPTH_MULTIPLIER = 0.25
DROPOUT_PROB = 0.2
