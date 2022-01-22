import shutil
import os
from ISL_params import *
from ISL_utils import *

# Create folders for each gestures, so as to store keypoints data
def make_dir(dir, train=1):
    try:
        shutil.rmtree(dir)
    except:
        pass
    os.makedirs(dir, exist_ok=True)
    for gesture in gestures:
        path = os.path.join(dir, str(gesture))
        os.makedirs(path, exist_ok=True)
    if train:
        for gesture in gestures:
            for video_num in range(1, NUMBER_OF_VIDEOS_PER_GESTURE + 1):
                path = os.path.join(dir, gesture, str(video_num))
                os.makedirs(path, exist_ok=True)
    else:
        for gesture in gestures:
            for video_num in range(1, NUMBER_OF_TEST_VIDEOS_PER_GESTURE + 1):
                path = os.path.join(dir, gesture, str(video_num))
                os.makedirs(path, exist_ok=True)

# make_dir(train_dir, train=1)
# make_dir(val_dir, train=0)
# make_dir(test_dir, train=0)
print('Done!')
