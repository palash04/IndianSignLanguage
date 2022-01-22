import os
from ISL_params import *
from ISL_utils import *


def get_sequences_labels(ROOT_DIR):
    sequences, labels = [], []
    for gesture_w in gestures:
        for video_num in np.array(os.listdir(os.path.join(ROOT_DIR, gesture_w))).astype(int):
            window = []
            for frame_num in range(FRAMES_PER_VIDEO):
                npypath = os.path.join(ROOT_DIR, gesture_w, str(video_num), f"{frame_num}.npy")
                res = np.load(npypath)  # (1662,)
                window.append(res)
            sequences.append(window)
            labels.append(gestures_to_labels[gesture_w])

    sequences = np.array(sequences)
    labels = np.array(labels)
    return sequences, labels


def main():
    sequences, labels = get_sequences_labels(ROOT_DIR=train_dir)
    print(sequences.shape)
    print(labels.shape)

    print(np.min(sequences))
    print(np.max(sequences))


if __name__ == "__main__":
    main()
