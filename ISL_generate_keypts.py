import cv2
import os
import numpy as np
from ISL_params import *
from ISL_utils import *


# Generate keypoints
def generate_keypoints_per_gesture(ROOT_DIR, gesture_w, train=1):
    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                              min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as holistic:

        # Create number_of_videos_per_gesture
        if train:
            end_idx = NUMBER_OF_VIDEOS_PER_GESTURE
        else:
            end_idx = NUMBER_OF_TEST_VIDEOS_PER_GESTURE

        for video_num in range(1, end_idx + 1):
            # Create frames_per_video
            for frame_num in range(FRAMES_PER_VIDEO):

                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120, 200),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(gesture_w, video_num),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(gesture_w, video_num),
                                (15, 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(ROOT_DIR, gesture_w, str(video_num), str(frame_num))
                np.save(npy_path, keypoints)

                # Break
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


gesture = 'I'   # rename and run the code for all words in Data directory
# generate_keypoints_per_gesture(ROOT_DIR=train_dir, gesture_w=gesture, train=1)
# generate_keypoints_per_gesture(ROOT_DIR=val_dir, gesture_w=gesture, train=0)
# generate_keypoints_per_gesture(ROOT_DIR=test_dir, gesture_w=gesture, train=0)
