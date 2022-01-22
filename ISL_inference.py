import torch
import torch.nn.functional as F

from ISL_utils import *
from ISL_params import *
from ISL_BiLSTM import BiLSTM
from ISL_Transformer import TransformerEncoder


def make_inference(model):
    model.eval()
    sequence = []
    sentence = []
    threshold = 0.65

    last_count = 0
    last_class = 'none'

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=MIN_DETECTION_CONFIDENCE,
                              min_tracking_confidence=MIN_TRACKING_CONFIDENCE) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)

            # Draw landmarks
            # draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-FRAMES_PER_VIDEO:]

            if len(sequence) == FRAMES_PER_VIDEO:
                x = np.array(sequence)
                x = torch.tensor(x).double()
                x = x.unsqueeze(0)

                outputs = model(x)  # (1, num_classes)

                preds = F.softmax(outputs, dim=1)
                pred_prob, pred_class_idx = preds.max(dim=1)

                pred_class = gestures[pred_class_idx.item()]
                # print(pred_prob.item(),pred_class)
                if pred_class in ['next']:
                    sentence = []   # remove this

                if pred_prob > threshold:

                    if last_count == 0:
                        last_count += 1
                        last_class = pred_class
                    else:
                        if last_class == pred_class:
                            last_count += 1
                        else:
                            last_class = pred_class
                            last_count = 1

                    if last_count >= 2:
                        if len(sentence) > 0:
                            last_pred = sentence[-1]
                            if pred_class != last_pred:
                                sentence.append(pred_class)
                        else:
                            sentence.append(pred_class)
                    print(pred_prob.item(), pred_class)

                if len(sentence) > 5:
                    sentence = sentence[-10:]

            if len(sentence) > 0:
                cv2.putText(image, ' '.join(sentence), (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            cv2.LINE_AA)

            cv2.imshow('Predictions', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


def main():
    num_classes = len(labels_to_gestures)
    model_name = 'transformer'
    if model_name == 'bilstm':
        model = BiLSTM(num_classes, hidden_size=HIDDEN_SIZE)  # switch between bilstm and transformer
    else:
        model = TransformerEncoder(num_classes, device='cpu', hidden_size=HIDDEN_SIZE)

    checkpoint = torch.load(f"best_model_{model_name}.pth.tar", map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.double()
    make_inference(model)


if __name__ == "__main__":
    main()
