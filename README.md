# Word Level Sentence Generation using Deep Learning for Indian Sign Language

This project belongs to my Master's major work guided by professor Dr. Rathna G N.

#### -- Project Timeline: [August 2021] - [June 2022]

## Abstract
Indian Sign Language refers to the language used by the speech impaired (deaf-dumb) population to communicate in the Indian Subcontinent. However, it becomes difficult for the general population to communicate well with the speech-impaired population primarily due to a lack of knowledge of sign language. So to bridge this gap of communication between the general and the speech-impaired people, in this project work, we have tried to recognize the Indian Sign Language word-by-word and thereby generate entire sentences. This project has focused on one-way communication, where the general people can understand speech impaired people with the help of our state-of-the-art models. We have attempted a three-step approach. In the first step, key points extraction of face, pose, left hand, and right hand is done from the video frames. Then in the second step, the extracted key points are given to the deep learning model such as LSTM or Transformer's Encoder to recognize which sign language (i.e., word) the key points belong. And in the final step, the recognized words are appended to the list to generate the desired sentence.

## Built With

- Python v3.7.10 or above
- Numpy, Pandas
- PyTorch v1.9.0 or above
- CUDA 10.2


## Dataset
The Indian Sign Language dataset we created is replicated from the freely available [ISLRTC New Delhi](https://www.youtube.com/channel/UC3AcGIlqVI4nJWCwHgHFXtg/playlists) YouTube page. The playlist found on the youtube channel consists of more than four thousand words and their associated gestures. Since there happened to be only one sample video consisting of sign language for each word, it became a need to generate a video on our own for each word so that we can create more dataset which is ideally suitable for data-hungry deep learning models. 
We made 50 videos per 20 gestures. Each video has 20 frames of sequential data, i.e.,  sign language. Then with the help of the mediapipe package, we extracted keypoints associated with each gesture per frame.
We extracted keypoints from the face, pose, left hand, and right hand for each frame. A total of 1662 keypoints are extracted per frame. That is, we get a tensor of shape 20x1662 per gesture (or per video). We then feed this sequential data with a sequence length of 20 to the sequential model to recognize that particular video.

We also created the test dataset separately, but with different conditions, i.e., we created half the dataset in dim light conditions, and the remaining half was created in normal light conditions.
We made 10 videos for each gestures. A total of 200 videos were made for testing purpose.


## Model Architectures

We trained the dataset on following two different architectures.

1. LSTM-RNN Classifier
2. Transformer-Encoder Classifier


After training on both these architectures it was found that Transformer-Encoder classifier is giving much better results as compared to LSTM-RNN classifier.

The architecture diagram of Transformer-Encoder Classifier is show in the below figure.

![Transformer_Encoder](https://user-images.githubusercontent.com/26361028/155157904-879212db-a8bb-4056-9bbc-efb819b6d15f.png)


## Results obtained on test dataset using Transformer-Encoder architecture

Classification report - </br>
![trans_class_report](https://user-images.githubusercontent.com/26361028/155164438-e0c2623d-cac2-410d-8c3a-99b66fafa46e.png)

Confusion matrix - </br>
![trans_confusion_matrix](https://user-images.githubusercontent.com/26361028/155164720-bf8b6d4d-e74c-41ab-a374-5b748d09a9dc.png)

It can be seen from the above images that the worst predicted classes are "no" and "see", whereas the highly predicted classes are "hat", "idea", "I".

We also got the 70% accuracy on the test dataset using Transformer-Encoder, which is a huge improvement over 58% from LSTM-RNN model.

## Real Time Testing

### Our Demo Video of sentence generation:

<!-- ![Demo Video](https://img.youtube.com/vi/06aPHvjbStE/maxresdefault.jpg) -->

[![Demo Video](https://img.youtube.com/vi/06aPHvjbStE/0.jpg)](https://www.youtube.com/watch?v=06aPHvjbStE)


### Demo Video of our model tested on ISLRTC New Delhi Data
Video showing recognition on three words:

[![Demo Video](https://img.youtube.com/vi/-LS4mHgZ0B8/0.jpg)](https://www.youtube.com/watch?v=-LS4mHgZ0B8)

<!-- https://youtu.be/-LS4mHgZ0B8 -->


## Authors
- Palash Mahendra Kamble - [palash04](https://github.com/palash04/)

