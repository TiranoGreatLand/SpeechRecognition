# SpeechRecognition

These are codes for speech recognition in deep learning.

The principle is encode sequence data to feature vector(or feature sequences) and decode to other sequences.

Currently there are two codes that a bit blend。 I would divide codes of data read and deep learning model in fulture for more convenient programming.

The data is an public Chinese language data base thchs-30.

The part of data preprocessing is referneced from '斗大的熊猫', I do a little change in this pre-processing(you may not find the difference if you are not familiar with it. I remove string of blank because it is too frequency that it would negatively imfluence the model).

The code of model in MFCC_LSTM is referneced from some code of OCR.

The code of model in SPECTROGRAM_CNN is refereneced from other trivial papers and some of my experience.

The two models are all use CTC loss as the loss function for back-propagation to train the network since the input and output are all variable length.

The decode part currently in two models are all one kind of beam search.

The scale of data of thchs-30 to be trained in model is adjustable. I just use a part of data base.

If you use more data or even the whole data base, it needs long-long time to train model.

The more data you used in train, the deeper network you needs to build.
