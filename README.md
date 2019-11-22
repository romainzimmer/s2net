# Supervised Spiking Network

PyTorch based implementation of Spiking Neural Network layers: 
* SpikingDenseLayer
* SpikingConv1DLayer
* SpikingConv2DLayer
* SpikingConv3DLayer
* ReadoutLayer

including optional lateral and recurrent connections.

## Speech commands data set

To run the speech_commands notebook:
* Download training data at http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz and put them in a folder "data/speech_commands/train"
* Download testing data at http://download.tensorflow.org/data/speech_commands_test_set_v0.01.tar.gz and put them in a folder "data/speech_commands/test"

This work is based on F. Zenke's tutorial on surrogate gradient learning in Spiking Neural Networks (https://github.com/fzenke/spytorch).