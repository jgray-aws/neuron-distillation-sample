#!/bin/bash

wget https://raw.githubusercontent.com/jimburtoft/Neuron-SDK-Detector/refs/heads/main/update_to_sdk_2_26_0.sh
chmod +x update_to_sdk_2_26_0.sh
./update_to_sdk_2_26_0

pip install git+https://github.com/huggingface/optimum-neuron.git@5faa93c1686f53568df22c1f9259d9fc97fdab27
pip install torch==2.7.1 torch-neuronx==2.7.0.2.10.13553+1e4dd6ca torch-xla==2.7.0

