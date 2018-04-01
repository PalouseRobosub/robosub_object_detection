# PalouseRobosub Object Detection
This repository contains the Palouse Robosub-specific ROS nodes for employing deep learning neural nets for vision inferences.

## To run with specific network run
Mostly networks are stored at /`repositories/robosub_object_detection/trained_models`. To run specific network, run
    `roslaunch robosub_object_detection tensorflow.launch model:={name_of_network}`

## To pull trained networks
All networks should be available on the robosub server (robosub.eecs.wsu.edu)
    at /data/vision/trained_models. To pull down new models, use
    `tools/pull_trained_models.sh`.

## Requirements
The current implementation relies on TensorFlow 1.5 - this can be installed
through Pip. You will also need atleast version 2.6 of the proto-compiler
(`sudo apt-get install proto-compiler`). Please verify the proto-compiler
version before continuing.

Next, run the `tools/initial_setup.sh` utility to configure your repository.

