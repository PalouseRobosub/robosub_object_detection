# PalouseRobosub Object Detection
This repository contains the Palouse Robosub-specific ROS nodes for employing deep learning neural nets for vision inferences.

## To pull trained networks
All networks should be available on the robosub server (robosub.eecs.wsu.edu)
    at /data/vision/trained_models. To pull down new models, use
    `tools/pull_trained_models.sh`.

## Requirements
The current implementation relies on TensorFlow 1.5 - this can be installed
through Pip.

Please update submodules to properly pull down the TensorFlow Models
repository. This repository contains the object detection API that is used by
the scripts in this repository. You will need to update your python path to
include `[repo_base]/models/research` and `[repo_base]/models/research/slim`.
