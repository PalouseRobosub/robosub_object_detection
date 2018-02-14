# rs_tensorflow
This repository contains the Palouse Robosub-specific ROS nodes for employing deep learning neural nets for vision inferences.

== To pull trained networks ==
All networks should be available on the robosub server (robosub.eecs.wsu.edu)
    at /data/vision/trained_models. To pull down new models, use
    `tools/pull_trained_models.sh`.

== Requirements ==
The current implementation relies on TensorFlow 1.5 - this can be installed
through Pip.
