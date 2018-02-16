#!/usr/bin/python
import argparse
import cv2
import cv_bridge
import numpy as np
import tensorflow as tf

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer labels in an image.')

    parser.add_argument('img', type=str, help='The image to infer.')
    parser.add_argument('model', type=str, help='The frozen inference graph to use')
    parser.add_argument('labels', type=str, help='The label map to use.')

    args = parser.parse_args()

    # Load the label map into TF to convert indices into labels.
    label_map = label_map_util.load_labelmap(args.labels)
    categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the model architecture into TF.
    graph = tf.Graph()

    with graph.as_default():
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(args.model, 'rb') as f:
            serialized_graph = f.read()

        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    d_boxes = graph.get_tensor_by_name('detection_boxes:0')
    d_scores = graph.get_tensor_by_name('detection_scores:0')
    d_classes = graph.get_tensor_by_name('detection_classes:0')
    num_d = graph.get_tensor_by_name('num_detections:0')

    # Read the image into memory.
    img = cv2.imread(args.img)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform an inference on the image.
    with graph.as_default():
        with tf.Session() as sess:
            (boxes, scores, classes, num) = sess.run(
                [d_boxes, d_scores, d_classes, num_d],
                feed_dict={
                    image_tensor: np.expand_dims(img_rgb, axis=0)
                })

    vis_util.visualize_boxes_and_labels_on_image_array(
            img,
            boxes[0],
            classes[0].astype(int),
            scores[0],
            category_index,
            use_normalized_coordinates=True,
            line_thickness=3)

    cv2.imshow('Detected Image', img)
    cv2.waitKey()
