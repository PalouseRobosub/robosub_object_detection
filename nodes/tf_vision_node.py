#!/usr/bin/python
import cv2
import cv_bridge
import mutex
import numpy as np
import rospy
import tensorflow as tf

from robosub_msgs.msg import DetectionImage, Detection
from sensor_msgs.msg import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util


class Node:
    def __init__(self, label_map, graph):
        # Generate the session associated with the current graph.
        self.labels = label_map
        self.graph = graph
        self.session = tf.Session(graph=self.graph)

        self.locker = mutex.mutex()
        self.left_sub = rospy.Subscriber('camera/left/undistorted', Image, self.left_callback)
        self.left_pub = rospy.Publisher('vision/left/intermediate', DetectionImage, queue_size=10)

        self.right_sub = rospy.Subscriber('camera/right/undistorted', Image, self.right_callback)
        self.right_pub = rospy.Publisher('vision/right/intermediate', DetectionImage, queue_size=10)

        self.bridge = cv_bridge.CvBridge()
        self.min_score = 0.10

        self.image_tensor = self.graph.get_tensor_by_name('image_tensor:0')
        self.d_boxes = self.graph.get_tensor_by_name('detection_boxes:0')
        self.d_scores = self.graph.get_tensor_by_name('detection_scores:0')
        self.d_classes = self.graph.get_tensor_by_name('detection_classes:0')
        self.num_d = self.graph.get_tensor_by_name('num_detections:0')

        rospy.loginfo('=== Node spin up successful ===')


    def right_callback(self, img_msg):
        self.locker.lock(self.callback, (img_msg, self.right_pub))
        self.locker.unlock()


    def left_callback(self, img_msg):
        self.locker.lock(self.callback, (img_msg, self.left_pub))
        self.locker.unlock()


    def callback(self, args):
        img_msg = args[0]
        publisher = args[1]
        callback_start = rospy.get_time()
        img = self.bridge.imgmsg_to_cv2(img_msg, 'bgr8')
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform an inference on the image.
        with self.graph.as_default():
            start_t = rospy.get_time()
            (boxes, scores, classes, num) = self.session.run(
                    [self.d_boxes, self.d_scores, self.d_classes, self.num_d],
                    feed_dict={
                        self.image_tensor: np.expand_dims(img_rgb, axis=0)
                    })

            rospy.loginfo(' === Inference took {} seconds === '.format(rospy.get_time() - start_t))

        # Process the array outputs from the inference into normal lists and
        # numbers.
        num_detections = int(num[0])
        classes = [int(x) for x in classes[0]]
        boxes = boxes[0]
        scores = scores[0]

        detection_msg = DetectionImage()

        kept_indices = []
        for i in range(0, num_detections):
            probability = scores[i]
            if probability >= self.min_score:
                box = boxes[i]
                classname = classes[i]

                detection = Detection()
                detection.height = box[2] - box[0]
                detection.width = box[3] - box[1]
                detection.x = (box[3] + box[1]) / 2
                detection.y = (box[2] + box[0]) / 2
                detection.label = self.labels[classname]['name']
                detection.probability = probability

                detection_msg.detections.append(detection)

        detection_msg.header.stamp = rospy.Time.now()
        detection_msg.image = img_msg
        publisher.publish(detection_msg)


    def close(self):
        self.session.close()

        self.left_pub.unregister()
        self.left_sub.unregister()

        self.right_pub.unregister()
        self.right_sub.unregister()


if __name__ == '__main__':
    rospy.init_node('tensorflow_detector')

    label_file = rospy.get_param('~labels')
    model_file = rospy.get_param('~model')

    rospy.loginfo('=== Parameters loaded ===')

    # Load the label map into TF to convert indices into labels.
    label_map = label_map_util.load_labelmap(label_file)
    categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    # Load the model architecture into TF.
    detection_graph = tf.Graph()

    with detection_graph.as_default():
        graph_def = tf.GraphDef()

        with tf.gfile.GFile(model_file, 'rb') as f:
            serialized_graph = f.read()

        graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(graph_def, name='')

    rospy.loginfo('=== Graph created ===')

    # Construct the node and begin the pub/sub loops.
    node = Node(category_index, detection_graph)

    rospy.spin()

    # Release the TF session.
    node.close()
