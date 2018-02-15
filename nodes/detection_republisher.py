#!/usr/bin/python
import cv_bridge
import rospy
import tensorflow as tf
import numpy as np

# Specify the Matplotlib rendering agent to use image backends instead of the
# Xserver for remote operation.
import matplotlib
matplotlib.use('Agg')

from object_detection.utils import visualization_utils as vis_util
from robosub_msgs.msg import DetectionImage, DetectionArray
from sensor_msgs.msg import Image

# Malisiewicz et al. taken from
# https://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return pick


class Node:

    def __init__(self, camera):
        self.bridge = cv_bridge.CvBridge()
        self.sub = rospy.Subscriber('vision/{}/intermediate'.format(camera), DetectionImage, self.callback)
        self.pretty_pub = rospy.Publisher('pretty/vision/{}'.format(camera), Image, queue_size=1)
        self.pub = rospy.Publisher('vision/{}'.format(camera), DetectionArray, queue_size=10)


    def callback(self, det_img_msg):
        t = rospy.get_time()
        img = self.bridge.imgmsg_to_cv2(det_img_msg.image, 'bgr8')

        (rows, columns, channels) = img.shape

        # Create a list of boxes and scores.
        boxes = []
        scores = []
        classes = []
        label_map = dict()
        for det in det_img_msg.detections:
            ymin = int((det.y - det.height / 2) * rows)
            ymax = int((det.y + det.height / 2) * rows)
            xmin = int((det.x - det.width / 2) * columns)
            xmax = int((det.x + det.width / 2) * columns)
            scores.append(det.probability)
            classes.append(det.label_id)
            label_map[det.label_id] = {'name': det.label}
            boxes.append(np.array([ymin, xmin, ymax, xmax]))

        # Remove overlapping detections with non maximum suppression.
        # TODO: Should this be run on each class?
        kept_indices = non_max_suppression_fast(np.array(boxes), 0.5)

        boxes = np.array(np.take(boxes, kept_indices, axis=0))
        scores = np.array(np.take(scores, kept_indices, axis=0))
        classes = np.array(np.take(classes, kept_indices, axis=0))

        # Construct the pretty image.
        if len(kept_indices) > 0:
            print boxes
            vis_util.visualize_boxes_and_labels_on_image_array(
                    img,
                    boxes,
                    classes,
                    scores,
                    label_map,
                    use_normalized_coordinates=True,
                    min_score_thresh=0,
                    line_thickness=3)

        # Publish the actual detections in the image
        output_detections = DetectionArray()
        output_detections.header.stamp = rospy.Time.now()
        output_detections.detections = [det_img_msg.detections[i] for i in kept_indices]

        self.pub.publish(output_detections)
        self.pretty_pub.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))
        t_end = rospy.get_time()
        rospy.loginfo('Callback took {}'.format(t_end - t))


if __name__ == '__main__':
    rospy.init_node('detection_republisher')

    camera = rospy.get_param('~camera')

    node = Node(camera)

    rospy.spin()
