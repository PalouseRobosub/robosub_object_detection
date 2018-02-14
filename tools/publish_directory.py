#!/usr/bin/python

import argparse
import cv2
import cv_bridge
import glob
import rospy
import sys

from sensor_msgs.msg import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dir')
    parser.add_argument('--rate', type=float, default=5,
            help='The rate at which to publish images.')
    parser.add_argument('--regex', default='*.png', type=str,
            help='The file regex to match for playback.')
    parser.add_argument('--topic', type=str, default='camera/left/undistorted',
            help='The topic to publish images to.')

    args = parser.parse_args()

    rospy.init_node('image_publisher', anonymous=True)

    pub = rospy.Publisher(args.topic, Image, queue_size=10)

    bridge = cv_bridge.CvBridge()
    r = rospy.Rate(args.rate)

    names = glob.glob('{}/{}'.format(args.dir, args.regex))
    names.sort()
    for fname in names:
        rospy.loginfo('Publishing {}'.format(fname))
        img = cv2.imread(fname)
        pub.publish(bridge.cv2_to_imgmsg(img, 'bgr8'))
        r.sleep()

        if rospy.is_shutdown():
            break
