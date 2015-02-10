# Copyright (c) 2015.
# Philipp Wagner <bytefish[at]gmx[dot]de> and
# Florian Lier <flier[at]techfak.uni-bielefeld.de> and
# Norman Koester <nkoester[at]techfak.uni-bielefeld.de>
#
#
# Released to public domain under terms of the BSD Simplified license.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   * Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#   * Neither the name of the organization nor the names of its contributors
#     may be used to endorse or promote products derived from this software
#     without specific prior written permission.
#
#    See <http://www.opensource.org/licenses/bsd-license>

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# STD Imports
import os
import cv2
import sys
import time
import glob
from Queue import Queue

# ROS Imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class StaticImageSender():
    def __init__(self):
        self.image_q = None
        self.bridge = CvBridge()
	self.images = []

    def glob_images(self, image_path):
        img_list = glob.glob(image_path)
        if len(img_list) <= 0:
            print '>> No Images Found in ' + image_path
            return
        self.image_q = Queue(len(img_list))
        for img in img_list:
            img = cv2.imread(img, 1)
            self.image_q.put(img)
	    self.images.append(img)

    def sender(self):
        pub = rospy.Publisher('ocvfacerec/static_image', Image, queue_size=4)
        rospy.init_node('ocvf_static_image_sender', anonymous=False)
	c = 0
	rate = rospy.Rate(0.5) # Every Two Seconds
    	while not rospy.is_shutdown():
                pub.publish(self.bridge.cv2_to_imgmsg(self.images[c], "bgr8"))
                time.sleep(2)
		print ">> Image " + str(c)
		c += 1
		if c > 3:
			c = 0

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print ">> Please provide a path to images"
        sys.exit(1)
    try:
        sim = StaticImageSender()
        sim.glob_images(str(sys.argv[1] + '/*'))
        sim.sender()
    except rospy.ROSInterruptException:
        pass
