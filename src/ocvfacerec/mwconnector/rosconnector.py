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
#   * Redistributions of source code must retain the above copyright
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
from Queue import Queue

# ROS IMPORTS
import rospy
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from people_msgs.msg import People
from people_msgs.msg import Person
from geometry_msgs.msg import Point

# OCVF Imports
from ocvfacerec.mwconnector.abtractconnector import MiddlewareConnector


class ROSConnector(MiddlewareConnector):
    # TODO Implement
    def __init__(self):
        self.bridge = CvBridge()
        self.restart_subscriber = None
        self.restart_publisher = None
        self.image_subscriber = None
        self.last_image = None
        self.last_train = None

    def add_last_image(self, image_data):
        try:
            self.last_image.get(False)
        except Exception, e:
            print e
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_data.data, "bgr8")
            self.last_image.put(cv_image, False)
        except Exception, e:
            print e

    def add_last_train(self, msg):
            try:
                self.last_train.get(False)
            except Exception, e:
                pass
            self.last_train.put(str(msg.data), False)

    def activate(self, image_source, retrain_source, restart_target):
        self.image_subscriber   = rospy.Subscriber(image_source, Image, self.add_last_image, queue_size=1)
        self.last_image = Queue(1)

        self.restart_subscriber = rospy.Subscriber(retrain_source, String, self.add_last_train, queue_size=1)
        self.last_train = Queue(1)

        self.restart_publisher  = rospy.Publisher(restart_target, String, queue_size=1)
        rospy.init_node('ros_connector_trainer', anonymous=False)

    def deactivate(self):
        self.restart_subscriber.unregister()
        self.image_subscriber.unregister()

    def restart_classifier(self):
        # Send a short "restart" event to the recognizer
        msg = "restart"
        self.restart_publisher.publish(msg)

    def wait_for_start_training(self):
        return self.last_train.get(True, timeout=1)

    def get_image(self):
        return self.last_image.get(True, timeout=10)