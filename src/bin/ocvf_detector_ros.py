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

# STD IMPORTS
import os
import cv2
import sys
import time
import rospy
import roslib
import signal
from optparse import OptionParser
from thread import start_new_thread


# ROS IMPORTS
from cv_bridge import CvBridge
from std_msgs.msg import String
from std_msgs.msg import Header
from sensor_msgs.msg import Image
from people_msgs.msg import People
from people_msgs.msg import Person
from geometry_msgs.msg import Point

# LOCAL IMPORTS
from ocvfacerec.helper.common import *
from ocvfacerec.trainer.thetrainer import TheTrainer
from ocvfacerec.facerec.serialization import load_model
from ocvfacerec.facedet.detector import CascadedDetector
from ocvfacerec.trainer.thetrainer import ExtendedPredictableModel


class RosPeople:
    def __init__(self):
        self.publisher = rospy.Publisher('ocvfacerec/ros/people', People, queue_size=1)
        rospy.init_node('ocvfacerec_people_publisher', anonymous=True)


def ros_spinning(message="None"):
    print ">> ROS is spinning()"
    rospy.spin()


class Recognizer(object):
    def __init__(self, cascade_filename, run_local, _rp, _wait):
        self.rp = _rp
        self.doRun = True
        self.wait = _wait
        self.restart = False
        self.ros_restart_request = False
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.1)
        if run_local:
            print ">> Error: Run local selected in ROS based Recognizer"
            sys.exit(1)
        else:
            self.bridge = CvBridge()

        def signal_handler(signal, frame):
            print ">> ROS Exiting"
            self.doRun = False

        signal.signal(signal.SIGINT, signal_handler)

    def image_callback(self, ros_data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_data, "bgr8")
        except Exception, ex:
            print ex
            return
        # Resize the frame to half the original size for speeding up the detection process.
        # In ROS we can control the size, so we are sending a 320*240 image by default.
        img = cv2.resize(cv_image, (320, 240), interpolation=cv2.INTER_CUBIC)
        # img = cv2.resize(cv_image, (cv_image.shape[1] / 2, cv_image.shape[0] / 2), interpolation=cv2.INTER_CUBIC)
        # img = cv_image
        imgout = img.copy()
        # Remember the Persons found in current image
        persons = []
        for _i, r in enumerate(self.detector.detect(img)):
            x0, y0, x1, y1 = r
            # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
            face = img[y0:y1, x0:x1]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            # face = cv2.resize(face, self.model.image_size, interpolation=cv2.INTER_CUBIC)
            # prediction = self.model.predict(face)
            # predicted_label = prediction[0]
            # classifier_output = prediction[1]
            # Now let's get the distance from the assuming a 1-Nearest Neighbor.
            # Since it's a 1-Nearest Neighbor only look take the zero-th element:
            # distance = classifier_output['distances'][0]
            # Draw the face area in image:
            cv2.rectangle(imgout, (x0, y0), (x1, y1), (0, 0, 255), 2)
            # Draw the predicted name (folder name...):
            # draw_str(imgout, (x0 - 20, y0 - 40), "Label " + self.model.subject_names[predicted_label])
            # draw_str(imgout, (x0 - 20, y0 - 20), "Feature Distance " + "%1.1f" % distance)
            msg = Person()
            point = Point()
            # Send the center of the person's bounding box
            mid_x = float(x1 + (x1 - x0) * 0.5)
            mid_y = float(y1 + (y1 - y0) * 0.5)
            point.x = mid_x
            point.y = mid_y
            # Z is "mis-used" to represent the size of the bounding box
            point.z = x1 - x0
            msg.position = point
            msg.name = str("unknown")
            msg.reliability = 1.0
            persons.append(msg)
        if len(persons) > 0:
            h = Header()
            h.stamp = rospy.Time.now()
            h.frame_id = '/ros_cam'
            msg = People()
            msg.header = h
            for p in persons:
                msg.people.append(p)
            self.rp.publisher.publish(msg)
        cv2.imshow('OCVFACEREC < ROS STREAM', imgout)
        cv2.waitKey(self.wait)

        try:
            z = self.ros_restart_request
            if z:
                self.restart = True
                self.doRun = False
        except Exception, e:
            pass

    def restart_callback(self, ros_data):
        print ">> Received Restart Request"
        if "restart" in str(ros_data):
            self.ros_restart_request = True

    def run_distributed(self, image_topic):
        print ">> Activating ROS Subscriber"
        image_subscriber = rospy.Subscriber(image_topic, Image, self.image_callback, queue_size=1)
        # print ">> Recognizer is running"
        while self.doRun:
            time.sleep(0.01)
            pass
        # Important: You need to unregister before restarting!
        image_subscriber.unregister()
        print ">> Deactivating ROS Subscriber"


if __name__ == '__main__':
    # model.pkl is a pickled (hopefully trained) PredictableModel, which is
    # used to make predictions. You can learn a model yourself by passing the
    # parameter -d (or --dataset) to learn the model from a given dataset.
    usage = "Usage: %prog [options]"
    # Add options for training, resizing, validation and setting the camera id:
    parser = OptionParser(usage=usage)
    parser.add_option("-s", "--ros-source", action="store", dest="ros_source", help="Grab video from ROS Middleware (default: %default).",
                      default="/usb_cam/image_raw")
    parser.add_option("-c", "--cascade", action="store", dest="cascade_filename",
                      help="Sets the path to the Haar Cascade used for the face detection part [haarcascade_frontalface_alt2.xml].")
    parser.add_option("-w", "--wait", action="store", dest="wait_time", default=20, type="int",
                      help="Amount of time (in ms) to sleep between face identification frames (default: %default).")
    (options, args) = parser.parse_args()
    if options.cascade_filename is None:
        print ">> Error: No cascade file was provded i.e. --cascade=/share/ocvfacerec/haarcascade_frontalface_alt2.xml"
        sys.exit(1)
    if options.ros_source is None:
        print ">> Error: No ROS Topic provided use i.e. --ros-source=/usb_cam/image_raw"
        sys.exit(1)
    if not os.path.exists(options.cascade_filename):
        print ">> Error: No Cascade File found at '%s'." % options.cascade_filename
        sys.exit(1)
    print ">> ROS Camera Input Stream <-- " + str(options.ros_source)
    print ">> Publishing People Info  --> /ocvfacerec/ros/people"
    # Init ROS People Publisher
    rp = RosPeople()
    start_new_thread(ros_spinning, ("None",))
    x = Recognizer(options.cascade_filename, False, rp, options.wait_time)
    x.run_distributed(str(options.ros_source))