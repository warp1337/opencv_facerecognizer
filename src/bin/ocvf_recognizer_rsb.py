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
from Queue import Queue
import cv2
from optparse import OptionParser
import os
import signal
import sys
import time

import rsb
from rstconverters.opencv import IplimageConverter
from rstsandbox.vision.HeadObjects_pb2 import HeadObjects

import numpy as np
from ocvfacerec.facedet.detector import CascadedDetector
from ocvfacerec.facerec.serialization import load_model
from ocvfacerec.helper.PersonWrapper import PersonWrapper
from ocvfacerec.helper.common import *
from ocvfacerec.trainer.thetrainer import ExtendedPredictableModel
from ocvfacerec.trainer.thetrainer import TheTrainer
from rst.geometry.PointCloud2DInt_pb2 import PointCloud2DInt


# OCVF Imports
# RSB Specifics
class Recognizer(object):

    def __init__(self, model, camera_id, cascade_filename, run_local, inscope="/rsbopencv/ipl",
                 outscope="/ocvfacerec/rsb/people", wait=50, notification="/ocvfacerec/rsb/restart/",
                 show_gui=False):
        self.model = model
        self.wait = wait
        self.notification_scope = notification
        self.show_gui = show_gui
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.1)

        if run_local:
            print ">> Error Run local selected in RSB based Recognizer"
            sys.exit(1)

        self.doRun = True
        self.restart = False

        def signal_handler(signal, frame):
            print "\n>> RSB Exiting"
            self.doRun = False

        signal.signal(signal.SIGINT, signal_handler)

        # RSB
        try:
            rsb.converter.registerGlobalConverter(IplimageConverter())
        except Exception, e:
            # If they are already loaded, the lib throws an exception ...
            # print ">> [Error] While loading RST converter: ", e
            pass

        self.listener = rsb.createListener(inscope)
        self.lastImage = Queue(1)

        self.restart_listener = rsb.createListener(self.notification_scope)
        self.last_restart_request = Queue(1)

        try:
            rsb.converter.registerGlobalConverter(rsb.converter.ProtocolBufferConverter(messageClass=HeadObjects))
        except Exception, e:
            # If they are already loaded, the lib throws an exception ...
            # print ">> [Error] While loading RST converter: ", e
            pass

        self.person_publisher = rsb.createInformer(outscope, dataType=HeadObjects)

        # This must be set at last!
        rsb.setDefaultParticipantConfig(rsb.ParticipantConfig.fromDefaultSources())

    def add_last_image(self, image_event):
        try:
            self.lastImage.get(False)
        except Exception, e:
            pass
        self.lastImage.put((np.asarray(image_event.data[:, :]), image_event.getId()), False)

    def add_restart_request(self, restart_event):
        try:
            self.last_restart_request.get(False)
        except Exception, e:
            pass
        self.last_restart_request.put(restart_event.data, False)

    def publish_persons(self, persons, cause_uuid):
        # Gather the information of every head
        rsb_person_list = HeadObjects()
        for a_person in persons:
            rsb_person_list.head_objects.extend([a_person.to_rsb_msg()])

        # Create the event and add the cause. Maybe some day someone will use
        # this reference to the cause :)
        event = rsb.Event(scope=self.person_publisher.getScope(),
                          data=rsb_person_list,
                          type=type(rsb_person_list),
                          causes=[cause_uuid])

        # Publish the data
        self.person_publisher.publishEvent(event)

    def run_distributed(self):
        print ">> Activating RSB Listener"
        self.listener.addHandler(self.add_last_image)
        self.restart_listener.addHandler(self.add_restart_request)
        # TODO # TODO Implement Result Informer (ClassificationResult)
        while self.doRun:
            # GetLastImage is blocking so we won't get a "None" Image
            image, cause_uuid = self.lastImage.get(True)
            # This should not be resized with a fixed rate, this should be rather configured by the sender
            # i.e. by sending smaller images. Don't fiddle with input data in two places.
            # img = cv2.resize(image, (image.shape[1] / 2, image.shape[0] / 2), interpolation=cv2.INTER_CUBIC)
            image_size = (320, 240)
            img = cv2.resize(image, image_size, interpolation=cv2.INTER_CUBIC)
            imgout = img.copy()

            persons = []
            for i, r in enumerate(self.detector.detect(img)):
                x0, y0, x1, y1 = r
                # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
                face = img[y0:y1, x0:x1]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, self.model.image_size, interpolation=cv2.INTER_CUBIC)

                # The prediction result
                prediction = self.model.predict(face)
                predicted_label = prediction[0]
                classifier_output = prediction[1]

                # Now let's get the distance from the assuming a 1-Nearest Neighbor.
                # Since it's a 1-Nearest Neighbor only look take the zero-th element:
                distance = float(classifier_output['distances'][0])
                name = str(self.model.subject_names[predicted_label])

                # Create a PersonWrapper
                a_person = PersonWrapper(r, name, distance, image_size)
                persons.append(a_person)

                # Draw the face area in image:
                cv2.rectangle(imgout, (x0, y0), (x1, y1), (0, 0, 255), 2)
                # Draw the predicted name (folder name...):
                draw_str(imgout, (x0 - 20, y0 - 40), "Label " + a_person.name)
                draw_str(imgout, (x0 - 20, y0 - 20), "Feature Distance " + "%1.1f" % a_person.reliability)

            if self.show_gui:
                cv2.imshow('OCVFACEREC < RSB STREAM', imgout)
                cv2.waitKey(self.wait)
            else:
                # Sleep for the desired time, less CPU
                time.sleep(self.wait * 0.01)

            if len(persons) > 0:
                # Publish the result
                self.publish_persons(persons, cause_uuid)

            # Check if external restart requested
            try:
                z = self.last_restart_request.get(False)
                if z:
                    self.restart = True
                    self.doRun = False
            except Exception, e:
                pass

        print ">> Deactivating RSB Listener"
        self.listener.deactivate()
        self.restart_listener.deactivate()
        self.person_publisher.deactivate()
        # informer.deactivate()


if __name__ == '__main__':
    # model.pkl is a pickled (hopefully trained) PredictableModel, which is
    # used to make predictions. You can learn a model yourself by passing the
    # parameter -d (or --dataset) to learn the model from a given dataset.
    usage = "Usage: %prog [options] model_filename"
    # Add options for training, resizing, validation and setting the camera id:
    parser = OptionParser(usage=usage)
    parser.add_option("-r", "--resize", action="store", type="string", dest="size", default="70x70",
                      help="Resizes the given dataset to a given size in format [width]x[height] (default: 70x70).")
    parser.add_option("-v", "--validate", action="store", dest="numfolds", type="int", default=None,
                      help="Performs a k-fold cross validation on the dataset, if given (default: None).")
    parser.add_option("-t", "--train", action="store", dest="dataset", type="string", default=None,
                      help="Trains the model on the given dataset.")
    parser.add_option("-c", "--cascade", action="store", dest="cascade_filename",
                      help="Sets the path to the Haar Cascade used for the face detection part [haarcascade_frontalface_alt2.xml].")
    parser.add_option("-s", "--rsb-source", action="store", dest="rsb_source", default="/rsbopencv/ipl",
                      help="Grab video from RSB Middleware (default: %default)")
    parser.add_option("-d", "--rsb-destination", action="store", dest="rsb_destination", default="/ocvfacerec/rsb/people",
                      help="Target RSB scope to which persons are published (default: %default).")
    parser.add_option("-n", "--restart-notification", action="store", dest="restart_notification",
                      default="/ocvfacerec/restart",
                      help="Target Topic where a simple restart message is received (default: %default).")
    parser.add_option("-w", "--wait", action="store", dest="wait_time", default=20, type="int",
                      help="Amount of time (in ms) to sleep between face identification frames (default: %default).")
    parser.add_option("-g", "--no-gui", dest="show_gui", action='store_false', default=True,
                      help="Hides the GUI elements for headless mode (default: show gui).")
    (options, args) = parser.parse_args()
    print "\n"
    # Check if a model name was passed:
    if len(args) == 0:
        print ">> [Error] No Prediction Model was given."
        sys.exit(1)
    # This model will be used (or created if the training parameter (-t, --train) exists:
    model_filename = args[0]
    # Check if the given model exists, if no dataset was passed:
    if (options.dataset is None) and (not os.path.exists(model_filename)):
        print ">> [Error] No Prediction Model found at '%s'." % model_filename
        sys.exit(1)
    # Check if the given (or default) cascade file exists:
    if not os.path.exists(options.cascade_filename):
        print ">> [Error] No Cascade File found at '%s'." % options.cascade_filename
        sys.exit(1)
    # We are resizing the images to a fixed size, as this is neccessary for some of
    # the algorithms, some algorithms like LBPH don't have this requirement. To
    # prevent problems from popping up, we resize them with a default value if none
    # was given:
    try:
        image_size = (int(options.size.split("x")[0]), int(options.size.split("x")[1]))
    except Exception, e:
        print ">> [Error] Unable to parse the given image size '%s'. Please pass it in the format [width]x[height]!" % options.size
        sys.exit(1)

    if options.dataset:
        trainer = TheTrainer(options.dataset, image_size, model_filename, _numfolds=options.numfolds)
        trainer.train()

    print ">> Loading model <-- " + str(model_filename)
    model = load_model(model_filename)
    print ">> Known Persons --> ", ", ".join(model.subject_names.values())
    if not isinstance(model, ExtendedPredictableModel):
        print ">> [Error] The given model is not of type '%s'." % "ExtendedPredictableModel"
        sys.exit(1)
    print ">> Using Remote RSB Camera Stream <-- " + str(options.rsb_source)
    print ">> Publishing regognised people to --> " + str(options.rsb_destination)
    print ">> Restart Recognizer Scope <-- " + str(options.restart_notification)
    x = Recognizer(model=model, camera_id=None, cascade_filename=options.cascade_filename, run_local=False,
                   inscope=options.rsb_source, outscope=str(options.rsb_destination), wait=options.wait_time,
                   notification=options.restart_notification, show_gui=options.show_gui)
    x.run_distributed()
    while x.restart:
        time.sleep(1)
        model = load_model(model_filename)
        print ">> Known Persons --> ", ", ".join(model.subject_names.values())
        x = Recognizer(model=model, camera_id=None, cascade_filename=options.cascade_filename, run_local=False,
                       inscope=options.rsb_source, outscope=str(options.rsb_destination), wait=options.wait_time,
                       notification=options.restart_notification, show_gui=options.show_gui)
        x.run_distributed()
