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

import os
import cv2
import sys
from ocvfacerec.helper.video import *
from ocvfacerec.helper.common import *
from ocvfacerec.trainer.thetrainer import TheTrainer
from ocvfacerec.facerec.serialization import load_model
from ocvfacerec.facedet.detector import CascadedDetector
from ocvfacerec.trainer.thetrainer import ExtendedPredictableModel


class Recognizer(object):
    def __init__(self, model, camera_id, cascade_filename, run_local, wait=50):
        self.model = model
        self.wait = wait
        self.detector = CascadedDetector(cascade_fn=cascade_filename, minNeighbors=5, scaleFactor=1.1)
        if run_local:
            self.cam = create_capture(camera_id)

    def run(self):
        while True:
            ret, frame = self.cam.read()
            # Resize the frame to half the original size for speeding up the detection process:
            img = cv2.resize(frame, (frame.shape[1] / 2, frame.shape[0] / 2), interpolation=cv2.INTER_CUBIC)
            imgout = img.copy()
            for i, r in enumerate(self.detector.detect(img)):
                x0, y0, x1, y1 = r
                # (1) Get face, (2) Convert to grayscale & (3) resize to image_size:
                face = img[y0:y1, x0:x1]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, self.model.image_size, interpolation=cv2.INTER_CUBIC)
                prediction = self.model.predict(face)
                predicted_label = prediction[0]
                classifier_output = prediction[1]
                # Now let's get the distance from the assuming a 1-Nearest Neighbor.
                # Since it's a 1-Nearest Neighbor only look take the zero-th element:
                distance = classifier_output['distances'][0]
                # Draw the face area in image:
                cv2.rectangle(imgout, (x0, y0), (x1, y1), (0, 0, 255), 2)
                # Draw the predicted name (folder name...):
                draw_str(imgout, (x0 - 20, y0 - 40), "Label " + self.model.subject_names[predicted_label])
                draw_str(imgout, (x0 - 20, y0 - 20), "Distance " + "%1.2f" % distance)
            cv2.imshow('OCVFACEREC LOCAL CAMERA', imgout)
            key = cv2.waitKey(self.wait)
            if 'q' == chr(key & 255):
                print "<q> Pressed. Exiting."
                break

if __name__ == '__main__':
    from optparse import OptionParser
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
    parser.add_option("-i", "--id", action="store", dest="camera_id", type="int", default=0,
                      help="Sets the Camera Id to be used (default: 0).")
    parser.add_option("-c", "--cascade", action="store", dest="cascade_filename",
                      default="haarcascade_frontalface_alt2.xml",
                      help="Sets the path to the Haar Cascade used for the face detection part (default: haarcascade_frontalface_alt2.xml).")
    parser.add_option("-w", "--wait", action="store", dest="wait_time", default=20, type="int",
                      help="Amount of time (in ms) to sleep between face identifaction runs (frames). Default is 20 ms")
    (options, args) = parser.parse_args()
    print "\n"
    # Check if a model name was passed:
    if len(args) == 0:
        print ">> [Error] No prediction model was given."
        sys.exit(1)
    # This model will be used (or created if the training parameter (-t, --train) exists:
    model_filename = args[0]
    # Check if the given model exists, if no dataset was passed:
    if (options.dataset is None) and (not os.path.exists(model_filename)):
        print ">> [Error] No prediction model found at '%s'." % model_filename
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
    # We have got a dataset to learn a new model from:
    if options.dataset:
        # Check if the given dataset exists:
        if not os.path.exists(options.dataset):
            print ">> [Error] No dataset found at '%s'." % options.dataset
            sys.exit(1)
            # Reads the images, labels and folder_names from a given dataset. Images

        trainer = TheTrainer(options.dataset, image_size, model_filename, _numfolds=options.numfolds)
        trainer.train()

    print ">> Loading model... " + str(model_filename)
    model = load_model(model_filename)
    # We operate on an ExtendedPredictableModel. Quit the Recognizerlication if this
    # isn't what we expect it to be:
    if not isinstance(model, ExtendedPredictableModel):
        print ">> [Error] The given model is not of type '%s'." % "ExtendedPredictableModel"
        sys.exit(1)
    # Now it's time to finally start the Recognizerlication! It simply get's the model
    # and the image size the incoming webcam or video images are resized to:
    print ">> Using Local Camera " + "/dev/video" + str(options.camera_id)
    Recognizer(model=model, camera_id=options.camera_id, cascade_filename=options.cascade_filename, run_local=True, wait=options.wait_time).run()

