# Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de> and
# Norman Koester <nkoester[at]techfak.uni-bielefeld.de> and
# Florian Lier <flier[at]techfak.uni-bielefeld.de>
#
# Released to public domain under terms of the BSD Simplified license.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# * Redistributions of source code must retain the above copyright
#          notice, this list of conditions and the following disclaimer.
#        * Redistributions in binary form must reproduce the above copyright
#          notice, this list of conditions and the following disclaimer in the
#          documentation and/or other materials provided with the distribution.
#        * Neither the name of the organization nor the names of its contributors
#          may be used to endorse or promote products derived from this software
#          without specific prior written permission.
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

import Image
from Queue import Queue
import cv2
import cv
import logging
import optparse
import os, errno
import signal
import sys
import traceback

import rsb
from rsb.converter import ProtocolBufferConverter
from rsb.converter import registerGlobalConverter
from rstconverters.opencv import IplimageConverter
import rstsandbox

import numpy as np
from ocvfacerec.facerec.classifier import NearestNeighbor
from ocvfacerec.facerec.distance import EuclideanDistance
from ocvfacerec.facerec.feature import Fisherfaces
from ocvfacerec.facerec.model import PredictableModel
from ocvfacerec.facerec.serialization import save_model
from ocvfacerec.facerec.validation import KFoldCrossValidation


def detect_face(image, face_cascade, return_image=False):

    # This function takes a grey scale cv image and finds
    # the patterns defined in the haarcascade function

    min_size = (20, 20)
    haar_scale = 1.1
    min_neighbors = 5
    haar_flags = 0

    # Equalize the histogram
    cv.EqualizeHist(image, image)

    # Detect the faces
    faces = cv.HaarDetectObjects(
        image, face_cascade, cv.CreateMemStorage(0),
        haar_scale, min_neighbors, haar_flags, min_size
    )

    # If faces are found
    if faces and return_image:
        for ((x, y, w, h), n) in faces:
            # Convert bounding box to two CvPoints
            pt1 = (int(x), int(y))
            pt2 = (int(x + w), int(y + h))
            cv.Rectangle(image, pt1, pt2, cv.RGB(255, 0, 0), 5, 8, 0)

    if return_image:
        return image
    else:
        return faces


def pil2_cvgrey(pil_im):
    # Convert a PIL image to a greyscale cv image
    # from: http://pythonpath.wordpress.com/2012/05/08/pil-to-opencv-image/
    pil_im = pil_im.convert('L')
    cv_im = cv.CreateImageHeader(pil_im.size, cv.IPL_DEPTH_8U, 1)
    cv.SetData(cv_im, pil_im.tostring(), pil_im.size[0])
    return cv_im

def img_crop(image, crop_box, box_scale=1):
    # Crop a PIL image with the provided box [x(left), y(upper), w(width), h(height)]

    # Calculate scale factors
    x_delta = max(crop_box[2] * (box_scale - 1), 0)
    y_delta = max(crop_box[3] * (box_scale - 1), 0)


    # Convert cv box to PIL box [left, upper, right, lower]
    pil_box = [crop_box[0] - x_delta, crop_box[1] - y_delta, crop_box[0] + crop_box[2] + x_delta,
               crop_box[1] + crop_box[3] + y_delta]

    return image.crop(pil_box)

def face_crop_single_image(pil_image, face_cascade, box_scale=1):

    cv_im = pil2_cvgrey(pil_image)
    faces = detect_face(cv_im, face_cascade)
    face_list = []

    cropped_image = None
    if faces:
        for face in faces:
            cropped_image = img_crop(pil_image, face[0], box_scale=box_scale)
            face_list.append(cropped_image)
    return cropped_image



def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


class ExtendedPredictableModel(PredictableModel):
    """ Subclasses the PredictableModel to store some more
        information, so we don't need to pass the dataset
        on each program call...
    """

    def __init__(self, feature, classifier, image_size, subject_names):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.image_size = image_size
        self.subject_names = subject_names


class MiddlewareConnector(object):
    # TODO USE ABC?
    pass


class ROSConnector(MiddlewareConnector):
    # TODO implement
    def __init__(self):
        raise Exception("Not Implemented yet ...")
        pass

    def activate(self, source):
        pass
    def deactivate(self):
        pass
    def get_image(self):
        pass


class RSBConnector(MiddlewareConnector):

    def __init__(self):
        pass

    def add_last_image(self, image_event):
        try:
            self.lastImage.get(False)
        except Exception, e:
            pass
        self.lastImage.put(np.asarray(image_event.data[:, :]), False)

    def add_last_train(self, retrain_event):
        try:
            self.last_train.get(False)
        except Exception, e:
            pass
        self.last_train.put(retrain_event.data, False)

    def activate(self, image_source, retrain_source, restart_target):
        registerGlobalConverter(IplimageConverter())
        rsb.setDefaultParticipantConfig(rsb.ParticipantConfig.fromDefaultSources())

        # listen to image events
        self.image_listener = rsb.createListener(image_source)
        self.lastImage = Queue(1)
        self.image_listener.addHandler(self.add_last_image)

        # listen to re-train events with a name
        self.training_start = rsb.createListener(retrain_source)
        self.last_train = Queue(10)
        self.training_start.addHandler(self.add_last_train)

        # publisher to restart recogniser
        self.restart_publisher = rsb.createInformer(restart_target, dataType=str)


    def deactivate(self):
        self.image_listener.deactivate()
        self.training_start.deactivate()
        self.restart_publisher.deactivate()

    def restart_classifier(self):
        # Send a short "restart" event to the recognizer
        self.restart_publisher.publishData("restart")

    def wait_for_start_training(self):
        return self.last_train.get(True, timeout=1)

    def get_image(self):
        return self.lastImage.get(True, timeout=10)


class Trainer(object):

    def __init__(self, options, middelware_connector):
        self.middleware = middelware_connector
        self.middleware_type = options.middleware_type
        self.retrain_source = options.retrain_source
        self.image_source = options.image_source
        self.restart_target = options.restart_target
        self.mugshot_size = options.mugshot_size
        self.counter = 0
        self.cascade_filename = cv.Load(options.cascade_filename)

        self.training_data_path = options.training_data_path
        self.training_image_number = options.training_image_number
        try:
            self.image_size = (int(options.image_size.split("x")[0]), int(options.image_size.split("x")[1]))
        except:
            print ">> [Error] Unable to parse the given image size '%s'. Please pass it in the format [width]x[height]!" % options.image_size
            sys.exit(1)

        self.model_path = options.model_path
        self.abort_training = False
        self.doRun = True

        def signal_handler(signal, frame):
            print ">> Exiting..."
            self.doRun = False
            self.abort_training = True
        signal.signal(signal.SIGINT, signal_handler)

    def run(self):
        print "path to training data: %s " % self.training_data_path
        print "path to model: %s\n" % self.model_path
        print "middleware: %s" % self.middleware_type
        print "image source: %s " % self.image_source
        print "retrain command source: %s\n" % self.retrain_source

        print "run dos run...\n"
        try:
            self.middleware.activate(self.image_source, self.retrain_source, self.restart_target)
        except Exception, e:
            print ">> [ERROR] can't activate middleware! "
            traceback.print_exc()

        try:
            self.middleware.activate(self.image_source, self.retrain_source, self.restart_target)
        except Exception, e:
            print "ERROR: ", e
            # self.doRun = False

        self.re_train()
        print ">> Ready.\n"
        while self.doRun:
            try:
                train_name = self.middleware.wait_for_start_training()
            except Exception, e:
                # Check every timeout seconds if we are supposed to exit
                continue

            try:
                print "Training for '%s' (run %d)" % (train_name, self.counter)
                if self.record_images(train_name):
                    self.re_train()
                    self.restart_classifier()
                    self.counter += 1
                else:
                    print ">>\tUnable to collect enough mugshots :("

                print ">> Ready.\n"

            except Exception, e:
                print ">> [ERROR]: ", e
                traceback.print_exc()
                continue


        print "Deacivating middleware ..."
        self.middleware.deactivate()
        print "done. bye bye!"

    def record_images(self, train_name):
        print ">> Recording %d images from %s..." % (self.training_image_number, self.image_source)
        person_image_path = os.path.join(self.training_data_path, train_name)
        mkdir_p(person_image_path)
        num_mughshots = 0
        abort_threshold = 40
        abort_count = 0
        switch = False
        print ">>\t",
        while num_mughshots < self.training_image_number and not self.abort_training and abort_count < abort_threshold:

            # take every second frame to add some more variance
            switch = not switch
            if switch:
                input_image = self.middleware.get_image()
            else:
                continue


            im = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
            cropped_image = face_crop_single_image(im, self.cascade_filename)

            ok_shot = False
            if cropped_image:
                if cropped_image.size[0] >= self.mugshot_size and cropped_image.size[1] >= self.mugshot_size:
                    sys.stdout.write("+")
                    sys.stdout.flush()
                    cropped_image.save(os.path.join(person_image_path, "%03d.jpg" % num_mughshots))
                    num_mughshots += 1
                    ok_shot = True

            if ok_shot is False:
                abort_count += 1
                sys.stdout.write("-")
                sys.stdout.flush()

        print ""
        if abort_count >= abort_threshold:
            return False
        else:
            return True
            # im.save(os.path.join(person_image_path, "%03d.jpg" % i))

    def re_train(self):
        print ">> Re-train running ..."
        walk_result = [x[0] for x in os.walk(self.training_data_path)][1:]
        if len(walk_result) > 0:
            print ">>\tpersons available for training: ", ", ".join([x.split("/")[-1] for x in walk_result])
        else:
            print ">>\tno persons found for training :("
            return

        [images, labels, subject_names] = self.read_images(self.training_data_path, self.image_size)

        if len(labels) == 0:
            self.doRun = False
            raise Exception("No images in folder %s This is bad!" % self.training_data_path)

        # Zip us a {label, name} dict from the given data:
        list_of_labels = list(xrange(max(labels) + 1))
        subject_dictionary = dict(zip(list_of_labels, subject_names))
        # Get the model we want to compute:
        model = self.get_model(image_size=self.image_size, subject_names=subject_dictionary)
        # Sometimes you want to know how good the model may perform on the data
        # given, the script allows you to perform a k-fold Cross Validation before
        # the Detection & Recognition part starts:
        if options.numfolds:
            print ">> Validating model with %s folds..." % options.numfolds
            # We want to have some log output, so set up a new logging handler
            # and point it to stdout:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            # Add a handler to facerec modules, so we see what's going on inside:
            logger = logging.getLogger("facerec")
            logger.addHandler(handler)
            logger.setLevel(logging.DEBUG)
            # Perform the validation & print results:
            crossval = KFoldCrossValidation(model, k=options.numfolds)
            crossval.validate(images, labels)
            crossval.print_results()
        # Compute the model:
        print ">> Computing the model..."
        model.compute(images, labels)
        # And save the model, which uses Pythons pickle module:
        print ">> Saving the model to... %s" % self.model_path
        save_model(self.model_path, model)

    def restart_classifier(self):
        print ">> Restarting classifier..."
        self.middleware.restart_classifier()

    def get_model(self, image_size, subject_names):
        """ This method returns the PredictableModel which is used to learn a model
            for possible further usage. If you want to define your own model, this
            is the method to return it from!
        """
        # Define the Fisherfaces Method as Feature Extraction method:
        feature = Fisherfaces()
        # Define a 1-NN classifier with Euclidean Distance:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
        # Return the model as the combination:
        return ExtendedPredictableModel(feature=feature, classifier=classifier, image_size=image_size,
                                        subject_names=subject_names)

    def read_images(self, path, image_size=None):
        """Reads the images in a given folder, resizes images on the fly if size is given.

        Args:
            path: Path to a folder with subfolders representing the subjects (persons).
            sz: A tuple with the size Resizes

        Returns:
            A list [X, y, folder_names]

                X: The images, which is a Python list of numpy arrays.
                y: The corresponding labels (the unique number of the subject, person) in a Python list.
                folder_names: The names of the folder, so you can display it in a prediction.
        """
        c = 0
        X = []
        y = []
        folder_names = []
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                folder_names.append(subdirname)
                subject_path = os.path.join(dirname, subdirname)
                for filename in os.listdir(subject_path):
                    try:
                        im = cv2.imread(os.path.join(subject_path, filename), cv2.IMREAD_GRAYSCALE)
                        # resize to given size (if given)
                        if (image_size is not None):
                            im = cv2.resize(im, image_size)
                        X.append(np.asarray(im, dtype=np.uint8))
                        y.append(c)
                    except IOError, (errno, strerror):
                        print ">> I/O error({0}): {1}".format(errno, strerror)
                    except:
                        print ">> Unexpected error:", sys.exc_info()[0]
                        raise
                c = c + 1
        return [X, y, folder_names]


if __name__ == '__main__':
    usage = "usage: %prog [options] model_filename"
    # Add options for training, resizing, validation and setting the camera id:
    parser = optparse.OptionParser(usage=usage)
    group_mw = optparse.OptionGroup(parser, 'Middleware Options')
    group_algorithm = optparse.OptionGroup(parser, 'Algorithm Options')
    group_io = optparse.OptionGroup(parser, 'IO Options')

    group_mw.add_option("-w", "--middleware", action="store",
                      dest="middleware_type", type="string", default="rsb",
                      help="Type of middleware to use. Currently supported: 'rsb' and 'ros' (default: %default).")
    group_mw.add_option("-s", "--image-source", action="store",
                      dest="image_source", default="/video/",
                      help="Source (topic/scope) from which to get video images (default: %default).")
    group_mw.add_option("-e", "--re-train-source", action="store",
                      dest="retrain_source", default="/ocvfacerec/trainer/retrain",
                      help="Source (topic/scope) from which to get a re-train message (=basic string, representing name of the person) (default: %default).")
    group_mw.add_option("-p", "--restart-target", action="store",
                      dest="restart_target", default="/ocvfacerec/recognizer/restart",
                      help="Target (topic/scope) to where a simple restart message is sent (=basic string, containing 'restart') (default: %default).")

    group_io.add_option("-m", "--model-path", action="store",
                      dest="model_path", default="/tmp/model.pkl",
                      help="Storage path for the model file (default: %default).")
    group_io.add_option("-t", "--training-path", action="store",
                      dest="training_data_path", default="/tmp/training_data",
                      help="Storage path for the training data files (default: %default).")

    group_algorithm.add_option("-n", "--training-images", action="store",
                      dest="training_image_number", type="int", default=20,
                      help="Number of images to use for training of a new person(default: %default).")
    group_algorithm.add_option("-r", "--resize", action="store", type="string",
                      dest="image_size", default="70x70",
                      help="Resizes the given and new dataset(s) to a given size in format [width]x[height] (default: %default).")
    group_algorithm.add_option("-v", "--validate", action="store",
                      dest="numfolds", type="int", default=None,
                      help="Performs a k-fold cross validation on the dataset, if given (default: %default).")
    group_algorithm.add_option("-c", "--cascade", action="store", dest="cascade_filename",
                      default="haarcascade_frontalface_alt2.xml",
                      help="Sets the path to the Haar Cascade used for the face detection part (default: %default).")
    group_algorithm.add_option("-l", "--mugshot-size", action="store", type="int", dest="mugshot_size",
                      default=110,
                      help="Sets minimal size (in pixels) required for a mugshot of a person in order to use it for training (default: %default).")


    parser.add_option_group(group_mw)
    parser.add_option_group(group_io)
    parser.add_option_group(group_algorithm)

    (options, args) = parser.parse_args()
    print "\n"

    try:
        mkdir_p(os.path.basename(options.model_path))
        mkdir_p(options.training_data_path)
    except Exception, e:
        print "Error: " + e
        sys.exit()

    if options.middleware_type == "rsb":
        Trainer(options, RSBConnector()).run()
    elif options.middleware_type == "ros":
        Trainer(options, ROSConnector()).run()
    else:
        print "Error! Middleware %s unknown." % options.middleware_type
        sys.exit()
