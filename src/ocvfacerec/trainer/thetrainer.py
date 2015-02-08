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
import os
import cv2
import sys
import logging
import numpy as np

# OCVF imports
from ocvfacerec.facerec.feature import Fisherfaces
from ocvfacerec.facerec.model import PredictableModel
from ocvfacerec.facerec.distance import EuclideanDistance
from ocvfacerec.facerec.classifier import NearestNeighbor
from ocvfacerec.facerec.validation import KFoldCrossValidation
from ocvfacerec.facerec.serialization import save_model


class ExtendedPredictableModel(PredictableModel):
    """ Subclasses the PredictableModel to store some more
        information, so we don't need to pass the dataset
        on each program call...
    """
    def __init__(self, feature, classifier, image_size, subject_names):
        PredictableModel.__init__(self, feature=feature, classifier=classifier)
        self.image_size = image_size
        self.subject_names = subject_names


class TheTrainer():

    def __init__(self, _data_set, _image_size, _model_filename, _numfolds=None):
        self.dataset = _data_set
        self.image_size = _image_size
        self.model_filename = _model_filename
        self.numfolds = _numfolds

    @staticmethod
    def read_images(path, image_size=None):
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
                        # Resize to given size (if given)
                        if image_size is not None:
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

    @staticmethod
    def get_model(image_size, subject_names):
        """ This method returns the PredictableModel which is used to learn a model
            for possible further usage. If you want to define your own model, this
            is the method to return it from!
        """
        # Define the Fisherfaces Method as Feature Extraction method:
        feature = Fisherfaces()
        # Define a 1-NN classifier with Euclidean Distance:
        classifier = NearestNeighbor(dist_metric=EuclideanDistance(), k=1)
        # Return the model as the combination:
        return ExtendedPredictableModel(feature=feature, classifier=classifier, image_size=image_size, subject_names=subject_names)

    def read_subject_names(path):
        """Reads the folders of a given directory, which are used to display some
            meaningful name instead of simply displaying a number.

        Args:
            path: Path to a folder with subfolders representing the subjects (persons).

        Returns:
            folder_names: The names of the folder, so you can display it in a prediction.
        """
        folder_names = []
        for dirname, dirnames, filenames in os.walk(path):
            for subdirname in dirnames:
                folder_names.append(subdirname)
        return folder_names

    def train(self):
        # Check if the given dataset exists:
        if not os.path.exists(self.dataset):
            print ">> [Error] No Dataset Found at '%s'." % self.dataset
            sys.exit(1)
        # Reads the images, labels and folder_names from a given dataset. Images
        # are resized to given size on the fly:
        print ">> Loading Dataset <-- " + self.dataset
        [images, labels, subject_names] = self.read_images(self.dataset, self.image_size)
        # Zip us a {label, name} dict from the given data:
        list_of_labels = list(xrange(max(labels) + 1))
        subject_dictionary = dict(zip(list_of_labels, subject_names))
        # Get the model we want to compute:
        model = self.get_model(image_size=self.image_size, subject_names=subject_dictionary)
        # Sometimes you want to know how good the model may perform on the data
        # given, the script allows you to perform a k-fold Cross Validation before
        # the Detection & Recognition part starts:
        if self.numfolds is not None:
            print ">> Validating Model With %s Folds..." % self.numfolds
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
            crossval = KFoldCrossValidation(model, k=self.numfolds)
            crossval.validate(images, labels)
            crossval.print_results()
        # Compute the model:
        print ">> Computing Model..."
        model.compute(images, labels)
        # And save the model, which uses Pythons pickle module:
        print ">> Saving Model..."
        save_model(self.model_filename, model)