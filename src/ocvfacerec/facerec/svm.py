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

from classifier import SVM
from ocvfacerec.facerec.validation import KFoldCrossValidation
from ocvfacerec.facerec.model import PredictableModel
from ocvfacerec.facerec.svmutil import *
from itertools import product
import numpy as np
import logging


def range_f(begin, end, step):
    seq = []
    while True:
        if step == 0: break
        if step > 0 and begin > end: break
        if step < 0 and begin < end: break
        seq.append(begin)
        begin = begin + step
    return seq


def grid(grid_parameters):
    grid = []
    for parameter in grid_parameters:
        begin, end, step = parameter
        grid.append(range_f(begin, end, step))
    return product(*grid)


def grid_search(model, X, y, C_range=(-5, 15, 2), gamma_range=(3, -15, -2), k=5, num_cores=1):
    if not isinstance(model, PredictableModel):
        raise TypeError(
            "GridSearch expects a PredictableModel. If you want to perform optimization on raw data use facerec.feature.Identity to pass unpreprocessed data!")
    if not isinstance(model.classifier, SVM):
        raise TypeError("GridSearch expects a SVM as classifier. Please use a facerec.classifier.SVM!")

    logger = logging.getLogger("facerec.svm.gridsearch")
    logger.info("Performing a Grid Search.")

    # best parameter combination to return
    best_parameter = svm_parameter("-q")
    best_parameter.kernel_type = model.classifier.param.kernel_type
    best_parameter.nu = model.classifier.param.nu
    best_parameter.coef0 = model.classifier.param.coef0
    # either no gamma given or kernel is linear (only C to optimize)
    if (gamma_range is None) or (model.classifier.param.kernel_type == LINEAR):
        gamma_range = (0, 0, 1)

    # best validation error so far
    best_accuracy = np.finfo('float').min

    # create grid (cartesian product of ranges)        
    g = grid([C_range, gamma_range])
    results = []
    for p in g:
        C, gamma = p
        C, gamma = 2 ** C, 2 ** gamma
        model.classifier.param.C, model.classifier.param.gamma = C, gamma

        # perform a k-fold cross validation
        cv = KFoldCrossValidation(model=model, k=k)
        cv.validate(X, y)

        # append parameter into list with accuracies for all parameter combinations
        results.append([C, gamma, cv.accuracy])

        # store best parameter combination
        if cv.accuracy > best_accuracy:
            logger.info("best_accuracy=%s" % (cv.accuracy))
            best_accuracy = cv.accuracy
            best_parameter.C, best_parameter.gamma = C, gamma

        logger.info("%d-CV Result = %.2f." % (k, cv.accuracy))

    # set best parameter combination to best found
    return best_parameter, results
