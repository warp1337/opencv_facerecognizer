# Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de> and
# Florian Lier <flier[at]techfak.uni-bielefeld.de>
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

import numpy as np
from ocvfacerec.facerec.feature import AbstractFeature
from ocvfacerec.facerec.util import as_column_matrix
from ocvfacerec.facerec.lbp import ExtendedLBP
from cvfacerec.facerec.normalization import zscore, minmax
from scipy import ndimage
from scipy.misc import imresize


class Resize(AbstractFeature):
    def __init__(self, size):
        AbstractFeature.__init__(self)
        self._size = size

    def compute(self, X, y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        return imresize(X, self._size)

    def __repr__(self):
        return "Resize (size=%s)" % (self._size,)


class HistogramEqualization(AbstractFeature):
    def __init__(self, num_bins=256):
        AbstractFeature.__init__(self)
        self._num_bins = num_bins

    def compute(self, X, y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        h, b = np.histogram(X.flatten(), self._num_bins, normed=True)
        cdf = h.cumsum()
        cdf = 255 * cdf / cdf[-1]
        return np.interp(X.flatten(), b[:-1], cdf).reshape(X.shape)

    def __repr__(self):
        return "HistogramEqualization (num_bins=%s)" % (self._num_bins)


class TanTriggsPreprocessing(AbstractFeature):
    def __init__(self, alpha=0.1, tau=10.0, gamma=0.2, sigma0=1.0, sigma1=2.0):
        AbstractFeature.__init__(self)
        self._alpha = float(alpha)
        self._tau = float(tau)
        self._gamma = float(gamma)
        self._sigma0 = float(sigma0)
        self._sigma1 = float(sigma1)

    def compute(self, X, y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        X = np.array(X, dtype=np.float32)
        X = np.power(X, self._gamma)
        X = np.asarray(ndimage.gaussian_filter(X, self._sigma1) - ndimage.gaussian_filter(X, self._sigma0))
        X = X / np.power(np.mean(np.power(np.abs(X), self._alpha)), 1.0 / self._alpha)
        X = X / np.power(np.mean(np.power(np.minimum(np.abs(X), self._tau), self._alpha)), 1.0 / self._alpha)
        X = self._tau * np.tanh(X / self._tau)
        return X

    def __repr__(self):
        return "TanTriggsPreprocessing (alpha=%.3f,tau=%.3f,gamma=%.3f,sigma0=%.3f,sigma1=%.3f)" % (
            self._alpha, self._tau, self._gamma, self._sigma0, self._sigma1)


class LBPPreprocessing(AbstractFeature):
    def __init__(self, lbp_operator=ExtendedLBP(radius=1, neighbors=8)):
        AbstractFeature.__init__(self)
        self._lbp_operator = lbp_operator

    def compute(self, X, y):
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        return self._lbp_operator(X)

    def __repr__(self):
        return "LBPPreprocessing (lbp_operator=%s)" % (repr(self._lbp_operator))


class MinMaxNormalizePreprocessing(AbstractFeature):
    def __init__(self, low=0, high=1):
        AbstractFeature.__init__(self)
        self._low = low
        self._high = high

    def compute(self, X, y):
        Xp = []
        XC = as_column_matrix(X)
        self._min = np.min(XC)
        self._max = np.max(XC)
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        return minmax(X, self._low, self._high, self._min, self._max)

    def __repr__(self):
        return "MinMaxNormalizePreprocessing (low=%s, high=%s)" % (self._low, self._high)


class ZScoreNormalizePreprocessing(AbstractFeature):
    def __init__(self):
        AbstractFeature.__init__(self)
        self._mean = 0.0
        self._std = 1.0

    def compute(self, X, y):
        XC = as_column_matrix(X)
        self._mean = XC.mean()
        self._std = XC.std()
        Xp = []
        for xi in X:
            Xp.append(self.extract(xi))
        return Xp

    def extract(self, X):
        return zscore(X, self._mean, self._std)

    def __repr__(self):
        return "ZScoreNormalizePreprocessing (mean=%s, std=%s)" % (self._mean, self._std)
