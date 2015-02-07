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

import numpy as np
from ocvfacerec.facerec.feature import AbstractFeature


class FeatureOperator(AbstractFeature):
    """
    A FeatureOperator operates on two feature models.
    
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
    """

    def __init__(self, model1, model2):
        if (not isinstance(model1, AbstractFeature)) or (not isinstance(model2, AbstractFeature)):
            raise Exception("A FeatureOperator only works on classes implementing an AbstractFeature!")
        self.model1 = model1
        self.model2 = model2

    def __repr__(self):
        return "FeatureOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"


class ChainOperator(FeatureOperator):
    """
    The ChainOperator chains two feature extraction modules:
        model2.compute(model1.compute(X,y),y)
    Where X can be generic input data.
    
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
    """

    def __init__(self, model1, model2):
        FeatureOperator.__init__(self, model1, model2)

    def compute(self, X, y):
        X = self.model1.compute(X, y)
        return self.model2.compute(X, y)

    def extract(self, X):
        X = self.model1.extract(X)
        return self.model2.extract(X)

    def __repr__(self):
        return "ChainOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"


class CombineOperator(FeatureOperator):
    """
    The CombineOperator combines the output of two feature extraction modules as:
      (model1.compute(X,y),model2.compute(X,y))
    , where    the output of each feature is a [1xN] or [Nx1] feature vector.
        
        
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
        
    """

    def __init__(self, model1, model2):
        FeatureOperator.__init__(self, model1, model2)

    def compute(self, X, y):
        A = self.model1.compute(X, y)
        B = self.model2.compute(X, y)
        C = []
        for i in range(0, len(A)):
            ai = np.asarray(A[i]).reshape(1, -1)
            bi = np.asarray(B[i]).reshape(1, -1)
            C.append(np.hstack((ai, bi)))
        return C

    def extract(self, X):
        ai = self.model1.extract(X)
        bi = self.model2.extract(X)
        ai = np.asarray(ai).reshape(1, -1)
        bi = np.asarray(bi).reshape(1, -1)
        return np.hstack((ai, bi))

    def __repr__(self):
        return "CombineOperator(" + repr(self.model1) + "," + repr(self.model2) + ")"


class CombineOperatorND(FeatureOperator):
    """
    The CombineOperator combines the output of two multidimensional feature extraction modules.
        (model1.compute(X,y),model2.compute(X,y))
        
    Args:
        model1 [AbstractFeature]
        model2 [AbstractFeature]
        hstack [bool] stacks data horizontally if True and vertically if False
        
    """

    def __init__(self, model1, model2, hstack=True):
        FeatureOperator.__init__(self, model1, model2)
        self._hstack = hstack

    def compute(self, X, y):
        A = self.model1.compute(X, y)
        B = self.model2.compute(X, y)
        C = []
        for i in range(0, len(A)):
            if self._hstack:
                C.append(np.hstack((A[i], B[i])))
            else:
                C.append(np.vstack((A[i], B[i])))
        return C

    def extract(self, X):
        ai = self.model1.extract(X)
        bi = self.model2.extract(X)
        if self._hstack:
            return np.hstack((ai, bi))
        return np.vstack((ai, bi))

    def __repr__(self):
        return "CombineOperatorND(" + repr(self.model1) + "," + repr(self.model2) + ", hstack=" + str(
            self._hstack) + ")"
