# Copyright (c) 2012. Philipp Wagner <bytefish[at]gmx[dot]de> and
# Florian Lier <flier@techfak.uni-bielefeld.de>
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

import os as os
from ocvfacerec.facerec.normalization import minmax
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# try to import the PIL Image module
try:
    from PIL import Image
except ImportError:
    import Image
import math as math


def create_font(fontname='Tahoma', fontsize=10):
    return {'fontname': fontname, 'fontsize': fontsize}


def plot_gray(X, sz=None, filename=None):
    if not sz is None:
        X = X.reshape(sz)
    X = minmax(I, 0, 255)
    fig = plt.figure()
    implot = plt.imshow(np.asarray(Ig), cmap=cm.gray)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename, format="png", transparent=False)


def plot_eigenvectors(eigenvectors, num_components, sz, filename=None, start_component=0, rows=None, cols=None,
                      title="Subplot", color=True):
    if (rows is None) or (cols is None):
        rows = cols = int(math.ceil(np.sqrt(num_components)))
    num_components = np.min(num_components, eigenvectors.shape[1])
    fig = plt.figure()
    for i in range(start_component, num_components):
        vi = eigenvectors[0:, i].copy()
        vi = minmax(np.asarray(vi), 0, 255, dtype=np.uint8)
        vi = vi.reshape(sz)

        ax0 = fig.add_subplot(rows, cols, (i - start_component) + 1)

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        plt.title("%s #%d" % (title, i), create_font('Tahoma', 10))
        if color:
            implot = plt.imshow(np.asarray(vi))
        else:
            implot = plt.imshow(np.asarray(vi), cmap=cm.grey)
    if filename is None:
        fig.show()
    else:
        fig.savefig(filename, format="png", transparent=False)


def subplot(title, images, rows, cols, sptitle="subplot", sptitles=[], colormap=cm.gray, ticks_visible=True,
            filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in xrange(len(images)):
        ax0 = fig.add_subplot(rows, cols, (i + 1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma', 10))
        else:
            plt.title("%s #%d" % (sptitle, (i + 1)), create_font('Tahoma', 10))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)


