# Copyright (c) 2012.
# Philipp Wagner <bytefish[at]gmx[dot]de> and
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

import os
import sys
from distutils.dir_util import copy_tree
from sys import platform as _platform
from setuptools import setup

setup(
    name='ocvfacerec',
    version='0.1',
    description='OpenCV Facerecognizer',
    long_description='''OpenCV Facerecognizer (OCVFACEREC) is a simple
        application for face detection and recognition. It provides a set of
        classifiers, features, algorithms, operators and examples including
        distributed processing via ROS and RSB Middleware. It is _heavily_ based on
        Philipp Wagner's facerec Framework [https://github.com/bytefish/facerec]''',
    author='Philipp Wagner, Florian Lier, Norman Koester',
    author_email='flier[at]techfak.uni-bielefeld.de',
    url='https://github.com/warp1337/opencv_facerecognizer.git',
    license='BSD',
    classifiers=[
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS',
    ],
    packages=['ocvfacerec', 'ocvfacerec/facedet', 'ocvfacerec/facerec', 'ocvfacerec/helper',
              'ocvfacerec/trainer', 'ocvfacerec/mwconnector'],
    scripts=["bin/ocvf_recognizer.py", "bin/ocvf_recognizer_ros.py", "bin/ocvf_recognizer_rsb.py",
             "bin/ocvf_interactive_trainer.py", "bin/ocvf_retrain_rsb.py", "bin/ocvf_retrain_ros.py"],
    # Due to heavy dependencies (liblas, ATLAS, etc..) it is easier to install 'SciPy >= 0.14.0'
    # and PIL >= 1.1.7 using your Package Manager, i.e., sudo apt-get install python-scipy python-imaging-*
    install_requires=['NumPy >=1.8.1', 'matplotlib >= 1.2.0']
)

if _platform == "linux" or _platform == "linux2":
    if os.path.isdir("../data"):
        home = os.getenv("HOME")
        copy_tree('../data', str(home) + "/ocvf_data/")
    else:
        pass