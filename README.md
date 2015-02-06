OpenCV FaceRecognizer
=======================

First things first: this software package is based on the great work of Philipp Wagner [1]. However,
Norman Koester and I (Florian Lier) created this package in order to provide a more distributed
approach to OpenCV based face recognition (and detection) using current robotics middleware
implementations and standardized installation and roll-out routines. At the time of writing
RSB [2] and ROS [3] are supported.

This documentation is *minimalitic*, which means it provides basic information on how to train
a model and run this software stack. If you need detailed information about the _internals_
please consult [4][5].

* [1] http://bytefish.de
* [2] https://code.cor-lab.org/projects/rsb
* [3] http://www.ros.org/
* [4] http://bytefish.de/blog/videofacerec/
* [5] http://www.bytefish.de/blog/fisherfaces/


Installation
-------------

Minimal Dependencies:

    sudo apt-get install python-dev python python-scipy python-imaging-* python-opencv python-setuptools

The most basic application, **ocvf_recognizer.py** will work without *ROS* or *RSB*. However, if you want to
make use of typed message passing and RPC calls please install the following [ROS Install Instructions](http://wiki.ros.org/indigo/Installation/Ubuntu)

Dependencies ROS (Indigo):

    sudo apt-get ros-indigo-desktop ros-indigo-people-msgs ros-indigo-usb-cam

Dependencies ROS (Groovy):

    sudo apt-get ros-groovy-desktop ros-groovy-people-msgs ros-groovy-usb-cam

*Hint*: You might save some disk space by installing ros-[*indigo*|*groovy*]-base. We haven't actually
checked if they contain all required packages. If so, you need to install missing packages manually.

Dependencies RSB:

    TODO


Installing Package
-------------------

    mkdir -p ~/ocvfacerecognizer && cd ~/ocvfacerecognizer
    git clone https://github.com/warp1337/opencv_facerecognizer.git .
    cd src
    python setup.py install


Installing Package Custom Prefix
---------------------------------

    mkdir -p ~/ocvfacerecognizer && cd ~/ocvfacerecognizer
    git clone https://github.com/warp1337/opencv_facerecognizer.git .
    cd src
    mkdir -p /tmp/custom-prefix/lib/python2.7/site-packages
    export PYTHONPATH=/tmp/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    python setup.py install --prefix=/tmp/custom-prefix/


Usage Training
---------------

    python ocvf_recognizer.py -c </path/to/cascade.xml> -t </path/to/images/> -v 10 </path/to/model.pkl> -r 70x70


Usage Recognition Simple
-------------------------

    python ocvf_recognizer.py -c </path/to/cascade.xml> </path/to/model.pkl>


Usage Recognition ROS
----------------------

    source /opt/ros/indigo/setup.bash
    python ocvf_recognizer_ros.py -c </path/to/cascade.xml> </path/to/model.pkl> --ros-source </input/topic>


Usage Recognition RSB
----------------------

    python ocvf_recognizer_rsb.py -c </path/to/cascade.xml> </path/to/model.pkl> --rsb-source </input/scope>



Usage Create Training Set
--------------------------
Download or record a set of images you want to train you model with. Save these files in separate folders like:

    my_images/
        ├── adam_sandler
        │   ├── adam_sandlerasd0.jpg
        │   ├── Sandleranm3bn23.jpg
        │   ├── 234k34lk55kl3.jpg
        │   ├── foobaz_sandler.jpg
        │   └── ...
        ├── alfred_molina
        │   ├── m00wlina.jpg
        │   ├── Alfred_Molina23.jpg
        │   ├── 1337Molina.jpg
        │   ├── alfred.jpg
        │   ├── Molina4.jpg
        │   └── ...
        ├── bratt_pitt
        │   ├── bratt.jpg
        │   ├── branjolina.jpg
        │   ├── ...


Now invoke the **face_cropper.py** tool that resides in the tools folder.

    python face_cropper.py <label> </path/to/images> <haarcascade.file>
    python face_cropper.py Adam_Sandler /tmp/selected/adam_sandler /homes/flier/dev/ocvfacerec/data/haarcascade_frontalface_alt2.xml


After successful execution you will end up with a separate *cropped* folder in each *person's* folder.
You need to copy/move the person/*cropped* folder to a different location and rename the folders according
to the desired label, person respectively. All this should look like this:


        celeb_database/
        ├── adam_sandler
        │   ├── Adam_Sandler_crop0.jpg
        │   ├── Adam_Sandler_crop10.jpg
        │   ├── Adam_Sandler_crop12.jpg
        │   ├── Adam_Sandler_crop1.jpg
        │   ├── Adam_Sandler_crop3.jpg
        │   ├── Adam_Sandler_crop4.jpg
        │   ├── Adam_Sandler_crop5.jpg
        │   ├── Adam_Sandler_crop7.jpg
        │   ├── Adam_Sandler_crop8.jpg
        │   └── Adam_Sandler_crop9.jpg
        ├── alfred_molina
        │   ├── Alfred_Molina_crop0.jpg
        │   ├── Alfred_Molina_crop1.jpg
        │   ├── Alfred_Molina_crop2.jpg
        │   ├── Alfred_Molina_crop3.jpg
        │   ├── Alfred_Molina_crop4.jpg
        │   ├── Alfred_Molina_crop5.jpg
        │   ├── Alfred_Molina_crop6.jpg
        │   ├── Alfred_Molina_crop7.jpg
        │   ├── Alfred_Molina_crop8.jpg
        │   └── Alfred_Molina_crop9.jpg



Replication
-------------
TODO


LICENSE
-------------


    Copyright (c) 2012.
    Philipp Wagner <bytefish[at]gmx[dot]de>
    Norman Koester <nkoester[at]techfak.uni-bielefeld.de>
    Florian Lier <flier[at]techfak.uni-bielefeld.de>

    Released to public domain under terms of the BSD Simplified license.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the organization nor the names of its contributors
      may be used to endorse or promote products derived from this software
      without specific prior written permission.

    See <http://www.opensource.org/licenses/bsd-license>

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
    "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
    LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
    FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
    COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
    INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
    BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
    LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
    CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
    LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
    ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
    POSSIBILITY OF SUCH DAMAGE.