OpenCV FaceRecognizer
=======================

First things first: this software package is based on the great work of Philipp Wagner [1]. However,
Norman Koester and I (Florian Lier) created this package in order to provide a distributed and dynamic
learning approach to OpenCV based face recognition (and detection) using current robotics middleware
implementations and standardized installation and roll-out routines. At the time of writing
RSB [2] and ROS [3] are supported.

* Changes
    * Decoupling and modularization of Py-Packages
    * Setuptools support
    * ROS and RSB middleware support
    * Typed message passing
    * "Live" re-training of individuals
    * "Live" recognizer restart using an updated model
    * Distributed camera streams
    * Publishing of classification results
    * Convenience Tools (cropper etc.)

This documentation is *minimalistic*, which means it provides basic information on how to train
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
make use of typed message passing and on-the-fly model training please install one of the following (RSB or ROS).

Dependencies ROS (Indigo):

    sudo apt-get ros-indigo-desktop ros-indigo-people-msgs ros-indigo-usb-cam

Dependencies ROS (Groovy):

    sudo apt-get ros-groovy-desktop ros-groovy-people-msgs ros-groovy-usb-cam

*Hint*: You might save some disk space by installing ros-[*indigo*|*groovy*]-base. We haven't actually
checked if they contain all required packages. If not, you need to install missing packages manually, obviously.

[General ROS Install Instructions](http://wiki.ros.org/indigo/Installation/Ubuntu)

Dependencies RSB:

    RSC, RSB, RST http://docs.cor-lab.de/rsb-manual/trunk/html/index.html
    RSB OpenCV https://code.cor-lab.org/projects/rsbvideoreceiver


Installing Package (sudo)
-------------------

    mkdir -p ~/ocvfacerecognizer && cd ~/ocvfacerecognizer
    git clone https://github.com/warp1337/opencv_facerecognizer.git .
    cd src
    sudo python setup.py install


Installing Package Custom Prefix (non sudo)
--------------------------------------------

    mkdir -p ~/ocvfacerecognizer && cd ~/ocvfacerecognizer
    git clone https://github.com/warp1337/opencv_facerecognizer.git .
    cd src
    mkdir -p ~/custom-prefix/lib/python2.7/site-packages
    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    python setup.py install --prefix=~/custom-prefix/


Creating a Training Set from Images
------------------------------------

Download or record a set of images you want to train you model with (at least two classes/persons). Save these files in separate folders (see below).
If you don't feel like assembling all the images yourself, you can also start with the [AT&T](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
database that already contains 40 individuals.


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


Now invoke the **face_cropper.py** tool that resides in the tools folder. Hint: The AT&T database is already cropped you can skip this step...

    python face_cropper.py <label> </path/to/images> <haarcascade.file>
    Example: python face_cropper.py Adam_Sandler /tmp/selected/adam_sandler /homes/flier/dev/ocvfacerec/data/haarcascade_frontalface_alt2.xml


After successful execution you will end up with a separate *cropped* folder in each *person's* folder.
You need to copy/move the person/*cropped* folder to a different location and rename the folders according
to the desired label, person respectively.

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


Now it is time to train your model.


Basic Usage Training
---------------------

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_recognizer.py -c </path/to/cascade.xml> -t </path/to/images/> -v 10 </path/to/model.pkl>


Basic Usage Recognition
------------------------

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_recognizer.py -c </path/to/cascade.xml> </path/to/model.pkl>


Distributed Recognition ROS
----------------------------

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    roslaunch ~/ocvfacerecognizer/data/ros_cam_node.launch &
    python ocvf_recognizer_ros.py -c </path/to/cascade.xml> </path/to/model.pkl>


Distributed Recognition RSB
----------------------------

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    rsb_videosender -o /rsbopencv/ipl
    python ocvf_recognizer_rsb.py -c </path/to/cascade.xml> </path/to/model.pkl>


Distributed Model Training ROS
-------------------------------

Distributed, or live training means you can either re-train an individual on the fly, or add more subjects to your
model. You just need to follow the instructions below.

In the first Terminal (Interactive Trainer)

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    roslaunch ~/ocvfacerecognizer/data/ros_cam_node.launch &
    python ocvf_interactive_trainer.py -c </path/to/cascade.xml> --image-source=/usb_cam/image_raw --middleware=ros </path/to/model.pkl>

In another Terminal (Recognizer)

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ocvf_recognizer_ros.py -c </path/to/cascade.xml> </path/to/model.pkl>

In yet another Terminal (Training Trigger)

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ocvf_retrain_ros.py <person_label>
    Example: python ocvf_retrain_ros.py florian

Now you should see (in the Interactive Trainer Terminal) that new images are recorded, 70 by default, and the recognizer
is restarted (in the Recognizer Terminal) using the updated model file as soon as the training is done.
You just successfully updated/added a person in/to your model! Much wow!


Distributed Model Training RSB
-------------------------------

In the first Terminal (Interactive Trainer)

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    rsb_videosender -o /rsbopencv/ipl
    python ocvf_interactive_trainer.py -c </path/to/cascade.xml> </path/to/model.pkl>

In another Terminal (Recognizer)

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_recognizer_rsb.py -c </path/to/cascade.xml> </path/to/model.pkl>

In yet another Terminal (Training Trigger)

    export PATH=YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_retrain_rsb.py <person_label>
    Example: python ocvf_retrain_rsb.py florian

Now you should see (in the Interactive Trainer Terminal) that new images are recorded, 70 by default, and the recognizer
is restarted (in the Recognizer Terminal) using the updated model file as soon as the training is done.
You just successfully updated/added a person in/to your model! Much wow!


Replication
-------------

    TODO


LICENSE
-------------


    Copyright (c) 2012.
    Philipp Wagner <bytefish[at]gmx[dot]de>
    Florian Lier <flier[at]techfak.uni-bielefeld.de>
    Norman Koester <nkoester[at]techfak.uni-bielefeld.de>

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