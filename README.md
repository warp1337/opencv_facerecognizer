![ocvf example output](doc/images/ocvf_example.jpg)

OpenCV FaceRecognizer
=======================

First things first: this software package is based on the great work of Philipp Wagner [1]. However,
Norman Koester and me (Florian Lier) created this package in order to provide a **distributed** and **dynamic**
on-the-fly learning approach to OpenCV based face detection and **recognition** (learning) using current robotics middleware
implementations and standardized installation and roll-out routines, as well as a jump start training set. 
At the time of writing RSB [2] and ROS [3] are supported.

* Major Changes
    * Decoupling and modularization of Py-Packages
    * On-the-fly re-training (learning) of individuals
    * On-the-fly recognizer restart using an updated model
    * Setuptools support for ease of installation
    * ROS and RSB middleware support
    * Typed messages (middleware specific)
    * Distributed camera streams
    * Published classification results (middleware specific)
    * Convenience Tools (face cropper etc.)

This documentation is *minimalistic*, which means it provides basic information on how to train
a model and run this software stack. If you need detailed information about the _internals_
please consult [4][5].

* [1] http://bytefish.de
* [2] https://code.cor-lab.org/projects/rsb
* [3] http://www.ros.org/
* [4] http://bytefish.de/blog/videofacerec/
* [5] http://www.bytefish.de/blog/fisherfaces/


Architecture Overview
----------------------

    TODO


Installation
-------------

For the live mode (cf. ROSBAG) you will need a webcam with a minimal resolution of 640x480 pixel. We have tested this package 
with *Ubuntu* 14.04 and 14.10 using *ROS* Indigo and *RSB* 0.11

Minimal Dependencies:

    sudo apt-get install python-dev python python-scipy python-imaging-* python-opencv python-setuptools

The most basic application, **ocvf_recognizer.py** will work without *ROS* or *RSB*. However, if you want to
make use of typed messages and on-the-fly model training, please install one of the following (RSB or ROS).

Dependencies ROS (Indigo):

    sudo apt-get ros-indigo-desktop ros-indigo-people-msgs ros-indigo-usb-cam

Dependencies ROS (Groovy):

    sudo apt-get ros-groovy-desktop ros-groovy-people-msgs ros-groovy-usb-cam

*Hint*: You might save some disk space by installing ros-$version-base. We haven't actually
checked if they contain all required packages. If this is not the case, you need to install 
missing packages manually.


If you are not familiar with ROS please visit the [ROS Installation Instructions](http://wiki.ros.org/indigo/Installation/Ubuntu)

Dependencies RSB:

You will need *RSC*, *RST* and *RSB* as well as the RSB-Python implementation and RSB-OpenCV

    http://docs.cor-lab.de/rsb-manual/trunk/html/index.html
    https://code.cor-lab.org/projects/rsbvideoreceiver


Installing OCVF Package (sudo)
-------------------------------

    mkdir -p ~/ocvfacerecognizer && cd ~/ocvfacerecognizer
    git clone https://github.com/warp1337/opencv_facerecognizer.git .
    cd src
    sudo python setup.py install


Installing OCVF Package (non-sudo)
------------------------------------------------------
    
    mkdir -p ~/custom-prefix/lib/python2.7/site-packages/
    mkdir -p ~/ocvfacerecognizer && cd ~/ocvfacerecognizer
    git clone https://github.com/warp1337/opencv_facerecognizer.git .
    cd src
    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    python setup.py install --prefix=~/custom-prefix/


Jump Start Basic
-----------------

In case you just want to try this out, execute the following assuming you installed OCVF correctly (non-sudo variant)
as described above.

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    python ~/custom-prefix/bin/ocvf_recognizer.py -c ~/ocvfacerecognizer/data/haarcascade_frontalface_alt2.xml ~/ocvfacerecognizer/data/individuals.pkl
    
Now point your camera at this [image](http://de.wikipedia.org/wiki/Linus_Torvalds#mediaviewer/File:Linus_Torvalds.jpeg) (Yes! Point it at the screen...)
The recognizer should detect **"linus"**

     
Jump Start ROS + ROSBAG
------------------------

We have pre-recorded a set of images. You can replay this set and watch the recognizer do its work... 

First Terminal (RECOGNIZER)

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ~/custom-prefix/bin/ocvf_recognizer_ros.py -c ~/ocvfacerecognizer/data/haarcascade_frontalface_alt2.xml -s /ocvfacerec/static_image ~/ocvfacerecognizer/data/individuals.pkl

Second Terminal (BAG PLAY)

    source /opt/ros/indigo/setup.bash
    roscore &
    rosbag play -l ~/ocvfacerecognizer/data/individuals.bag

Just have a look at the recognizer output and enjoy ;)


Jump Start ROS + On-The-Fly Training
-------------------------------------

Here's the ROS Jump Start including on-the-fly training (non-sudo variant).

First Terminal (CAM SOURCE)
    
    source /opt/ros/indigo/setup.bash
    roslaunch ~/ocvfacerecognizer/ros_cam_node/ros_cam_node.launch
    
Second Terminal (RECOGNIZER)

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ~/custom-prefix/bin/ocvf_recognizer_ros.py -c ~/ocvfacerecognizer/data/haarcascade_frontalface_alt2.xml ~/ocvfacerecognizer/data/individuals.pkl

Third Terminal (INTERACTIVE TRAINER)

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ~/custom-prefix/bin/ocvf_interactive_trainer.py -c ~/ocvfacerecognizer/data/haarcascade_frontalface_alt2.xml -w ros -t ~/ocvfacerecognizer/data/individuals -s /usb_cam/image_raw ~/ocvfacerecognizer/data/individuals.pkl    

Fourth Terminal (TRIGGER)

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ~/custom-prefix/bin/ocvf_retrain_ros.py YOUR_NAME   

At this point the Trainer records 70 images (of you), updates the model and then restarts the recognizer. You should now
be detected in the OCVF window! Yeah.


Jump Start RSB + On-The-Fly Training
-------------------------------------

And finally, here's the RSB Jump Start including on-the-fly training (non-sudo variant).

First Terminal (CAM SOURCE)
    
    rsb_videosender -o /rsbopencv/ipl
    
Second Terminal (RECOGNIZER)

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    python ~/custom-prefix/bin/ocvf_recognizer_rsb.py -c ~/ocvfacerecognizer/data/haarcascade_frontalface_alt2.xml ~/ocvfacerecognizer/data/individuals.pkl

Third Terminal (INTERACTIVE TRAINER)

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    python ~/custom-prefix/bin/ocvf_interactive_trainer.py -c ~/ocvfacerecognizer/data/haarcascade_frontalface_alt2.xml -w rsb -t ~/ocvfacerecognizer/data/individuals -s /rsbopencv/ipl ~/ocvfacerecognizer/data/individuals.pkl    

Fourth Terminal (TRIGGER)

    export PYTHONPATH=~/custom-prefix/lib/python2.7/site-packages:$PYTHONPATH
    python ~/custom-prefix/bin/ocvf_retrain_rsb.py YOUR_NAME   

At this point the Trainer records 70 images (of you), updates the model and then restarts the recognizer. You should now
be detected in the OCVF window! Yeah.


Creating a Training Set from Images
------------------------------------

Download or record a set of images you want to train you model with, at least two classes/persons are required. Save these files in separate folders (see below).
If you don't feel like assembling images yourself, you can also start with the [AT&T](http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html)
database that already contains 40 individuals. Another alternative is to use the provided data set including 4 well-known individuals. The data set can
be found in the data folder.


    ../data/individuals/
        ├── bill
        │   ├── bill_crop0.jpg
        │   ├── bill_crop1.jpg
        │   ├── bill_crop2.jpg
        │   ├── bill_crop3.jpg
        │   └── ...
        ├── dennis
        │   ├── dennis_crop0.jpg
        │   ├── dennis_crop1.jpg
        │   ├── dennis_crop2.jpg
        │   ├── dennis_crop3.jpg
        │   └── ...
        ├── linus
        │   ├── linus_crop0.jpg
        │   ├── linus_crop0.jpg
        │   ├── ...


If you'd like to use your own data set (not the case for AT&T and the included data set) you need to invoke the **face_cropper.py** tool that resides 
in the tools folder.

    python face_cropper.py <label> </path/to/images> <haarcascade.file>
    
    Example: python face_cropper.py adam_sandler /tmp/my_images/adam_sandler /homes/flier/dev/ocvfacerec/data/haarcascade_frontalface_alt2.xml


After successful execution you will end up with a separate *cropped* folder in each *person's* folder.
You need to copy/move the person/**cropped** folder to a another location and rename the folders according
to the desired label, person respectively.

        my_cropped_images/
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


Now it is time to train your model. Hint: $YOUR_PREFIX is the location where OCVF has been installed
If you haven't trained a model yet, which might be the case, the model.pkl file it will be created in the following step. 


Basic Usage Training
---------------------

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_recognizer.py -c </path/to/cascade.xml> -t </path/to/cropped_images/> -v 10 </path/to/model.pkl>


Basic Usage Recognition
------------------------

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_recognizer.py -c </path/to/cascade.xml> </path/to/model.pkl>


Distributed Recognition ROS
----------------------------

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    roslaunch ~/ocvfacerecognizer/data/ros_cam_node.launch &
    python ocvf_recognizer_ros.py -c </path/to/cascade.xml> </path/to/model.pkl>


Distributed Recognition RSB
----------------------------

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    rsb_videosender -o /rsbopencv/ipl
    python ocvf_recognizer_rsb.py -c </path/to/cascade.xml> </path/to/model.pkl>


Distributed Model Training ROS
-------------------------------

Distributed or on-the-fly training means you can either re-train an individual on-the-fly, or add more subjects to your
model. You just need to follow the instructions below. It is assumed you have assembled a set of cropped images as introduced above.

In the first Terminal (Interactive Trainer)

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    roslaunch ~/ocvfacerecognizer/data/ros_cam_node.launch &
    python ocvf_interactive_trainer.py -c </path/to/cascade.xml> --image-source=/usb_cam/image_raw --middleware=ros -t </path/to/cropped_images/> </path/to/model.pkl>

In another Terminal (Recognizer)

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ocvf_recognizer_ros.py -c </path/to/cascade.xml> </path/to/model.pkl>

In yet another Terminal (Training Trigger)

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    source /opt/ros/indigo/setup.bash
    python ocvf_retrain_ros.py <person_label>
    
    Example: python ocvf_retrain_ros.py florian

Now you should see (in the Interactive Trainer Terminal) that new images are recorded, 70 by default, and the recognizer
is restarted (in the Recognizer Terminal) using the newly updated model file as soon as the training is done.
You just successfully updated/added a person in/to your model! Much wow!


Distributed Model Training RSB
-------------------------------

Distributed or on-the-fly training means you can either re-train an individual on-the-fly, or add more subjects to your
model. You just need to follow the instructions below. It is assumed you have assembled a set of cropped images as introduced above.

In the first Terminal (Interactive Trainer)

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    rsb_videosender -o /rsbopencv/ipl
    python ocvf_interactive_trainer.py -c </path/to/cascade.xml> -t </path/to/cropped_images/> </path/to/model.pkl>

In another Terminal (Recognizer)

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_recognizer_rsb.py -c </path/to/cascade.xml> </path/to/model.pkl>

In yet another Terminal (Training Trigger)

    export PATH=$YOUR_PREFIX/bin:$PATH
    export PYTHONPATH=$YOUR_PREFIX/lib/python2.7/site-packages:$PYTHONPATH
    python ocvf_retrain_rsb.py <person_label>
    
    Example: python ocvf_retrain_rsb.py florian

Now you should see (in the Interactive Trainer Terminal) that new images are recorded, 70 by default, and the recognizer
is restarted (in the Recognizer Terminal) using the newly updated model file as soon as the training is done.
You just successfully updated/added a person in/to your model! Much wow!


View Published Messages ROS
----------------------------

    source /opt/ros/indigo/setup.bash
    rostopic echo /ocvfacerec/ros/people


View Published Messages RSB
----------------------------

    TODO


Replication
-------------

    TODO


LICENSE
-------------


    Copyright (c) 2015.
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


AFFILIATION
------------

This work is supported by CoR-Lab and CITEC

Florian Lier is currently with: [CITEC](http://www.cit-ec.de)

Norman Koester is currently with: [CITEC](http://www.cit-ec.de) and [CoR-Lab](http://www.cor-lab.de)