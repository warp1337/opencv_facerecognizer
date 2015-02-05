OpenCV FaceRecognizer
=======================

Installation
-------------

Dependencies ROS:
	sudo apt-get install ros-indigo-people-msgs ros-indigo-usb-cam ros-indigo-desktop

Dependencies RSB:

cd /tmp
git pull XXX
cd XXX
export prefix=/tmp/sandbox/ocvf/
mkdir -p $prefix/lib/python2.7/site-packages/
export PYTHONPATH=/tmp/sandbox/ocvf/:$PYTHONPATH
python setup.py install --prefix=$prefix

Usage Learning
---------------
python $prefix/bin/ocvf_recognizer.py -c /path/to/haarcascade_frontalface_alt2.xml -t /path/to/images/ my_model.pkl


Usage Recognition
-----
python $prefix/bin/ocvf_recognizer.py -c /path/to/haarcascade_frontalface_alt2.xml path/to/model/celebs.pkl

