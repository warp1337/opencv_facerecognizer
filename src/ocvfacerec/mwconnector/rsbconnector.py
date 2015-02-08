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
import numpy as np
from Queue import Queue

# RSB Specifics
import rsb
import rstsandbox
from rsb.converter import ProtocolBufferConverter
from rsb.converter import registerGlobalConverter
from rstconverters.opencv import IplimageConverter

# OCVF Imports
from ocvfacerec.mwconnector import MiddlewareConnector


class RSBConnector(MiddlewareConnector):
    def __init__(self):
        pass

    def add_last_image(self, image_event):
        try:
            self.lastImage.get(False)
        except Exception, e:
            pass
        self.lastImage.put(np.asarray(image_event.data[:, :]), False)

    def add_last_train(self, retrain_event):
        try:
            self.last_train.get(False)
        except Exception, e:
            pass
        self.last_train.put(retrain_event.data, False)

    def activate(self, image_source, retrain_source, restart_target):
        registerGlobalConverter(IplimageConverter())
        rsb.setDefaultParticipantConfig(rsb.ParticipantConfig.fromDefaultSources())

        # Listen to image events
        self.image_listener = rsb.createListener(image_source)
        self.lastImage = Queue(1)
        self.image_listener.addHandler(self.add_last_image)

        # Listen to re-train events with a name
        self.training_start = rsb.createListener(retrain_source)
        self.last_train = Queue(10)
        self.training_start.addHandler(self.add_last_train)

        # Publisher to restart recogniser
        self.restart_publisher = rsb.createInformer(restart_target, dataType=str)

    def deactivate(self):
        self.image_listener.deactivate()
        self.training_start.deactivate()
        self.restart_publisher.deactivate()

    def restart_classifier(self):
        # Send a short "restart" event to the recognizer
        self.restart_publisher.publishData("restart")

    def wait_for_start_training(self):
        return self.last_train.get(True, timeout=1)

    def get_image(self):
        return self.lastImage.get(True, timeout=10)