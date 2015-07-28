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
import copy

# RSB Specifics
import rsb
import rst
import rstsandbox
from rsb.converter import ProtocolBufferConverter
from rsb.converter import registerGlobalConverter
from rstconverters.opencv import IplimageConverter

from rst.communicationpatterns.TaskState_pb2 import TaskState

# OCVF Imports
from ocvfacerec.mwconnector.abtractconnector import MiddlewareConnector
from rst.communicationpatterns import TaskState_pb2
from rst.communicationpatterns.TaskState_pb2 import TaskState


class RSBConnector(MiddlewareConnector):

    def __init__(self, socket_mode=False):
        self.socket_mode = socket_mode

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

    def get_modified_participantconfig(self):

        # get current config as a copy
        config = copy.deepcopy(rsb.getDefaultParticipantConfig())
        transports = config.getTransports(includeDisabled=True)

        # modify it
        for aTransport in transports:
            # is this the transport we're looking for?
            if aTransport.name is "socket":
                #modify desired transport here ...
                aTransport.enabled = True
            else:
                aTransport.enabled = False

        # all done
        return config

    def activate(self, image_source, retrain_source, restart_target):
        # In order to convert the images
        registerGlobalConverter(IplimageConverter())
        rsb.setDefaultParticipantConfig(rsb.ParticipantConfig.fromDefaultSources())

        # Listen to Image Events
        if self.socket_mode:
            print ">> WARN: receiving images from SOCKET!"
            self.image_listener = rsb.createListener(image_source, config=self.get_modified_participantconfig())
        else:
            self.image_listener = rsb.createListener(image_source)


        self.lastImage = Queue(1)
        self.image_listener.addHandler(self.add_last_image)

        # To parse TaskState events
        converter = rsb.converter.ProtocolBufferConverter(messageClass=rst.communicationpatterns.TaskState_pb2.TaskState)
        rsb.converter.registerGlobalConverter(converter)
        # Listen to Re-Train events with a Person Label
        self.training_start_listener = rsb.createListener(retrain_source)
        self.last_train = Queue(100)
        self.training_start_listener.addHandler(self.add_last_train)

        # Publish updated events
        self.training_update_publisher = rsb.createInformer(retrain_source, dataType=TaskState)

        # Publisher to Restart Recognizer
        self.restart_publisher = rsb.createInformer(restart_target, dataType=str)

    def training_accept(self):
        self.training_update_publisher.publishData()

    def deactivate(self):
        self.image_listener.deactivate()
        self.training_start_listener.deactivate()
        self.training_update_publisher.deactivate()
        self.restart_publisher.deactivate()

    def restart_classifier(self):
        # Send a short "restart" event to the recognizer
        self.restart_publisher.publishData("restart")

    def get_image(self):
        return self.lastImage.get(True, timeout=10)

    def wait_for_start_training(self):
        is_new_task = False

        while not is_new_task:
            new_task = self.last_train.get(True, timeout=1)
            assert isinstance(new_task, TaskState)
            if new_task.state is TaskState_pb2.TaskState.INITIATED and\
               new_task.origin is TaskState_pb2.TaskState.SUBMITTER:
                is_new_task = True

        # update task description
        new_task.origin = TaskState_pb2.TaskState.HANDLER
        new_task.state = TaskState_pb2.TaskState.ACCEPTED
        new_task.serial += 1

        # extract name
        new_name = new_task.payload

        new_task.payload = ""
        # accept task
        self.training_update_publisher.publishData(new_task)
        self.last_task = new_task
        return new_name

    def update_last_task_status(self, percent):
        self.last_task.state = TaskState_pb2.TaskState.UPDATE
        self.last_task.serial += 1
        self.last_task.payload = str(percent)
        self.training_update_publisher.publishData(self.last_task)

    def fail_last_task_status(self, reason=""):
        self.last_task.state = TaskState_pb2.TaskState.FAILED
        self.last_task.serial += 1
        self.last_task.payload = reason
        self.training_update_publisher.publishData(self.last_task)

    def complete_last_task_status(self):
        self.last_task.state = TaskState_pb2.TaskState.COMPLETED
        self.last_task.serial += 1
        self.training_update_publisher.publishData(self.last_task)
