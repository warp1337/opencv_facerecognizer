'''
Created on Feb 9, 2015

@author: nkoester
'''

class PersonWrapper(object):
    '''
    Simple class representing a person
    '''

    position = None
    name = None
    reliability = None

    def __init__(self, _position, _name, _reliability):
        '''
        Constructor.
        '''
        #bounding box of the person
        self.x0, self.y0, self.x1, self.y1 = _position
        self.position = self._person_center()
        self.name = _name
        self.reliability = _reliability
        self._import_done = False

    def _person_center(self):
        mid_x = float(self.x1 + (self.x1 - self.x0) * 0.5)
        mid_y = float(self.y1 + (self.y1 - self.y0) * 0.5)
        mid_z = float(self.x1 - self.x0)
        self.position = mid_x, mid_y, mid_z
        return (mid_x, mid_y, mid_z)

    def to_ros_msg(self):
        raise Exception("Not implemented yet.")

    def to_rsb_msg(self):
        # TODO: export the middleware knowledge to this class -> further
        # encapsulation of middleware && less duplicated code.
#         if not self._import_done:
#             from rst.classification.ClassificationResult_pb2 import ClassificationResult
#             self._import_done = True
#
#         person = ClassificationResult()
        raise Exception("Not implemented yet.")

