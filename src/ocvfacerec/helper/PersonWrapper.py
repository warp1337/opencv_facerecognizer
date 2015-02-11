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

    def __init__(self, _position, _name, _reliability, _image_size):
        '''
        Constructor.
        '''
        #bounding box of the person
        self.x0, self.y0, self.x1, self.y1 = _position
        self.position = self._person_center()
        self.name = _name
        self.reliability = _reliability
        self.image_width = _image_size[0]
        self.image_height = _image_size[1]
        self._import_done = False

    def _person_center(self):
        mid_x = float(self.x1 + (self.x1 - self.x0) * 0.5)
        mid_y = float(self.y1 + (self.y1 - self.y0) * 0.5)
        mid_z = float(self.x1 - self.x0)
        self.position = mid_x, mid_y, mid_z
        return (mid_x, mid_y, mid_z)

    def to_ros_msg(self):
        # TODO: export the middleware knowledge to this class -> further
        # encapsulation of middleware && less duplicated code.
        raise Exception("Not implemented yet.")

    def to_rsb_msg(self):
        # Creates a RST representation of this person
        # The RST datatype used here:
        #     [* HeadObjects]
        #     ** (repeated) HeadObject
        #     *** LabledFace
        #     **** Name
        #     **** Face
        #     ***** Confidence
        #     ***** BoundingBox

        from rstsandbox.vision.HeadObject_pb2 import HeadObject
        from rst.vision.Face_pb2 import Face
        from rst.geometry.BoundingBox_pb2 import BoundingBox
        from rst.math.Vec2DInt_pb2 import Vec2DInt

        new_head = HeadObject()
        new_labled_face = new_head.faces.add()
        new_labled_face.label = self.name
        new_labled_face.face.confidence = self.reliability
        new_labled_face.face.region.top_left.x = int(self.x0)
        new_labled_face.face.region.top_left.y = int(self.y0)
        new_labled_face.face.region.width = int(self.x1 - self.x0)
        new_labled_face.face.region.height = int(self.y1 - self.y0)
        new_labled_face.face.region.image_width = int(self.image_width)
        new_labled_face.face.region.image_height = int(self.image_height)

        return new_head
