from pushsim import PushSim
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


if __name__ == '__main__':
    import os
    _metadata_dir = os.path.dirname(os.path.abspath(__file__)) + '/../data/metadata/'
    _nn_finding_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../*/nn/*/'

    _is_muscle = True
    _is_pushed_during_training = False
    _is_multi_seg_foot = False
    _is_walking_variance = True
    _is_walking_param_normal_trained = False
    _crouch = input('crouch angle(0, 20, 30, 60, all)? ')
    _params = (_is_muscle, _is_pushed_during_training, _is_multi_seg_foot, _is_walking_variance, _is_walking_param_normal_trained, _crouch)

    sim = PushSim(_params, _metadata_dir, _nn_finding_dir)
