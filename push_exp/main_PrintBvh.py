import os
import numpy as np
import time

import csv
import socket
import datetime
import math
import glob
from scipy.spatial.transform import Rotation

from pypushexp import PushSim


if __name__ == '__main__':
    import sys
    import re

    if len(sys.argv) > 1:
        option = sys.argv[1]
        trial_angle = sys.argv[2]
    else:
        option = 'torque_push_both_sf_crouch30_uniform'
        option = 'torque_nopush_sf_crouch30_uniform'
        trial_angle = '30'

    _metadata_dir = os.path.dirname(os.path.abspath(__file__)) + '/../data/metadata/'
    _nn_finding_dir = os.path.dirname(os.path.abspath(__file__)) + '/../nn/don2/'

    nn_dirs = glob.glob(_nn_finding_dir + option)
    if len(nn_dirs) == 0:
        _nn_finding_dir = os.path.dirname(os.path.abspath(__file__)) + '/../nn/done/'
        nn_dirs = glob.glob(_nn_finding_dir + option)

    nn_dir = nn_dirs[0]
    meta_file = _metadata_dir + option + '.txt'

    _sim = None
    if 'muscle' in option:
        _sim = PushSim(meta_file, nn_dir+'/max.pt', nn_dir+'/max_muscle.pt')
    else:
        _sim = PushSim(meta_file, nn_dir+'/max.pt')

    push_step = 8
    push_duration = 0.2
    push_force = 150.
    push_start_timing = 13

    crouch_angle = int(trial_angle)
    step_length = 0.9158506655
    walk_speed = 0.7880050552

    # step length
    motion_stride_bvh_after_default_param = 1.1886
    step_length_ratio = step_length / motion_stride_bvh_after_default_param

    # walk speed
    speed_bvh_after_default_param = 0.9134
    walk_speed_ratio = walk_speed / speed_bvh_after_default_param

    _sim.setParamedStepParams(int(crouch_angle), step_length_ratio, walk_speed_ratio)
    _sim.setPushParams(push_step, push_duration, -push_force, push_start_timing)

    stopcode = _sim.simulate()
    motion_skel = _sim.getMotions()

    motion = []
    for i in range(len(motion_skel)):
        motion.append(np.zeros(len(motion_skel[i])))
        motion[i][:3] = motion_skel[i][:3]
        for j in range(3, len(motion_skel[i]), 3):
            motion[i][j:j+3] = Rotation.from_rotvec(motion_skel[i][j:j+3]).as_euler('ZXY', degrees=True)
    print(_sim.getPushStartFrame(), _sim.getPushEndFrame())

    with open('/Users/trif/works/ProjectPushRecovery/result_motion/push_150N_13t.bvh', 'w') as fout:
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../data/motion/bvh_base.bvh', 'r') as f_base:
            fout.write(f_base.read())
        fout.write('MOTION\n')
        fout.write('Frames: '+str(len(motion))+'\n')
        fout.write('Frame Time: 0.033333333333\n')
        for i in range(len(motion)):
            fout.write(' '.join(list(map(str, motion[i]))) + '\n')


