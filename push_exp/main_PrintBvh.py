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


def getMotionFromMotionSkel(_motion_skel):
    _motion = []
    for _i in range(len(_motion_skel)):
        _motion.append(np.zeros(len(_motion_skel[_i])))
        _motion[_i][:3] = _motion_skel[_i][:3]
        for _j in range(3, len(_motion_skel[_i]), 3):
            _motion[_i][_j:_j+3] = Rotation.from_rotvec(np.asarray(_motion_skel[_i][_j:_j+3])).as_euler('ZXY', degrees=True)
    return _motion


if __name__ == '__main__':
    # for stand push
    path_prefix = '/Users/trif/works/ProjectPushRecovery/result_motion/'
    filename = 'interactive_result/figure_bvh.txt'
    skel_motion = []
    with open(path_prefix + filename, 'r') as f:
        s = f.readline()
        while s != '':
            skel_motion.append(list(map(float, s.split())))
            s = f.readline()

    motion_ref = getMotionFromMotionSkel(skel_motion)

    filename_ref = 'interactive_result/figure.bvh'

    with open(path_prefix+filename_ref, 'w') as fout:
        with open(os.path.dirname(os.path.abspath(__file__)) + '/../data/motion/bvh_base.bvh', 'r') as f_base:
            fout.write(f_base.read())
        fout.write('MOTION\n')
        fout.write('Frames: '+str(len(motion_ref))+'\n')
        fout.write('Frame Time: 0.033333333333\n')
        for i in range(len(motion_ref)):
            fout.write(' '.join(list(map(str, motion_ref[i]))) + '\n')

if __name__ == '__main__' and False:
    import sys
    import re

    path_prefix = '/Users/trif/works/ProjectPushRecovery/result_motion/'

    if len(sys.argv) > 1:
        option = sys.argv[1]
        trial_angle = sys.argv[2]
    else:
        trial_angle = str(0)
        # option = 'torque_push_both_sf_all_adaptive_k1_depth3'
        # option = 'torque_nopush_sf_all_adaptive_k1_depth3'
        # option = 'torque_push_both_sf_all_uniform_depth3'
        option = 'torque_nopush_sf_all_uniform_depth3'

    push_step = 8
    push_duration = 0.2
    push_force = 103.752937314773
    push_start_timing = 24.1850018150851

    crouch_angle = int(trial_angle)
    crouch_idx = ['0', '20', '30', '60'].index(trial_angle)
    stride_means = [1.1262070300, 0.9529737358, 0.9158506655, 0.8755451448]
    speed_means = [0.9943359644, 0.8080297151, 0.7880050552, 0.7435198328]

    stride_vars = [0.03234099289, 0.02508595114, 0.02772452640, 0.02817863267]
    # stride_speed_covars = [0.03779884365, 0.02225320798, 0.02906793442, 0.03000639027]
    speed_vars = [0.06929309644, 0.04421889347, 0.04899931048, 0.05194827755]

    # step_length = 1.1
    # walk_speed = 1.0
    step_length = stride_means[crouch_idx] + 0.0 * math.sqrt(stride_vars[crouch_idx])
    walk_speed = speed_means[crouch_idx] + 0.0 * math.sqrt(speed_vars[crouch_idx])

    filename_prefix = 'nopush_' if 'nopush' in option else 'push_'
    filename_prefix += 'ad_' if 'ad' in option else 'bf_'
    filename_prefix += 'dep3_' if 'dep3' in option else ''
    if push_force > 0:
        filename = filename_prefix+str(trial_angle)+'deg_'+str(push_force)+'N_'+str(push_start_timing)+'t_step_'+str(step_length)[:5]+'_speed_'+str(walk_speed)[:5]
    else:
        filename = filename_prefix+str(trial_angle)+'deg_step_'+str(step_length)[:5]+'_speed_'+str(walk_speed)[:5]

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

    # step length
    motion_stride_bvh_after_default_param = 1.1886
    step_length_ratio = step_length / motion_stride_bvh_after_default_param

    # walk speed
    speed_bvh_after_default_param = 0.9134
    walk_speed_ratio = walk_speed / speed_bvh_after_default_param

    _sim.setParamedStepParams(int(crouch_angle), step_length_ratio, walk_speed_ratio)
    _sim.setPushParams(push_step, push_duration, -push_force, push_start_timing)

    stopcode = _sim.simulate()

    print(filename)
    print("stopcode:", stopcode)

    if stopcode != 1:
        motion = getMotionFromMotionSkel(_sim.getMotions())

        start, end = _sim.getPushStartFrame(), _sim.getPushEndFrame()
        if push_force > 0:
            filename += '_'+str(start)+'_'+str(end)+'.bvh'
        else:
            filename += '.bvh'
        with open(path_prefix+filename, 'w') as fout:
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../data/motion/bvh_base.bvh', 'r') as f_base:
                fout.write(f_base.read())
            fout.write('MOTION\n')
            fout.write('Frames: '+str(len(motion))+'\n')
            fout.write('Frame Time: 0.033333333333\n')
            for i in range(len(motion)):
                fout.write(' '.join(list(map(str, motion[i]))) + '\n')

    # for reference motion
    filename_ref = 'ref_' + str(trial_angle)+'deg_step_'+str(step_length)[:5]+'_speed_'+str(walk_speed)[:5] + '.bvh'
    if not os.path.exists(path_prefix+filename_ref):
        _sim.setParamedStepParams(int(crouch_angle), step_length_ratio, walk_speed_ratio)
        _sim.setPushParams(push_step, push_duration, -push_force, push_start_timing)
        _sim.simulate_motion()
        motion_ref = getMotionFromMotionSkel(_sim.getMotions())

        with open(path_prefix+filename_ref, 'w') as fout:
            with open(os.path.dirname(os.path.abspath(__file__)) + '/../data/motion/bvh_base.bvh', 'r') as f_base:
                fout.write(f_base.read())
            fout.write('MOTION\n')
            fout.write('Frames: '+str(len(motion_ref))+'\n')
            fout.write('Frame Time: 0.033333333333\n')
            for i in range(len(motion_ref)):
                fout.write(' '.join(list(map(str, motion_ref[i]))) + '\n')


