import os
import numpy as np
import time

import multiprocessing as mp
import csv
import socket
import datetime
import math
import glob

from pypushexp import PushSim

import matplotlib.pyplot as plt

#    # input - [recorded item]
#    [weight] : 48
#    [height] : 160
#    [crouch_angle] (deg)
#    [step_length_ratio]  
#    [halfcycle_duration_ratio]
#    [push_step] : 8
#    [push_duration] (sec) : .2
#    [push_force] (N)
#    [push_start_timing] (half gait cycle percent)
#    
#    # output
#    [pushed_length] (m) : sim.out_pushed_length
#    [pushed_steps]  : sim.out_pushed_steps
#    [push_strength] : abs(push_force * push_duration / weight)
#    [step_length] (m) : sim.getPushedLength()
#    [walking_speed] (m/s) : sim.getWalkingSpeed()
#    [halfcycle_duration] (s) : sim.getStepLength() /sim.getWalkingSpeed()
#    
#    # output for hospital
#    [distance] : pushed_length * 1000.
#    [speed] : walking_speed * 1000.
#    [force] : push_strength * 1000.
#    [stride] : step_length * 1000.
#    [start_timing_time_ic] = sim.start_timing_time_ic
#    [mid_timing_time_ic] = sim.mid_timing_time_ic
#    [start_timing_foot_ic] = sim.getStartTimingFootIC()
#    [mid_timing_foot_ic] = sim.getMidTimingFootIC()
#    [start_timing_time_fl] = sim.getStartTimingTimeFL()
#    [mid_timing_time_fl] = sim.getMidTimingTimeFL()
#    [start_timing_foot_fl] = sim.getStartTimingFootFL()
#    [mid_timing_foot_fl] = sim.getMidTimingFootFL()

#    # not used
#    subject no
#    sex
#    left leg length
#    right leg length
#    stride
#    speed
#    experiment
#    file name
#    trial no
#    push timing : 'left stance'
#    push direction : 'from left'
#    normalized push length
#    push length until first step
#    push end timing (time)
#    push end timing (foot pos)
#    return during first step
#    push duration
#    push start time


def gettimestringisoformat():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def worker_simulation(sim, param, _x_0, _y_0, _z_0, _x_2, _y_2, _z_2):
    push_step, push_duration,\
                       crouch_angle, step_length_ratio, walk_speed_ratio, push_force, push_start_timing, crouch_label,\
                       weight, height, ith, q = param

    # print(int(crouch_angle), step_length_ratio, walk_speed_ratio, push_force, push_start_timing)
    sim.setParamedStepParams(int(crouch_angle), step_length_ratio, walk_speed_ratio)
    sim.setPushParams(push_step, push_duration, push_force, push_start_timing)

    stopcode = sim.simulate()

    if stopcode != 1 and stopcode != 4:
        pushed_length = sim.getPushedLength()
        pushed_steps = sim.getPushedStep()
        push_strength = abs(push_force * push_duration / weight)
        step_length = sim.getStepLength()
        walking_speed = sim.getWalkingSpeed()
        halfcycle_duration = sim.getStepLength() / sim.getWalkingSpeed()

        distance = pushed_length * 1000.
        speed = walking_speed * 1000.
        force = push_strength * 1000.
        stride = step_length * 1000.
        duration = halfcycle_duration

        # start_timing_time_ic = sim.start_timing_time_ic
        # mid_timing_time_ic = sim.mid_timing_time_ic
        start_timing_time_ic = sim.getStartTimingTimeIC()
        mid_timing_time_ic = sim.getMidTimingTimeIC()
        start_timing_foot_ic = sim.getStartTimingFootIC()
        mid_timing_foot_ic = sim.getMidTimingFootIC()
        start_timing_time_fl = sim.getStartTimingTimeFL()
        mid_timing_time_fl = sim.getMidTimingTimeFL()
        start_timing_foot_fl = sim.getStartTimingFootFL()
        mid_timing_foot_fl = sim.getMidTimingFootFL()

        st_pos = sim.getPushedStanceFootPosition()
        fp_pos = sim.getFootPlacementPosition()
        diff = fp_pos - st_pos
        if stopcode == 0:
            _x_0.append(-diff[0])
            _y_0.append(diff[2])
            _z_0.append(push_force)
        elif stopcode == 2:
            _x_2.append(-diff[0])
            _y_2.append(diff[2])
            _z_2.append(push_force)

        # print(stopcode, fp_pos - st_pos)


def write_start(csvfilepath):
    csvfile = open(csvfilepath, 'w')
    csvfile.write('ith,weight,height,distance,speed,force,stride,duration,crouch_angle,crouch_label,start_timing_time_ic,mid_timing_time_ic,start_timing_foot_ic,mid_timing_foot_ic,start_timing_time_fl,mid_timing_time_fl,start_timing_foot_fl,mid_timing_foot_fl,step_length,walking_speed,halfcycle_duration,push_strength,push_start_timing,pushed_length,pushed_steps,sim.stopcode,sim.step_length_ratio,sim.halfcycle_duration_ratio,sim.push_step,sim.push_duration,sim.push_force\n')
    return csvfile


def write_body(q, csvfile):
    while True:
        try:
            ith, weight, height, distance, speed, force, stride, duration, crouch_angle, crouch_label, \
            start_timing_time_ic, mid_timing_time_ic, start_timing_foot_ic, mid_timing_foot_ic, \
            start_timing_time_fl, mid_timing_time_fl, start_timing_foot_fl, mid_timing_foot_fl, \
            step_length, walking_speed, halfcycle_duration, push_strength, push_start_timing, pushed_length, pushed_steps, \
            stopcode, step_length_ratio, halfcycle_duration_ratio, push_step, push_duration, push_force = q.get(False)
                
            csvfile.write('%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(\
                        ith, weight, height, distance, speed, force, stride, duration, crouch_angle, crouch_label, \
                        start_timing_time_ic, mid_timing_time_ic, start_timing_foot_ic, mid_timing_foot_ic, \
                        start_timing_time_fl, mid_timing_time_fl, start_timing_foot_fl, mid_timing_foot_fl, \
                        step_length, walking_speed, halfcycle_duration, push_strength, push_start_timing, pushed_length, pushed_steps, \
                        stopcode, step_length_ratio, halfcycle_duration_ratio, push_step, push_duration, push_force))
            csvfile.flush()
        except:
            break


def write_end(csvfile):
    csvfile.close()


def simulate(sim, launch_order, num, option_str=''):
    #=======================================================================
    # settings
    #=======================================================================
    TEST = True if launch_order is None else False
    # TEST = True
    # TEST = False

    weight = 72
    height = 170

    push_step = 8
    push_duration = .2
    
    test_params = []    # element: (crouch_angle, step_length_ratio, halfcycle_duration_ratio, push_force, push_start_timing)

    # ===========================================================================
    #
    # ===========================================================================
    if TEST:
        # test
        additional_str = ''

        num = 2
        # num = 5000
        mean_crouch = [0, 20, 30, 60]
        
    else:
        # real
        all_mean_crouch = [0, 20, 30, 60]
        
        mean_crouch = [all_mean_crouch[launch_order % len(all_mean_crouch)]]
        additional_str = '_%ddeg__push' % mean_crouch[0]
        
        # if launch_order==0:
        #     param_opt_result = '130810_113234_0_60_push'
        #     additional_str = '_0_60_push'
        # elif launch_order==2:
        #     param_opt_result = '130810_161152_0_30_60_push'
        #     additional_str = '_0_30_60_push'

    # =======================================================================
    # set logger
    # =======================================================================

    outDir = os.path.dirname(os.path.abspath(__file__)) + '/results/'

    # if not os.path.exists(outDir):
    #     os.makedirs(outDir)

    csvfilepath = outDir + option_str + '_' + gettimestringisoformat() + '_' + str(num) + 'trials_' + socket.gethostname() + '.csv'
    print('start logging at', gettimestringisoformat())
    print()

    print('<simulation setting>')
    print('weight', weight)
    print('height', height)
    print('push_step', push_step)
    print('push_duration', push_duration)

    # =======================================================================
    # test2 : multivariate normal distribution
    # =======================================================================
    stride_means = [1.1262070300, 0.9529737358, 0.9158506655, 0.8755451448]
    speed_means = [0.9943359644, 0.8080297151, 0.7880050552, 0.7435198328]

    stride_vars = [0.03234099289, 0.02508595114, 0.02772452640, 0.02817863267]
    stride_speed_covars = [0.03779884365, 0.02225320798, 0.02906793442, 0.03000639027]
    speed_vars = [0.06929309644, 0.04421889347, 0.04899931048, 0.05194827755]

    # crouch angle
    # mean_crouch = [0,20,30,60]
    std_crouch = 1

    # step length
    motion_stride_bvh_after_default_param = 1.1886
    experi_stride_mean = stride_means[launch_order]
    experi_stride_std = math.sqrt(stride_vars[launch_order])
    mean_length_ratio = experi_stride_mean / motion_stride_bvh_after_default_param
    std_length_ratio = experi_stride_std / motion_stride_bvh_after_default_param

    # walk speed
    speed_bvh_after_default_param = 0.9134
    experi_speed_mean = speed_means[launch_order]
    experi_speed_std = math.sqrt(speed_vars[launch_order])
    mean_speed_ratio = experi_speed_mean / speed_bvh_after_default_param
    std_speed_ratio = experi_speed_std / speed_bvh_after_default_param

    # push strength
    mean_strength = .535
    std_strength = .096
    mean_force = -(mean_strength*weight/push_duration)
    std_force = (std_strength*weight/push_duration)
    
    # push timing
    mean_timing = 34
    std_timing = 21
    
    if TEST:
        np.set_printoptions(precision=4, linewidth=200)

    # for i in range(len(mean_crouch)):
    #     mean =          [mean_crouch[i], mean_length_ratio, mean_duration_ratio, mean_force, mean_timing,          mean_crouch[i]]
    #     cov = np.diag(  [std_crouch**2, std_length_ratio**2, std_duration_ratio**2, std_force**2, std_timing**2,   0])
    for i in range(len(mean_crouch)):
        mean =        [mean_crouch[i], mean_length_ratio,   mean_speed_ratio,   mean_force,   mean_timing,   mean_crouch[i]]
        cov = np.diag([0             , std_length_ratio**2, std_speed_ratio**2, std_force**2, std_timing**2, 0])
        cov[1, 2] = stride_speed_covars[i] / speed_bvh_after_default_param / motion_stride_bvh_after_default_param
        cov[2, 1] = stride_speed_covars[i] / speed_bvh_after_default_param / motion_stride_bvh_after_default_param

        if len(test_params) == 0:
            test_params = np.random.multivariate_normal(mean, cov, num)
        else:
            test_params = np.vstack((test_params, np.random.multivariate_normal(mean, cov, num)))

    # no negative crouch angle
    for i in range(len(test_params)):
        test_params[i][0] = abs(test_params[i][0])
        test_params[i][2] = abs(test_params[i][2])
        test_params[i][3] = -abs(test_params[i][3])
        
    # print(test_params)

    print()
    print('multivariate normal distribution')
    print()
    print('mean_crouch', mean_crouch)
    print('std_crouch', std_crouch)
    print()
    print('motion_step_stride', motion_stride_bvh_after_default_param)
    print('experi_step_length_mean', experi_stride_mean)
    print('experi_step_length_std', experi_stride_std)
    print('mean_length_ratio', mean_length_ratio)
    print('std_length_ratio', std_length_ratio)
    print()
    print('motion_speed', speed_bvh_after_default_param)
    print('experi_speed_mean', experi_speed_mean)
    print('experi_speed_std', experi_speed_std)
    print('mean_speed_ratio', mean_speed_ratio)
    print('std_speed_ratio', std_speed_ratio)
    print()
    print('mean_strength', mean_strength)
    print('std_strength', std_strength)
    print('mean_force', mean_force)
    print('std_force', std_force)
    print()
    print('mean_timing', mean_timing)
    print('std_timing', std_timing)
    print('num', num)
    print()
    print('total # of simulations', len(test_params))
    print()
    
    # =======================================================================
    # simulation
    # =======================================================================
    pt = time.time()

    print('<start simulation>')
    print('hostname %s ' % socket.gethostname())
    print()

    q = mp.Manager().Queue()

    groupsize = 100
    paramgroups = [[] for i in range( len(test_params)//groupsize + 1 )]
    ith = 1
    for i in range(len(test_params)):
        crouch_angle                = test_params[i][0]
        step_length_ratio           = test_params[i][1]
        walk_speed_ratio            = test_params[i][2]
        push_force                  = test_params[i][3]
        push_start_timing           = test_params[i][4]
        crouch_label                = test_params[i][5]
        paramgroups[i//groupsize].append((push_step, push_duration,
                       crouch_angle, step_length_ratio, walk_speed_ratio, push_force, push_start_timing, crouch_label,
                       weight, height, ith, q))
        ith += 1

    x_0 = []
    y_0 = []
    z_0 = []

    x_2 = []
    y_2 = []
    z_2 = []

    plt.ion()

    # csvfile = write_start(csvfilepath)
    for i in range(len(paramgroups)):
        for j in range(len(paramgroups[i])):
            worker_simulation(sim, paramgroups[i][j], x_0, y_0, z_0, x_2, y_2, z_2)
            plt.figure(0)
            plt.clf()
            plt.plot(x_0, y_0, 'ob')
            plt.plot(x_2, y_2, 'or')
            plt.xlim(left=0.)
            plt.ylim(bottom=0.)
            plt.show()
            plt.pause(0.001)
        # write_body(q, csvfile)
    # write_end(csvfile)
    input()

    print()
    _s = time.time() - pt
    _h = _s // 3600
    _m = _s // 60
    _s -= 60 * _m
    _m -= 60 * _h
    print('elapsed time = %d h:%d m:%d s' % (int(_h), int(_m), int(_s)))
    print()
    print('end logging at', gettimestringisoformat())


if __name__ == '__main__':
    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-p', '--pushed', help='using trained with push', action='store_true')
    # parser.add_argument('-f', '--segfoot', help='using multi-segment foot', action='store_true')
    # parser.add_argument('type', help='actuation type', choices=['torque', 'muscle'])
    # parser.add_argument('crouch', help='crouch angle', choices=['0', '20', '30', '60', 'all'])
    # parser.add_argument('dist', help='distribution method', choices=['normal', 'uniform', 'adaptive'])
    # parser.add_argument('trial_num', help='actuation type', type=int)
    #
    # args = parser.parse_args()
    # trial_num = args.trial_num
    # crouch = args.crouch
    #
    # option = ''
    # option += args.type + '_'
    # option += 'push_' if args.pushed else 'nopush_'
    # option += 'msf_' if args.segfoot else 'sf_'
    #
    # if args.crouch != 'all':
    #     option += 'crouch'
    # option += args.crouch
    # option += '_mean_var_'
    # option += args.dist

    import sys
    import re

    # option = sys.argv[1]
    # trial_num = int(sys.argv[2])
    # trial_angle = sys.argv[3]

    option = 'torque_push_both_sf_all_uniform'
    trial_num = 1000
    trial_angle = '20'

    _metadata_dir = os.path.dirname(os.path.abspath(__file__)) + '/../data/metadata/'
    _nn_finding_dir = os.path.dirname(os.path.abspath(__file__)) + '/../nn/*/'

    nn_dir = None
    if _nn_finding_dir is not None:
        nn_dir = glob.glob(_nn_finding_dir + option)[0]
    meta_file = _metadata_dir + option + '.txt'

    sim = None
    if 'muscle' in option:
        sim = PushSim(meta_file, nn_dir+'/max.pt', nn_dir+'/max_muscle.pt')
    else:
        sim = PushSim(meta_file, nn_dir+'/max.pt')

    if "all" in option:
        # simulate(sim, 0, trial_num, option)
        # simulate(sim, 1, trial_num, option)
        # simulate(sim, 2, trial_num, option)
        # simulate(sim, 3, trial_num, option)
        simulate(sim, ['0', '20', '30', '60'].index(trial_angle), trial_num, option)
    else:
        crouch = re.findall(r'crouch\d+', option)[0][6:]
        simulate(sim, ['0', '20', '30', '60'].index(crouch), trial_num, option)
