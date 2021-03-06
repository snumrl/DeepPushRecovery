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


def worker_simulation(sim, param):
    push_step, push_duration,\
                       crouch_angle, step_length_ratio, walk_speed_ratio, push_force, push_start_timing, crouch_label,\
                       weight, height, ith, q = param

    # print(int(crouch_angle), step_length_ratio, walk_speed_ratio, push_force, push_start_timing)
    sim.setParamedStepParams(int(crouch_angle), step_length_ratio, walk_speed_ratio)
    sim.setPushParams(push_step, push_duration, push_force, push_start_timing)

    stopcode = sim.simulate()
    # stopcode = 0

    # stopcode
    # 0: success
    # 1: fall down before push
    # 2: fall down after push
    # 3: push step > 3
    # 4: push length is zero
    # 5: NaN

    if stopcode == 1 or stopcode == 5:
        pushed_length = 0.
        pushed_steps = 0
        push_strength = 0.
        step_length = 0.
        walking_speed = 0.
        halfcycle_duration = 0.
    else:
        pushed_length = sim.getPushedLength()
        pushed_steps = sim.getPushedStep()
        push_strength = abs(push_force * push_duration / weight)
        step_length = sim.getStepLength()
        walking_speed = sim.getWalkingSpeed()
        halfcycle_duration = sim.getStepLength() / sim.getWalkingSpeed()

    # print(pushed_length, pushed_steps, push_strength, step_length, walking_speed)

    distance = pushed_length * 1000.
    speed = walking_speed * 1000.
    force = push_strength * 1000.
    stride = step_length * 1000.
    duration = halfcycle_duration

    if stopcode == 0 or stopcode == 2 or stopcode == 3:
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
    else:
        start_timing_time_ic = 0.
        mid_timing_time_ic = 0.
        start_timing_foot_ic = 0.
        mid_timing_foot_ic = 0.
        start_timing_time_fl = 0.
        mid_timing_time_fl = 0.
        start_timing_foot_fl = 0.
        mid_timing_foot_fl = 0.
        
    if stopcode == 0 or stopcode == 2 or stopcode == 3:
        stance_foot_pos = sim.getPushedStanceFootPosition()
        foot_placement_pos = sim.getFootPlacementPosition()
        foot_diff = foot_placement_pos - stance_foot_pos
        foot_placement_x = -foot_diff[0]
        foot_placement_y = foot_diff[2]

        #com_vel_foot_placement = sim.getCOMVelocityFootPlacement()
        #com_vel_x_foot_placement = -com_vel_foot_placement[0]
        #com_vel_y_foot_placement = com_vel_foot_placement[2]
        #com_vel_z_foot_placement = com_vel_foot_placement[1]
    else:
        foot_placement_x = 0.
        foot_placement_y = 0.
    com_vel_x_foot_placement = 0.
    com_vel_y_foot_placement = 0.
    com_vel_z_foot_placement = 0.

    halfcycle_duration_ratio = step_length_ratio / walk_speed_ratio

    q.put((ith, weight, height, distance, speed, force, stride, duration, crouch_angle, crouch_label,
           start_timing_time_ic, mid_timing_time_ic, start_timing_foot_ic, mid_timing_foot_ic,
           start_timing_time_fl, mid_timing_time_fl, start_timing_foot_fl, mid_timing_foot_fl,
           step_length, walking_speed, halfcycle_duration, push_strength, push_start_timing, pushed_length, pushed_steps,
           stopcode, step_length_ratio, halfcycle_duration_ratio, push_step, push_duration, push_force,
           foot_placement_x, foot_placement_y, com_vel_x_foot_placement, com_vel_y_foot_placement, com_vel_z_foot_placement))


def write_start(csvfilepath, exist=False):
    if exist:
        csvfile = open(csvfilepath, 'a')
    else:
        csvfile = open(csvfilepath, 'w')
        csvfile.write('ith,weight,height,distance,speed,force,stride,duration,crouch_angle,crouch_label,start_timing_time_ic,mid_timing_time_ic,start_timing_foot_ic,mid_timing_foot_ic,start_timing_time_fl,mid_timing_time_fl,start_timing_foot_fl,mid_timing_foot_fl,step_length,walking_speed,halfcycle_duration,push_strength,push_start_timing,pushed_length,pushed_steps,sim.stopcode,sim.step_length_ratio,sim.halfcycle_duration_ratio,sim.push_step,sim.push_duration,sim.push_force,foot_placement_x,foot_placement_y,com_vel_x_foot_placement,com_vel_y_foot_placement,com_vel_z_foot_placement\n')
    return csvfile


def write_body(q, csvfile):
    while True:
        try:
            ith, weight, height, distance, speed, force, stride, duration, crouch_angle, crouch_label, \
            start_timing_time_ic, mid_timing_time_ic, start_timing_foot_ic, mid_timing_foot_ic, \
            start_timing_time_fl, mid_timing_time_fl, start_timing_foot_fl, mid_timing_foot_fl, \
            step_length, walking_speed, halfcycle_duration, push_strength, push_start_timing, pushed_length, pushed_steps, \
            stopcode, step_length_ratio, halfcycle_duration_ratio, push_step, push_duration, push_force, \
            foot_placement_x, foot_placement_y, com_vel_x_foot_placement, com_vel_y_foot_placement, com_vel_z_foot_placement = q.get(False)
                
            csvfile.write('%d,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n'%(
                        ith, weight, height, distance, speed, force, stride, duration, crouch_angle, crouch_label,
                        start_timing_time_ic, mid_timing_time_ic, start_timing_foot_ic, mid_timing_foot_ic,
                        start_timing_time_fl, mid_timing_time_fl, start_timing_foot_fl, mid_timing_foot_fl,
                        step_length, walking_speed, halfcycle_duration, push_strength, push_start_timing, pushed_length, pushed_steps,
                        stopcode, step_length_ratio, halfcycle_duration_ratio, push_step, push_duration, push_force,
                        foot_placement_x, foot_placement_y, com_vel_x_foot_placement, com_vel_y_foot_placement, com_vel_z_foot_placement))
            csvfile.flush()
        except:
            # print('write error!')
            break


def write_end(csvfile):
    csvfile.close()


def simulate(sim, launch_order, num=100, option_str='', trial_force=None):
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
        if trial_force is None:
            additional_str = '_{deg}deg__push'.format(deg=mean_crouch[0])
        elif trial_force == -5:
            additional_str = '_{deg}deg__push_fix_length_speed_force'.format(deg=mean_crouch[0])
        elif trial_force == -4:
            additional_str = '_{deg}deg__push_fix_length_speed_timing'.format(deg=mean_crouch[0])
        elif trial_force == -3:
            additional_str = '_{deg}deg__push_fix_length_speed'.format(deg=mean_crouch[0])
        elif trial_force == -2:
            additional_str = '_{deg}deg__push_fix_length_speed_force'.format(deg=mean_crouch[0])
        elif trial_force == -1:
            additional_str = '_{deg}deg__push_fix_length_speed_timing'.format(deg=mean_crouch[0])
        elif trial_force == 0:
            additional_str = '_{deg}deg__push_fix_length_speed'.format(deg=mean_crouch[0])
        else:
            additional_str = '_{deg}deg__push_{force}N'.format(deg=mean_crouch[0], force=trial_force)

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

    if not os.path.exists(outDir):
        os.makedirs(outDir)

    csvfilepaths = glob.glob(outDir + option_str + additional_str + '*.csv')
    exist = False
    if csvfilepaths:
        csvfilepath = csvfilepaths[0]
        exist = True
    else:
        csvfilepath = outDir + option_str + additional_str + '_' + gettimestringisoformat() + '.csv'

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

    # including intended slow and narrow
    stride_means = [1.1262070300, 0.9529737358, 0.9158506655, 0.8755451448]
    speed_means = [0.9943359644, 0.8080297151, 0.7880050552, 0.7435198328]

    stride_vars = [0.03234099289, 0.02508595114, 0.02772452640, 0.02817863267]
    stride_speed_covars = [0.03779884365, 0.02225320798, 0.02906793442, 0.03000639027]
    speed_vars = [0.06929309644, 0.04421889347, 0.04899931048, 0.05194827755]

    # excluding intended slow and narrow
    # stride_means = [1.22868552, 0.9529737358, 0.9158506655, 0.8755451448]
    # speed_means = [1.15250623, 0.8080297151, 0.7880050552, 0.7435198328]
    #
    # stride_vars = [0.0128631447, 0.02508595114, 0.02772452640, 0.02817863267]
    # stride_speed_covars = [0.0130122325, 0.02225320798, 0.02906793442, 0.03000639027]
    # speed_vars = [0.0242001197, 0.04421889347, 0.04899931048, 0.05194827755]

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
    if True:
        for i in range(len(mean_crouch)):
            mean =        [mean_crouch[i], mean_length_ratio,   mean_speed_ratio,   mean_force,   mean_timing,   mean_crouch[i]]
            cov = np.diag([0             , std_length_ratio**2, std_speed_ratio**2, std_force**2, std_timing**2, 0])
            cov[1, 2] = stride_speed_covars[launch_order] / speed_bvh_after_default_param / motion_stride_bvh_after_default_param
            cov[2, 1] = stride_speed_covars[launch_order] / speed_bvh_after_default_param / motion_stride_bvh_after_default_param

            if len(test_params) == 0:
                test_params = np.random.multivariate_normal(mean, cov, num)
            else:
                test_params = np.vstack((test_params, np.random.multivariate_normal(mean, cov, num)))

    # no negative crouch angle
    for i in range(len(test_params)):
        test_params[i][0] = abs(test_params[i][0])
        test_params[i][2] = abs(test_params[i][2])
        if trial_force is None:
            test_params[i][3] = -abs(test_params[i][3])
        elif trial_force in [-5, -4, -3, -2, -1, 0]:
            test_params[i][3] = -abs(test_params[i][3])
        else:
            test_params[i][3] = -trial_force

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

    groupsize = 10
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

    csvfile = write_start(csvfilepath, exist)
    for i in range(len(paramgroups)):
        for j in range(len(paramgroups[i])):
            worker_simulation(sim, paramgroups[i][j])
        write_body(q, csvfile)
    write_end(csvfile)

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
    import sys
    import re
    
    q = mp.Manager().Queue()
    # option = 'torque_push_both_sf_crouch0_adaptive_k1'
    option = sys.argv[1]
    meta_file = os.path.dirname(os.path.abspath(__file__)) + '/../data/metadata/' + option + '.txt'
    nn_dir = os.path.dirname(os.path.abspath(__file__)) + '/../nn/done/'+option
    _sim = PushSim(meta_file, nn_dir+'/max.pt')

    fout = write_start('like_human-data_'+option+'.csv')

    with open(os.path.dirname(os.path.abspath(__file__)) + '/../data/human_data/human-data.csv', 'r') as csv_f:
        reader = csv.reader(csv_f)
        first_line = None

        experi_count = [0 for _ in range(30)]

        stride_idx = -1
        speed_idx = -1
        crouch_idx = -1
        height_idx = -1
        mass_idx = -1
        force_idx = -1
        push_start_idx = -1
        push_duration_idx = -1

        for line in reader:
            if first_line is None:
                first_line = line
                stride_idx = first_line.index('actual_stride')
                speed_idx = first_line.index('actual_speed')
                height_idx = first_line.index('height')
                mass_idx = first_line.index('weight')
                crouch_idx = first_line.index('crouch')
                force_idx = first_line.index('push_force_normalized')
                push_start_idx = first_line.index('push_start')
                push_duration_idx = first_line.index('push_duration')
                continue

            if line[crouch_idx] == '0':
                crouch = '0'
            elif line[crouch_idx] == '1':
                crouch = '20'
            elif line[crouch_idx] == '2':
                crouch = '30'
            else:
                crouch = '60'

            if not('crouch'+crouch in option):
                continue

            stride = float(line[stride_idx]) / 1000.
            speed = float(line[speed_idx]) / 1000.
            force = float(line[force_idx]) / 1000.
            push_start = float(line[push_start_idx])
            push_duration = float(line[push_duration_idx])
            push_step = 8


            crouch_angle                = int(crouch)
            step_length_ratio           = stride / 1.1886
            walk_speed_ratio            = speed / 0.9134
            push_force                  = force * 5. * 72.
            push_start_timing           = push_start
            crouch_label                = int(crouch)
            weight = 72.
            height = 170.
            ith = int(line[0][7:])
            param = (push_step, push_duration,
                           crouch_angle, step_length_ratio, walk_speed_ratio, push_force, push_start_timing, crouch_label,
                           weight, height, ith, q)
            worker_simulation(_sim, param)
            write_body(q, fout)
    write_end(fout)
