import os
import sys
import glob
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__))+'/../python')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from Model import *
from pymss import EnvWrapper
import copy


def calculate_distance_to_line(point, line_unit_vec, point_on_line):
    point_vec = point - point_on_line
    point_vec_perp = point_vec - np.dot(point_vec, line_unit_vec) * line_unit_vec
    return np.linalg.norm(point_vec_perp)


class WalkFSM(object):
    def __init__(self):
        self.last_sw = 'r'
        self.step_count = 0
        self.double = False

    def reset(self):
        self.last_sw = 'r'
        self.step_count = 0
        self.double = False

    def check(self, bool_l, bool_r):
        if not self.double and bool_l and bool_r:
            self.double = True
            self.step_count += 1
            self.last_sw = 'r' if self.last_sw == 'l' else 'l'

        elif self.double and (not bool_l) and self.last_sw == 'r':
            self.double = False

        elif self.double and (not bool_r) and self.last_sw == 'l':
            self.double = False


class PushSim(object):
    def __init__(self, params, metadata_dir, nn_finding_dir=None):
        self.is_muscle, self.is_pushed_during_training, self.is_multi_seg_foot, self.is_walking_variance, self.is_walking_param_normal_trained, self.crouch = \
            params

        option = ''
        option += 'muscle_' if self.is_muscle else 'torque_'
        option += 'push_' if self.is_pushed_during_training else 'nopush_'
        option += 'msf_' if self.is_multi_seg_foot else 'sf_'

        assert self.crouch in ['0', '20', '30', '60', 'all']
        if self.crouch != 'all':
            option += 'crouch'
        option += self.crouch
        option += '_mean'
        if self.is_walking_variance:
            option += '_var_'
            option += 'normal' if self.is_walking_param_normal_trained else 'uniform'

        nn_dir = None
        if nn_finding_dir is not None:
            nn_dir = glob.glob(nn_finding_dir + option)[0]

        self.env = EnvWrapper(metadata_dir+option+'.txt')
        num_state = self.env.GetNumState()
        num_action = self.env.GetNumAction()
        num_actions = self.env.GetNumAction()

        self.nn_module = None

        if nn_dir is not None:
            self.nn_module = SimulationNN(num_state, num_action)
            self.nn_module.load(nn_dir + '/max.pt')

        self.muscle_nn_module = None

        if self.is_muscle and nn_dir is not None:
            num_total_muscle_related_dofs = self.env.GetNumTotalMuscleRelatedDofs()
            num_muscles = self.env.GetNumMuscles()
            self.muscle_nn_module = MuscleNN(num_total_muscle_related_dofs, num_actions, num_muscles)
            self.muscle_nn_module.load(nn_dir + '/max_muscle.pt')

        self.walk_fsm = WalkFSM()
        self.push_step = 8
        self.push_duration = .2
        self.push_force = 50.
        self.push_start_timing = 50.

        self.step_length_ratio = 1.
        self.walk_speed_ratio = 1.
        self.duration_ratio = 1.

        self.info_start_time = 0.
        self.info_end_time = 0.
        self.info_root_pos = []
        self.info_left_foot_pos = []
        self.info_right_foot_pos = []

        self.push_start_time = 30.
        self.push_end_time = 0.
        self.walking_dir = np.zeros(3)

        self.pushed_step = 0
        self.pushed_length = 0

        self.max_detour_length = 0.
        self.max_detour_step_count = 0

        self.valid = True

    def GetActionFromNN(self):
        return self.nn_module.get_action(self.env.GetState())

    def GetActivationFromNN(self, mt):
        if not self.is_muscle:
            self.env.GetDesiredTorques()
            return np.zeros(self.env.GetNumMuscles())

        dt = self.env.GetDesiredTorques()
        return self.muscle_nn_module.get_activation(mt, dt)

    def step(self):
        num = self.env.GetSimulationHz() // self.env.GetControlHz()
        action = self.GetActionFromNN() if self.nn_module is not None else np.zeros(self.env.GetNumAction())
        self.env.SetAction(action)
        if self.is_muscle:
            inference_per_sim = 2
            for i in range(0, num, inference_per_sim):
                mt = self.env.GetMuscleTorques()
                self.env.SetActivationLevels(self.GetActivationFromNN(mt))
                for j in range(inference_per_sim):
                    if self.push_start_time <= self.env.GetSimulationTime() <= self.push_end_time:
                        self.env.AddBodyExtForce("ArmL", np.array([self.push_force, 0., 0.]))

                    self.env.Step()
        else:
            if self.push_start_time <= self.env.GetSimulationTime() <= self.push_end_time:
                self.env.AddBodyExtForce("ArmL", np.array([self.push_force, 0., 0.]))
            self.env.Step()

    def reset(self, rsi=True):
        self.env.Reset(rsi)
        self.env.PrintWalkingParamsSampled()

    def simulate(self):
        self.env.PrintWalkingParamsSampled()
        self.info_start_time = 0.
        self.info_end_time = 0.
        self.info_root_pos = []
        self.info_left_foot_pos = []
        self.info_right_foot_pos = []

        self.push_start_time = 30.
        self.push_end_time = 0.
        self.walking_dir = np.zeros(3)

        self.pushed_step = 0
        self.pushed_length = 0

        self.valid = True

        self.walk_fsm.reset()

        self.max_detour_length = 0.
        self.max_detour_step_count = 0

        while True:
            bool_l = self.env.IsBodyContact("TalusL") or self.env.IsBodyContact("FootThumbL") or self.env.IsBodyContact("FootPinkyL")
            bool_r = self.env.IsBodyContact("TalusR") or self.env.IsBodyContact("FootThumbR") or self.env.IsBodyContact("FootPinkyR")
            last_step_count = copy.deepcopy(self.walk_fsm.step_count)
            self.walk_fsm.check(bool_l, bool_r)

            if last_step_count == self.walk_fsm.step_count-1:
                print(last_step_count, '->', self.walk_fsm.step_count, self.env.GetSimulationTime())

            if last_step_count == 3 and self.walk_fsm.step_count == 4:
                self.info_start_time = self.env.GetSimulationTime()
                self.info_root_pos.append(self.env.GetBodyPosition("Pelvis"))
                self.info_root_pos[0][1] = 0.
                if self.walk_fsm.last_sw == 'r':
                    self.info_right_foot_pos.append(self.env.GetBodyPosition("TalusR"))
                elif self.walk_fsm.last_sw == 'l':
                    self.info_left_foot_pos.append(self.env.GetBodyPosition("TalusL"))

            if last_step_count == 4 and self.walk_fsm.step_count == 5:
                # info_root_pos.append(self.env.GetBodyPosition("Pelvis"))
                if self.walk_fsm.last_sw == 'r':
                    self.info_right_foot_pos.append(self.env.GetBodyPosition("TalusR"))
                elif self.walk_fsm.last_sw == 'l':
                    self.info_left_foot_pos.append(self.env.GetBodyPosition("TalusL"))

            if last_step_count == 5 and self.walk_fsm.step_count == 6:
                # info_root_pos.append(self.env.GetBodyPosition("Pelvis"))
                if self.walk_fsm.last_sw == 'r':
                    self.info_right_foot_pos.append(self.env.GetBodyPosition("TalusR"))
                elif self.walk_fsm.last_sw == 'l':
                    self.info_left_foot_pos.append(self.env.GetBodyPosition("TalusL"))

            if last_step_count == 6 and self.walk_fsm.step_count == 7:
                # info_root_pos.append(self.env.GetBodyPosition("Pelvis"))
                if self.walk_fsm.last_sw == 'r':
                    self.info_right_foot_pos.append(self.env.GetBodyPosition("TalusR"))
                elif self.walk_fsm.last_sw == 'l':
                    self.info_left_foot_pos.append(self.env.GetBodyPosition("TalusL"))

            if last_step_count == 7 and self.walk_fsm.step_count == 8:
                print(self.env.GetBodyPosition("TalusL")[2])
                print(self.env.GetBodyPosition("TalusR")[2])
                self.info_end_time = self.env.GetSimulationTime()
                self.info_root_pos.append(self.env.GetBodyPosition("Pelvis"))
                self.info_root_pos[1][1] = 0.
                if self.walk_fsm.last_sw == 'r':
                    self.info_right_foot_pos.append(self.env.GetBodyPosition("TalusR"))
                elif self.walk_fsm.last_sw == 'l':
                    self.info_left_foot_pos.append(self.env.GetBodyPosition("TalusL"))
                20, 0.807417, 0.930071, 3.1266666666667

                self.walking_dir = self.info_root_pos[1] - self.info_root_pos[0]
                self.walking_dir[1] = 0.
                self.walking_dir /= np.linalg.norm(self.walking_dir)

                self.push_start_time = self.env.GetSimulationTime() + (self.push_start_timing/100.) * self.env.GetMotionHalfCycleDuration()
                self.push_end_time = self.push_start_time + self.push_duration
                print("push at ", self.push_start_time)

            if self.env.GetSimulationTime() >= self.push_start_time:
                root_pos_plane = self.env.GetBodyPosition("Pelvis")
                root_pos_plane[1] = 0.
                detour_length = calculate_distance_to_line(root_pos_plane, self.walking_dir, self.info_root_pos[0])
                if self.max_detour_length < detour_length:
                    self.max_detour_length = detour_length
                    self.max_detour_step_count = self.walk_fsm.step_count

            if self.env.GetSimulationTime() >= self.push_start_time + 10.:
                break

            if self.env.GetBodyPosition("Pelvis")[1] < 0.3:
                print("fallen at ", self.walk_fsm.step_count, self.env.GetSimulationTime(), 's')
                self.valid = False
                break

            # print(self.env.GetBodyPosition("Pelvis"))
            # print(self.walk_fsm.step_count)

            self.step()
        print('end!', self.valid)

    def setParamedStepParams(self, crouch_angle, step_length_ratio, walk_speed_ratio):
        self.step_length_ratio = step_length_ratio
        self.walk_speed_ratio = walk_speed_ratio
        self.duration_ratio = step_length_ratio / walk_speed_ratio
        self.env.SetWalkingParams(int(crouch_angle), step_length_ratio, walk_speed_ratio)

    def setPushParams(self, push_step, push_duration, push_force, push_start_timing):
        self.push_step = push_step
        self.push_duration = push_duration
        self.push_force = push_force/2.
        self.push_start_timing = push_start_timing
        self.env.SetPushParams(push_step, push_duration, push_force, push_start_timing)

    def getPushedLength(self):
        return self.max_detour_length

    def getPushedStep(self):
        return self.max_detour_step_count

    def getStepLength(self):
        sum_stride_length = 0.
        stride_info_num = 0
        for info_foot_pos in [self.info_right_foot_pos, self.info_left_foot_pos]:
            for i in range(len(info_foot_pos)-1):
                stride_vec = info_foot_pos[i+1] - info_foot_pos[i]
                stride_vec[1] = 0.
                sum_stride_length += np.linalg.norm(stride_vec)
                stride_info_num += 1

        return sum_stride_length/stride_info_num

    def getWalkingSpeed(self):
        walking_vec = self.info_root_pos[1] - self.info_root_pos[0]
        walking_vec[1] = 0.
        distance = np.linalg.norm(walking_vec)
        return distance/(self.info_end_time-self.info_start_time)

    def getStartTimingFootIC(self):
        return 0.

    def getMidTimingFootIC(self):
        return 0.

    def getStartTimingTimeFL(self):
        return 0.

    def getMidTimingTimeFL(self):
        return 0.

    def getStartTimingFootFL(self):
        return 0.

    def getMidTimingFootFL(self):
        return 0.


if __name__ == '__main__':
    _metadata_dir = os.path.dirname(os.path.abspath(__file__)) + '/../data/metadata/'
    _nn_finding_dir = os.path.dirname(os.path.abspath(__file__)) + '/../../*/nn/*/'

    _is_muscle = False
    _is_pushed_during_training = False
    _is_multi_seg_foot = False
    _is_walking_variance = True
    _is_walking_param_normal_trained = False
    _crouch = input('crouch angle(0, 20, 30, 60, all)? ')
    _params = (_is_muscle, _is_pushed_during_training, _is_multi_seg_foot, _is_walking_variance, _is_walking_param_normal_trained, _crouch)

    sim = PushSim(_params, _metadata_dir, _nn_finding_dir)
