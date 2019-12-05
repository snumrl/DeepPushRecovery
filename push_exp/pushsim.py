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
                    self.env.Step()
        else:
            self.env.Step()

    def reset(self, rsi=True):
        self.env.Reset(rsi)
        self.env.PrintWalkingParamsSampled()

    def simulate(self):
        for i in range(40):
            self.step()
            print(self.env.GetSimulationTime())
        raise NotImplementedError

    def setParamedStepParams(self, crouch_angle, step_length_ratio, walk_speed_ratio):
        self.env.SetWalkingParams(int(crouch_angle), step_length_ratio, walk_speed_ratio)

    def setPushParams(self, push_step, push_duration, push_force, push_start_timing):
        self.env.SetPushParams(push_step, push_duration, push_force, push_start_timing)

    def getPushedLength(self):
        raise NotImplementedError

    def getPushedStep(self):
        raise NotImplementedError

    def getStepLength(self):
        raise NotImplementedError

    def getWalkingSpeed(self):
        raise NotImplementedError

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
