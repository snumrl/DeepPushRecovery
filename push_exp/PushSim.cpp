//
// Created by trif on 06/12/2019.
//

#include "PushSim.h"

using MASS::PushSim;
using MASS::Environment;

static np::ndarray toNumPyArray(const Eigen::VectorXd& vec)
{
    int n = vec.rows();
    p::tuple shape = p::make_tuple(n);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray array = np::empty(shape,dtype);

    float* dest = reinterpret_cast<float*>(array.get_data());
    for(int i =0;i<n;i++)
    {
        dest[i] = vec[i];
    }

    return array;
}

PushSim::
PushSim(const std::string &meta_file, const std::string& nn_path)
    : mMuscleNNLoaded(false)
{
    mEnv = new MASS::Environment(0);
    dart::math::seedRand();
    mEnv->Initialize(meta_file);

    mEnv->PrintWalkingParams();

    mm = p::import("__main__");
    mns = mm.attr("__dict__");
    sys_module = p::import("sys");

    p::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();
    sys_module.attr("path").attr("insert")(1, module_dir);
    p::exec("import torch",mns);
    p::exec("import torch.nn as nn",mns);
    p::exec("import torch.optim as optim",mns);
    p::exec("import torch.nn.functional as F",mns);
    p::exec("import torchvision.transforms as T",mns);
    p::exec("import numpy as np",mns);
    p::exec("from Model import *",mns);

    mNNLoaded = true;

    boost::python::str str = ("num_state = "+std::to_string(mEnv->GetNumState())).c_str();
    p::exec(str,mns);
    str = ("num_action = "+std::to_string(mEnv->GetNumAction())).c_str();
    p::exec(str,mns);

    nn_module = p::eval("SimulationNN(num_state,num_action)",mns);

    p::object load = nn_module.attr("load");
    load(nn_path);

    this->walk_fsm = WalkFSM();
    this->push_step = 8;
    this->push_duration = .2;
    this->push_force = 50.;
    this->push_start_timing = 50.;

    this->step_length_ratio = 1.;
    this->walk_speed_ratio = 1.;
    this->duration_ratio = 1.;

    this->info_start_time = 0.;
    this->info_end_time = 0.;
    this->info_root_pos.clear();
    this->info_left_foot_pos.clear();
    this->info_right_foot_pos.clear();

    this->push_start_time = 30.;
    this->push_end_time = 0.;
    this->walking_dir = Eigen::Vector3d::Zero();

    this->pushed_step = 0;
    this->pushed_length = 0;

    this->max_detour_length = 0.;
    this->max_detour_step_count = 0;

    this->valid = true;

}

PushSim::
PushSim(const std::string &meta_file,const std::string& nn_path,const std::string& muscle_nn_path)
        : PushSim(meta_file, nn_path)
{
    mMuscleNNLoaded = true;

    boost::python::str str = ("num_total_muscle_related_dofs = "+std::to_string(mEnv->GetNumTotalRelatedDofs())).c_str();
    p::exec(str,mns);
    str = ("num_actions = "+std::to_string(mEnv->GetNumAction())).c_str();
    p::exec(str,mns);
    str = ("num_muscles = "+std::to_string(mEnv->GetCharacter()->GetMuscles().size())).c_str();
    p::exec(str,mns);

    muscle_nn_module = p::eval("MuscleNN(num_total_muscle_related_dofs,num_actions,num_muscles)",mns);

    p::object load = muscle_nn_module.attr("load");
    load(muscle_nn_path);
}

PushSim::
~PushSim()
{
    delete mEnv;
}

void
PushSim::
Step()
{
    int num = mEnv->GetSimulationHz() / mEnv->GetControlHz();
    mEnv->SetAction(GetActionFromNN());

//	std::cout << mEnv->GetCharacter()->GetSkeleton()->getBodyNode(0)->getTransform().translation() << std::endl;

    if(mEnv->GetUseMuscle())
    {
        int inference_per_sim = 2;
        for(int i=0 ; i < num ; i += inference_per_sim){
            Eigen::VectorXd mt = mEnv->GetMuscleTorques();
            mEnv->SetActivationLevels(GetActivationFromNN(mt));
            for(int j=0;j<inference_per_sim;j++){
                if(this->push_start_time <= this->GetSimulationTime() && this->GetSimulationTime() <= this->push_end_time) {
                    this->AddBodyExtForce("ArmL", Eigen::Vector3d(this->push_force, 0., 0.));
                }
                mEnv->Step();
            }
        }
    }
    else
    {
        for(int i=0;i<num;i++) {
            mEnv->Step();
        }
    }
}

void
PushSim::
Reset(bool RSI)
{
    mEnv->Reset(RSI);
    mEnv->PrintWalkingParamsSampled();
}

Eigen::VectorXd
PushSim::
GetActionFromNN()
{
    p::object get_action;
    get_action= nn_module.attr("get_action");
    Eigen::VectorXd state = mEnv->GetState();
    p::tuple shape = p::make_tuple(state.rows());
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray state_np = np::empty(shape,dtype);

    float* dest = reinterpret_cast<float*>(state_np.get_data());
    for(int i =0;i<state.rows();i++)
        dest[i] = state[i];

    p::object temp = get_action(state_np);
    np::ndarray action_np = np::from_object(temp);

    float* srcs = reinterpret_cast<float*>(action_np.get_data());

    Eigen::VectorXd action(mEnv->GetNumAction());
    for(int i=0;i<action.rows();i++)
        action[i] = srcs[i];

    return action;
}

Eigen::VectorXd
PushSim::
GetActivationFromNN(const Eigen::VectorXd& mt)
{
    if(!mMuscleNNLoaded)
    {
        mEnv->GetDesiredTorques();
        return Eigen::VectorXd::Zero(mEnv->GetCharacter()->GetMuscles().size());
    }
    p::object get_activation = muscle_nn_module.attr("get_activation");
    Eigen::VectorXd dt = mEnv->GetDesiredTorques();
    np::ndarray mt_np = toNumPyArray(mt);
    np::ndarray dt_np = toNumPyArray(dt);

    p::object temp = get_activation(mt_np,dt_np);
    np::ndarray activation_np = np::from_object(temp);

    Eigen::VectorXd activation(mEnv->GetCharacter()->GetMuscles().size());
    float* srcs = reinterpret_cast<float*>(activation_np.get_data());
    for(int i=0;i<activation.rows();i++)
        activation[i] = srcs[i];

    return activation;


}

void
PushSim::
simulate(){
    this->mEnv->PrintWalkingParamsSampled();
    this->info_start_time = 0.;
    this->info_end_time = 0.;
    this->info_root_pos.clear();
    this->info_left_foot_pos.clear();
    this->info_right_foot_pos.clear();

    this->push_start_time = 30.;
    this->push_end_time = 0.;
    this->walking_dir = Eigen::Vector3d::Zero();

    this->pushed_step = 0;
    this->pushed_length = 0;

    this->valid = frue;

    this->walk_fsm.reset();

    this->max_detour_length = 0.;
    this->max_detour_step_count = 0;

    while (true) {
        bool bool_l = this->mEnv->IsBodyContact("TalusL") or this->mEnv->IsBodyContact("FootThumbL") or
                 this->mEnv->IsBodyContact("FootPinkyL");
        bool bool_r = this->mEnv->IsBodyContact("TalusR") or this->mEnv->IsBodyContact("FootThumbR") or
                 this->mEnv->IsBodyContact("FootPinkyR");
        int last_step_count = this->walk_fsm.step_count;
        this->walk_fsm.check(bool_l, bool_r);

        if(last_step_count == this->walk_fsm.step_count - 1) {
            std::cout << last_step_count << "->" << this->walk_fsm.step_count << this->GetSimulationTime() << std::endl;
        }

        if(last_step_count == 3 && this->walk_fsm.step_count == 4) {
            this->info_start_time = this->mEnv->GetSimulationTime();
            this->info_root_pos.push_back(this->mEnv->GetBodyPosition("Pelvis"));
            this->info_root_pos[0][1] = 0.;
            if (this->walk_fsm.last_sw == 'r')
                this->info_right_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusR"));
            else if (this->walk_fsm.last_sw == 'l')
                this->info_left_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusL"));
        }

        if(last_step_count == 4 and this->walk_fsm.step_count == 5) {
            // info_root_pos.push_back(this->mEnv->GetBodyPosition("Pelvis"))
            if (this->walk_fsm.last_sw == 'r')
                this->info_right_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusR"));
            else if (this->walk_fsm.last_sw == 'l')
                this->info_left_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusL"));
        }

        if(last_step_count == 5 and this->walk_fsm.step_count == 6) {
            // info_root_pos.push_back(this->mEnv->GetBodyPosition("Pelvis"))
            if (this->walk_fsm.last_sw == 'r')
                this->info_right_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusR"));
            else if (this->walk_fsm.last_sw == 'l')
                this->info_left_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusL"));
        }

        if(last_step_count == 6 and this->walk_fsm.step_count == 7) {
            // info_root_pos.push_back(this->mEnv->GetBodyPosition("Pelvis"))
            if (this->walk_fsm.last_sw == 'r')
                this->info_right_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusR"));
            else if (this->walk_fsm.last_sw == 'l')
                this->info_left_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusL"));
        }

        if(last_step_count == 7 and this->walk_fsm.step_count == 8) {
            std::cout << this->GetBodyPosition("TalusL")[2] << std::endl;
            std::cout << this->GetBodyPosition("TalusR")[2] << std::endl;
            this->info_end_time = this->GetSimulationTime();
            this->info_root_pos.push_back(this->GetBodyPosition("Pelvis"));
            this->info_root_pos[1][1] = 0.;
            if (this->walk_fsm.last_sw == 'r')
                this->info_right_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusR"));
            else if (this->walk_fsm.last_sw == 'l')
                this->info_left_foot_pos.push_back(this->mEnv->GetBodyPosition("TalusL"));
            20, 0.807417, 0.930071, 3.1266666666667;

            this->walking_dir = this->info_root_pos[1] - this->info_root_pos[0];
            this->walking_dir[1] = 0.;
            this->walking_dir /= np.linalg.norm(this->walking_dir);

            this->push_start_time = this->GetSimulationTime() +
                                    (this->push_start_timing / 100.) * this->GetMotionHalfCycleDuration();
            this->push_end_time = this->push_start_time + this->push_duration;
            std::cout << "push at " << this->push_start_time << std::endl;
        }

        if (this->GetSimulationTime() >= this->push_start_time) {
            Eigen::Vector3d root_pos_plane = this->GetBodyPosition("Pelvis");
            root_pos_plane[1] = 0.;
            double detour_length = calculate_distance_to_line(root_pos_plane, this->walking_dir, this->info_root_pos[0]);
            if (this->max_detour_length < detour_length) {
                this->max_detour_length = detour_length;
                this->max_detour_step_count = this->walk_fsm.step_count;
            }
        }

        if (this->GetSimulationTime() >= this->push_start_time + 10.)
            break;

        if (this->mEnv->GetBodyPosition("Pelvis")[1] < 0.3) {
            std::cout << "fallen at " << this->walk_fsm.step_count << this->mEnv->GetSimulationTime() << 's' << std::endl;
            this->valid = false;
            break;
        }

        // std::cout << this->mEnv->GetBodyPosition("Pelvis") std::endl;
        // std::cout << this->walk_fsm.step_count std::endl;

        this->Step();
    }
    std::cout << 'end!' << this->valid << std::endl;

}

void
PushSim::
setParamedStepParams(int _crouch_angle, double _step_length_ratio, double _walk_speed_ratio){
    this->step_length_ratio = step_length_ratio;
    this->walk_speed_ratio = walk_speed_ratio;
    this->duration_ratio = step_length_ratio / walk_speed_ratio;
    this->mEnv->SetWalkingParams((int)_crouch_angle), _step_length_ratio, _walk_speed_ratio);

}

void
PushSim::
setPushParams(int _push_step, double _push_duration, double _push_force, double _push_start_timing){
    this->push_step = _push_step;
    this->push_duration = _push_duration;
    this->push_force = _push_force/2.;
    this->push_start_timing = _push_start_timing;
    this->mEnv->SetPushParams(_push_step, _push_duration, _push_force, _push_start_timing);
}

double
PushSim::
getPushedLength(){
    return this->max_detour_length;
}

double
PushSim::
getPushedStep(){
    return this->max_detour_step_count;
}

double
PushSim::
getStepLength(){
    sum_stride_length = 0.;
    stride_info_num = 0;
    for info_foot_pos in [this->info_right_foot_pos, this->info_left_foot_pos]
            {}
    for i in range(len(info_foot_pos)-1):
    stride_vec = info_foot_pos[i+1] - info_foot_pos[i];
    stride_vec[1] = 0.;
    sum_stride_length += np.linalg.norm(stride_vec);
    stride_info_num += 1;

    return sum_stride_length/stride_info_num;
}

double
PushSim::
getWalkingSpeed(){
    walking_vec = this->info_root_pos[1] - this->info_root_pos[0];
    walking_vec[1] = 0.;
    distance = np.linalg.norm(walking_vec);
    return distance/(this->info_end_time-this->info_start_time);
}

double
PushSim::
getStartTimingFootIC(){
    return 0.;
}

double
PushSim::
getMidTimingFootIC(){
    return 0.;
}

double
PushSim::
getStartTimingTimeFL(){
    return 0.;
}

double
PushSim::
getMidTimingTimeFL(){
    return 0.;
}

double
PushSim::
getStartTimingFootFL(){
    return 0.;
}

double
PushSim::
getMidTimingFootFL() {
    return 0.;
}

void
PushSim::
PrintWalkingParams()
{
    mEnv->PrintWalkingParams();
}

void
PushSim::
PrintWalkingParamsSampled()
{
    mEnv->PrintWalkingParamsSampled();
}


void
PushSim::
SetWalkingParams(int crouch_angle, double stride_length, double walk_speed)
{
    mEnv->SetWalkingParams(crouch_angle, stride_length, walk_speed);
}

void
PushSim::
SetPushParams(int _push_step, double _push_duration, double _push_force, double _push_start_timing)
{
    mEnv->SetPushParams(_push_step, _push_duration, _push_force, _push_start_timing);
}

double
PushSim::
GetSimulationTime()
{
    return mEnv->GetWorld()->getTime();
}

bool
PushSim::
IsBodyContact(const std::string &name)
{
    return mEnv->GetWorld()->getLastCollisionResult().inCollision(
            mEnv->GetCharacter()->GetSkeleton()->getBodyNode(name)
    );
}

void
PushSim::
AddBodyExtForce(const std::string &name, const Eigen::Vector3d &_force)
{
    mEnv->GetCharacter()->GetSkeleton()->getBodyNode(name)->addExtForce(_force);
}

Eigen::Vector3d
PushSim::
GetBodyPosition(const std::string &name)
{
    return mEnv->GetCharacter()->GetSkeleton()->getBodyNode(name)->getTransform().translation();
}

double
PushSim::
GetMotionHalfCycleDuration()
{
    return mEnv->GetCharacter()->GetBVH()->GetMaxTime()/2.;
}
