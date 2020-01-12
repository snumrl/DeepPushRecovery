//
// Created by trif on 06/12/2019.
//

#include "PushSim.h"

using MASS::WalkFSM;
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
static np::ndarray toNumPyArray(const std::vector<Eigen::VectorXd>& val)
{
    int n =val.size();
    int m = val[0].rows();
    p::tuple shape = p::make_tuple(n,m);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray array = np::empty(shape,dtype);

    float* dest = reinterpret_cast<float*>(array.get_data());
    int index = 0;
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<m;j++)
        {
            dest[index++] = val[i][j];
        }
    }

    return array;
}

static double calculate_distance_to_line(const Eigen::Vector3d &point, const Eigen::Vector3d &line_unit_vec, const Eigen::Vector3d &point_on_line, const Eigen::Vector3d &force_vec) {
    Eigen::Vector3d point_vec = point - point_on_line;
    Eigen::Vector3d point_vec_perp = point_vec - point_vec.dot(line_unit_vec) * line_unit_vec;
    return copysign(point_vec_perp.norm(), point_vec_perp.dot(force_vec));
}


PushSim::
PushSim(const std::string &meta_file, const std::string& nn_path)
    : mMuscleNNLoaded(false)
{
    mEnv = new MASS::Environment(0);
    dart::math::seedRand();
    mEnv->Initialize(meta_file);
    mEnv->SetPushEnable(false);

    // mEnv->PrintWalkingParams();

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

    this->push_step = 8;
    this->push_duration = .2;
    this->push_force = 50.;
    this->push_start_timing = 50.;
    this->push_force_vec.setZero();

    this->step_length_ratio = 1.;
    this->walk_speed_ratio = 1.;
    this->duration_ratio = 1.;


    this->simulatePrepare();
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

    if(mEnv->GetUseMuscle()) {
        int inference_per_sim = 2;
        for(int i=0 ; i < num ; i += inference_per_sim) {
            Eigen::VectorXd mt = mEnv->GetMuscleTorques();
            mEnv->SetActivationLevels(GetActivationFromNN(mt));
            for(int j=0;j<inference_per_sim;j++) {
                mEnv->Step();
            }
        }
    }
    else {
        for(int i=0;i<num;i++) {
            mEnv->Step();
        }
    }
}

void
PushSim::
PushStep()
{
    int num = mEnv->GetSimulationHz() / mEnv->GetControlHz();
    mEnv->SetAction(GetActionFromNN());

    last_root_pos = GetBodyPosition("Pelvis");
    last_root_pos[1] = 0.;

    if(mEnv->GetUseMuscle()) {
        int inference_per_sim = 2;
        for(int i=0 ; i < num ; i += inference_per_sim) {
            Eigen::VectorXd mt = mEnv->GetMuscleTorques();
            mEnv->SetActivationLevels(GetActivationFromNN(mt));
            for(int j=0;j<inference_per_sim;j++) {
                _PushStep();
                if(this->push_start_time <= this->GetSimulationTime() && this->GetSimulationTime() <= this->push_end_time) {
                    this->AddBodyExtForce("ArmL", this->push_force_vec);
                }
                mEnv->Step();
            }
        }
    }
    else {
        for(int i=0;i<num;i++) {
            _PushStep();
            if(this->push_start_time <= this->GetSimulationTime() && this->GetSimulationTime() <= this->push_end_time) {
                this->AddBodyExtForce("ArmL", this->push_force_vec);
                if (push_start_frame == 10000)
                    push_start_frame = this->motion.size();
            }
            else{
                if (push_start_frame != 10000 && push_end_frame == -1)
                    push_end_frame = this->motion.size();
            }
            mEnv->Step();
        }
    }

    Eigen::Vector3d root_pos = this->GetBodyPosition("Pelvis");
    root_pos[1] = 0.;
    this->travelDistance += (root_pos - last_root_pos).norm();

    this->motion.push_back(this->getPoseForBvh());
}

void
PushSim::
Reset(bool RSI)
{
    mEnv->Reset(RSI);
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
simulatePrepare()
{
    // this->mEnv->PrintWalkingParamsSampled();
    // this->mEnv->PrintPushParamsSampled();
    this->info_start_time = 0.;
    this->info_end_time = 0.;
    this->info_root_pos.clear();
    this->info_left_foot_pos.clear();
    this->info_right_foot_pos.clear();
    this->info_left_foot_pos_with_toe_off.clear();
    this->info_right_foot_pos_with_toe_off.clear();

    this->info_com_vel.clear();

    this->motion.clear();
    this->motion.push_back(this->getPoseForBvh());

    this->info_start_time_backup = 0.;
    this->info_root_pos_backup.setZero();

    this->pushed_step_time = 0.;
    this->pushed_next_step_time = 0.;

    this->pushed_step_time_toe_off = 0.;
    this->pushed_next_step_time_toe_off = 0.;

    this->push_start_frame = 10000;
    this->push_end_frame = -1;
    this->push_start_time = 30.;
    this->push_mid_time = 30.;
    this->push_end_time = 30.;
    this->walking_dir = Eigen::Vector3d::Zero();
    this->push_force_vec = Eigen::Vector3d::Zero();

    this->pushed_step = 0;
    this->pushed_length = 0;
    this->valid = true;
    this->stopcode = 0;

    this->max_detour_root_pos.setZero();
    this->max_detour_on_line.setZero();

    this->walk_fsm.reset();

    this->push_ready = false;

    this->pushed_start = false;
    this->pushed_start_pos.setZero();
    this->pushed_start_foot_pos.setZero();
    this->pushed_start_toe_pos.setZero();

    this->pushed_mid = false;
    this->pushed_mid_pos.setZero();
    this->pushed_mid_foot_pos.setZero();
    this->pushed_mid_toe_pos.setZero();

    this->travelDistance = 0.;
    this->last_root_pos.setZero();

    this->first_root_pos.setZero();
}


void
PushSim::
PushStep_old()
{
    bool bool_l = this->IsBodyContact("TalusL") || this->IsBodyContact("FootThumbL")
                  || this->IsBodyContact("FootPinkyL");
    bool bool_r = this->IsBodyContact("TalusR") || this->IsBodyContact("FootThumbR")
                  || this->IsBodyContact("FootPinkyR");
    int last_step_count = this->walk_fsm.step_count;
    bool last_step_double_st = this->walk_fsm.is_double_st;
    this->walk_fsm.check(bool_l, bool_r);
    this->last_root_pos = this->GetBodyPosition("Pelvis");
    this->last_root_pos[1] = 0.;

    if(last_step_count + 1 == this->walk_fsm.step_count) {
        // std::cout << last_step_count << "->" << this->walk_fsm.step_count << " "<< this->GetSimulationTime() << std::endl;
    }
    if(last_step_count == 0 && this->walk_fsm.step_count == 1) {
        this->first_root_pos = this->GetBodyPosition("Pelvis");
    }

    if (this->walk_fsm.step_count >= 3 && this->walk_fsm.step_count <= 9 && last_step_count + 1 == this->walk_fsm.step_count) {
        if (this->walk_fsm.step_count == 4){
            this->info_start_time = this->GetSimulationTime();
            this->info_root_pos.push_back(this->GetBodyPosition("Pelvis"));
        }
        if (this->walk_fsm.step_count % 2 == 0) {
            this->info_left_foot_pos.push_back(this->GetBodyPosition("TalusL"));
        }
        else {
            this->info_right_foot_pos.push_back(this->GetBodyPosition("TalusR"));
        }
    }

    if(last_step_count == 7 && this->walk_fsm.step_count == 8) {
        this->info_end_time = this->GetSimulationTime();
        this->pushed_step_time = this->GetSimulationTime();
        this->info_root_pos.push_back(this->GetBodyPosition("Pelvis"));
        // this->info_root_pos[1][1] = 0.;

        // this->walking_dir = this->info_root_pos[1] - this->info_root_pos[0];
        this->walking_dir = this->info_root_pos[1] - this->first_root_pos;
        this->walking_dir[1] = 0.;
        this->walking_dir.normalize();

        this->push_start_time = this->GetSimulationTime() +
                                (this->push_start_timing / 100.) * this->GetMotionHalfCycleDuration();
        this->push_mid_time = this->push_start_time + .5 * this->push_duration;
        this->push_end_time = this->push_start_time + this->push_duration;

        this->push_force_vec = this->push_force * Eigen::Vector3d::UnitY().cross(this->walking_dir);
        // std::cout << "push at " << this->push_start_time << std::endl;
    }

    if(last_step_count == 8 && this->walk_fsm.step_count == 9) {
        this->pushed_next_step_time = this->GetSimulationTime();
    }

    if(last_step_double_st && !this->walk_fsm.is_double_st) {
        if (this->walk_fsm.step_count == 8) {
            info_right_foot_pos_with_toe_off.push_back(this->GetBodyPosition("FootThumbR"));
            this->pushed_step_time_toe_off = this->GetSimulationTime();
        }
        if (this->walk_fsm.step_count == 9) {
            info_right_foot_pos_with_toe_off.push_back(this->GetBodyPosition("FootThumbR"));
            this->pushed_next_step_time_toe_off = this->GetSimulationTime();
        }
    }

    if (this->GetSimulationTime() >= this->push_start_time) {
        Eigen::Vector3d root_pos_plane = this->GetBodyPosition("Pelvis");
        if (!pushed_start) {
            pushed_start = true;
            pushed_start_pos = root_pos_plane;
            pushed_start_foot_pos = this->GetBodyPosition("TalusR");
            pushed_start_toe_pos = this->GetBodyPosition("FootThumbR");
        }
        root_pos_plane[1] = 0.;
        // Eigen::Vector3d point_on_line = this->info_root_pos[0];
        Eigen::Vector3d point_on_line = pushed_start_pos;
        point_on_line[1] = 0.;
        double detour_length = calculate_distance_to_line(root_pos_plane, this->walking_dir, point_on_line, this->push_force_vec);
        // std::cout <<"info" <<std::endl;
        // std::cout << root_pos_plane << std::endl;
        // std::cout << this->walking_dir << std::endl;
        // std::cout << point_on_line << std::endl;
        // std::cout << "detour_length: " << detour_length << std::endl;

        if (this->walk_fsm.step_count < 13 && this->pushed_length < detour_length) {
            this->pushed_length = detour_length;
            this->pushed_step = this->walk_fsm.step_count - 8;
            this->max_detour_root_pos = GetBodyPosition("Pelvis");
            this->max_detour_root_pos[1] = this->info_root_pos[1][1];
            // this->max_detour_on_line = this->info_root_pos[0] + this->walking_dir.dot(root_pos_plane - point_on_line) * this->walking_dir;
            // this->max_detour_on_line[1] = this->info_root_pos[1][1];
            this->max_detour_on_line = this->pushed_start_pos + this->walking_dir.dot(root_pos_plane - point_on_line) * this->walking_dir;
            // TODO: second max
        }

        if(this->pushed_step > 3)
        {
            this->stopcode = 3;  // pushed, didn't falling down but distance is so far
        }
    }
    if (this->GetSimulationTime() >= this->push_mid_time && !pushed_mid) {
        pushed_mid = true;
        pushed_mid_pos = this->GetBodyPosition("Pelvis");
        pushed_mid_foot_pos = this->GetBodyPosition("TalusR");
        // std::cout << "pushed_mid_foot_pos: " << pushed_mid_foot_pos << std::endl;
        pushed_mid_toe_pos = this->GetBodyPosition("FootThumbR");
    }

    this->Step();
    Eigen::Vector3d root_pos = this->GetBodyPosition("Pelvis");
    root_pos[1] = 0.;
    this->travelDistance += (root_pos - last_root_pos).norm();

}

void PushSim::_PushStep() {
    const double current_time = GetSimulationTime();
    const double bvh_cycle_duration = mEnv->GetCharacter()->GetBVH()->GetMaxTime();
    const double half_cycle_duration = bvh_cycle_duration / 2.;

    const double phase = std::fmod(current_time, half_cycle_duration);
    const int steps = (int) (current_time / half_cycle_duration) + 1;
    const double steps_double = current_time / half_cycle_duration + 1;

    if (steps_double >= 7.3 && info_right_foot_pos_with_toe_off.size() == 0 &&
        (!IsBodyContact("TalusR") && !IsBodyContact("FootThumbR") && !IsBodyContact("FootPinkyR"))
            )
    {
        info_right_foot_pos_with_toe_off.push_back(this->GetBodyPosition("FootThumbR"));
        pushed_step_time_toe_off = current_time;
    }
    if (steps >= 8 && info_right_foot_pos_with_toe_off.size() == 1 && info_right_foot_pos.size() == 3 &&
        IsBodyContact("TalusR")
            )
    {
        pushed_next_step_time = current_time;
        info_right_foot_pos.push_back(GetBodyPosition("TalusR"));
        this->info_com_vel.push_back(this->mEnv->GetCharacter()->GetSkeleton()->getCOMLinearVelocity());
    }
    if (steps >= 8 && info_right_foot_pos.size() == 4 && info_right_foot_pos_with_toe_off.size() == 1 &&
        (!IsBodyContact("TalusR") && !IsBodyContact("FootThumbR") && !IsBodyContact("FootPinkyR"))
            )
    {
        pushed_next_step_time_toe_off = current_time;
        info_right_foot_pos_with_toe_off.push_back(GetBodyPosition("FootThumbR"));
    }

    if ( steps_double >= 7.7 && !push_ready &&
            (IsBodyContact("TalusL") || IsBodyContact("FootThumbL") || IsBodyContact("FootPinkyL"))
            )
    {
        push_ready = true;
        info_end_time = current_time;
        pushed_step_time = current_time;
        info_root_pos.push_back(this->GetBodyPosition("Pelvis"));

        walking_dir = info_root_pos[1] - info_root_pos[0];
        walking_dir[1] = 0.;
        walking_dir.normalize();

        push_start_time = current_time + (push_start_timing/100.) * half_cycle_duration;
        push_mid_time = push_start_time + .5 * push_duration;
        push_end_time = push_start_time + push_duration;
        push_force_vec = push_force * Eigen::Vector3d::UnitY().cross(walking_dir);
    }

    if (phase >= 0. && phase - (1./mEnv->GetSimulationHz()) <= 0.) {
        // foot contact
        if (steps == 4) {
            info_start_time = current_time;
            info_root_pos.push_back(this->GetBodyPosition("Pelvis"));
        }

        if (steps >= 3 && steps <= 8) {
            if(steps %2 == 0)
                info_left_foot_pos.push_back(GetBodyPosition("TalusL"));
            else
                info_right_foot_pos.push_back(GetBodyPosition("TalusR"));
        }
    }

    if (current_time >= push_start_time && steps < 13) {
        Eigen::Vector3d root_pos_plane = GetBodyPosition("Pelvis");
        if (!pushed_start) {
            pushed_start = true;
            pushed_start_pos = root_pos_plane;
            pushed_start_foot_pos = GetBodyPosition("TalusR");
            pushed_start_toe_pos = GetBodyPosition("FootThumbR");
        }
        root_pos_plane[1] = 0.;
        Eigen::Vector3d point_on_line = pushed_start_pos;
        point_on_line[1] = 0.;
        double detour_length = calculate_distance_to_line(root_pos_plane, walking_dir, point_on_line, push_force_vec);

        if (pushed_length < detour_length) {
            pushed_length = detour_length;
            pushed_step = steps - 8;
            max_detour_root_pos = GetBodyPosition("Pelvis");
            max_detour_root_pos[1] = info_root_pos[1][1];
            max_detour_on_line = pushed_start_pos + walking_dir.dot(root_pos_plane - point_on_line) * walking_dir;
        }

        if(pushed_step > 3)
        {
            stopcode = 3;  // pushed, didn't falling down but distance is so far
        }
    }

    if (current_time >= push_mid_time && !pushed_mid) {
        pushed_mid = true;
        pushed_mid_pos = GetBodyPosition("Pelvis");
        pushed_mid_foot_pos = GetBodyPosition("TalusR");
        pushed_mid_toe_pos = GetBodyPosition("FootThumbR");
    }
}

void
PushSim::
simulate_motion() {
    simulatePrepare();
    double t = mEnv->GetWorld()->getTime();
    double dt = 1./mEnv->GetControlHz();
    while (t < 10.) {
        Eigen::VectorXd p = mEnv->GetCharacter()->GetTargetPositions(t, 1./mEnv->GetControlHz());
        mEnv->GetCharacter()->GetSkeleton()->setPositions(p);
        mEnv->GetCharacter()->GetSkeleton()->computeForwardKinematics(true, false, false);
        mEnv->GetWorld()->setTime(t + dt);
        t = mEnv->GetWorld()->getTime();
        this->motion.push_back(this->getPoseForBvh());
    }
}

int
PushSim::
simulate(){
    simulatePrepare();


    while (this->valid) {
        PushStep();

        Eigen::VectorXd p = mEnv->GetCharacter()->GetSkeleton()->getPositions();
        Eigen::VectorXd v = mEnv->GetCharacter()->GetSkeleton()->getVelocities();

        if (dart::math::isNan(p) || dart::math::isNan(v))
        {
            this->valid = false;
            this->stopcode = 5;
        }

        if (this->GetSimulationTime() >= this->push_start_time + 10.)
            break;

        if (this->GetBodyPosition("Pelvis")[1] < 0.3 * mEnv->GetHeightScale()) {
            // std::cout << "fallen at " << this->walk_fsm.step_count << " "<< this->GetSimulationTime() << "s" << std::endl;
            this->valid = false;
            if (this->pushed_start)
                this->stopcode = 2; // falling down after push
            else
                this->stopcode = 1; // falling down before push
            break;
        }

    }
    if (pushed_step == 0 && this->valid) {
        this->stopcode = 4;  // pushed but pushed distance is minus
        this->valid = false;
    }
    if (pushed_step > 3 && this->valid){
        this->valid = false;
    }

    return this->stopcode;

    // std::cout << "end!" << " " << (this->valid ? "True":"False") << " " << this->walk_fsm.step_count << " steps"<< std::endl;
}

void
PushSim::
setParamedStepParams(int _crouch_angle, double _step_length_ratio, double _walk_speed_ratio){
    this->step_length_ratio = _step_length_ratio;
    this->walk_speed_ratio = _walk_speed_ratio;
    this->duration_ratio = _step_length_ratio / _walk_speed_ratio;
    double motion_stride_bvh_after_default_param = 1.1886;
    double speed_bvh_after_default_param = 0.9134;
    this->mEnv->SetWalkingParams((int)_crouch_angle,
            _step_length_ratio * motion_stride_bvh_after_default_param,
            _walk_speed_ratio * speed_bvh_after_default_param);
    // this->mEnv->PrintWalkingParamsSampled();
}

void
PushSim::
setPushParams(int _push_step, double _push_duration, double _push_force, double _push_start_timing){
    this->push_step = _push_step;
    this->push_duration = _push_duration;
    this->push_force = _push_force;
    this->push_start_timing = _push_start_timing;
    // this->mEnv->SetPushParams(_push_step, _push_duration, _push_force, _push_start_timing);

    // push_force_vec should be set in PushStep
    // this->push_force_vec = Eigen::Vector3d(this->push_force, 0., 0.);
    // this->mEnv->PrintPushParamsSampled();
}

double
PushSim::
getPushedLength(){
    return this->pushed_length;
}

double
PushSim::
getPushedStep(){
    return this->pushed_step;
}

double
PushSim::
getStepLength(){
    double sum_stride_length = 0.;
    int stride_info_num = 0;

    for(int i=0; i< this->info_left_foot_pos.size()-1; i++) {
        Eigen::Vector3d stride_vec = info_left_foot_pos[i + 1] - info_left_foot_pos[i];
        stride_vec[1] = 0.;
        sum_stride_length += stride_vec.norm();
        stride_info_num += 1;
    }

    for(int i=0; i< this->info_right_foot_pos.size()-2; i++) {
        Eigen::Vector3d stride_vec = info_right_foot_pos[i + 1] - info_right_foot_pos[i];
        stride_vec[1] = 0.;
        sum_stride_length += stride_vec.norm();
        stride_info_num += 1;
    }

    return sum_stride_length / stride_info_num;
}

double
PushSim::
getWalkingSpeed(){
    Eigen::Vector3d walking_vec = this->info_root_pos[1] - this->info_root_pos[0];
    walking_vec[1] = 0.;
    return walking_vec.norm() / (this->info_end_time - this->info_start_time);
}

double
PushSim::
getStartTimingTimeIC(){
//    (push_start_time - prev_foot_ic_time) / (swing_foot_ic_time - prev_foot_ic_time);
    return 100.*(push_start_time - pushed_step_time) / (pushed_next_step_time - pushed_step_time);
}

double
PushSim::
getMidTimingTimeIC(){
    return 100.*(push_mid_time - pushed_step_time) / (pushed_next_step_time - pushed_step_time);
}

double
PushSim::
getStartTimingFootIC(){
    // (push_start_pos - prev_foot_ic_pos) / (swing_foot_ic_pos - prev_foot_ic_pos)
    return 100.*walking_dir.dot(this->pushed_start_foot_pos - this->info_right_foot_pos[2])
         / walking_dir.dot(this->info_right_foot_pos[3] - this->info_right_foot_pos[2]);
}

double
PushSim::
getMidTimingFootIC(){
    return 100.*walking_dir.dot(this->pushed_mid_foot_pos - this->info_right_foot_pos[2])
        / walking_dir.dot(this->info_right_foot_pos[3] - this->info_right_foot_pos[2]);
}

double
PushSim::
getStartTimingTimeFL(){
    // (push_start_time - prev_foot_fl_time) / (swing_foot_fl_time - prev_foot_fl_time)
    return 100.*(push_start_time - pushed_step_time_toe_off) / (pushed_next_step_time_toe_off - pushed_step_time_toe_off);
}

double
PushSim::
getMidTimingTimeFL(){
    return 100.*(push_mid_time - pushed_step_time_toe_off) / (pushed_next_step_time_toe_off - pushed_step_time_toe_off);
}

double
PushSim::
getStartTimingFootFL(){
    return 100.*walking_dir.dot(this->pushed_start_toe_pos - this->info_right_foot_pos_with_toe_off[0])
           / walking_dir.dot(this->info_right_foot_pos_with_toe_off[1] - this->info_right_foot_pos_with_toe_off[0]);
}

double
PushSim::
getMidTimingFootFL() {
    // (push_start_pos - prev_foot_fl_pos) / (swing_foot_fl_pos - prev_foot_fl_pos)
    return 100.*walking_dir.dot(this->pushed_mid_toe_pos - this->info_right_foot_pos_with_toe_off[0])
           / walking_dir.dot(this->info_right_foot_pos_with_toe_off[1] - this->info_right_foot_pos_with_toe_off[0]);
}


double PushSim::getMechanicalWork() {
    return this->mEnv->GetMechanicalWork();
}

double PushSim::getTravelDistance() {
    return this->travelDistance;
}

double PushSim::getCostOfTransport() {
    return this->getMechanicalWork() / (72. * 9.8 * this->getTravelDistance());
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
np::ndarray
PushSim::
getPushedStanceFootPosition()
{
    p::tuple shape = p::make_tuple(3);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray array = np::empty(shape,dtype);

    float* dest = reinterpret_cast<float*>(array.get_data());
    for(int i =0;i < 3;i++)
    {
        dest[i] = info_left_foot_pos.back()[i];
    }

    return array;
}

np::ndarray
PushSim::
getFootPlacementPosition()
{
    p::tuple shape = p::make_tuple(3);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray array = np::empty(shape,dtype);

    float* dest = reinterpret_cast<float*>(array.get_data());
    for(int i =0;i < 3;i++)
    {
        dest[i] = info_right_foot_pos.back()[i];
    }

    return array;
}

np::ndarray
PushSim::
getCOMVelocityFootPlacement()
{
    p::tuple shape = p::make_tuple(3);
    np::dtype dtype = np::dtype::get_builtin<float>();
    np::ndarray array = np::empty(shape,dtype);

    float* dest = reinterpret_cast<float*>(array.get_data());
    for(int i =0;i < 3;i++)
    {
        dest[i] = info_com_vel.back()[i];
    }

    return array;
}

Eigen::VectorXd
PushSim::
getPoseForBvh()
{
    auto skeleton = mEnv->GetCharacter()->GetSkeleton();
    auto &node_names = mEnv->GetCharacter()->GetBVH()->mNodeNames;
    auto &node_map = mEnv->GetCharacter()->GetBVH()->mBVHToSkelMap;
    int pose_idx = 0;
    Eigen::VectorXd pose(3*node_names.size()+3);
    pose.setZero();
//    std::cout << node_names.size() << " " << node_map.size() << std::endl;
//    for(int i=0; i<node_names.size(); i++)
//    {
//        std::cout << node_names[i] <<std::endl;
//    }
    for(int i=0; i<node_names.size(); i++)
    {
//        std::cout << i << " " << node_names[i] << std::endl;
        if (node_map.find(node_names[i]) == node_map.end())
        {
            pose_idx += 3;
        }
        else {
            auto joint = skeleton->getJoint(node_map[node_names[i]]);
//            std::cout << joint->getName() << std::endl;
            Eigen::VectorXd joint_position = joint->getPositions();
//            std::cout << joint_position << std::endl;
            if (i == 0) {
                pose.head(3) = Eigen::Vector3d(0., 98.09, -3.08) + joint_position.segment(3, 3) * 100. / mEnv->GetHeightScale();
                pose.segment(3, 3) = joint_position.head(3);
                pose_idx += 6;
            } else {
                if (joint->getNumDofs() == 1) {
                    pose.segment(pose_idx, 3) = joint_position[0] * ((dart::dynamics::RevoluteJoint *) joint)->getAxis();
                } else if (joint->getNumDofs() == 3) {
                    pose.segment(pose_idx, 3) = joint_position;
                }
                pose_idx += 3;
            }
        }
    }
    return pose;
}

np::ndarray
PushSim::
getMotions()
{
    return toNumPyArray(this->motion);
}