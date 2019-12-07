//
// Created by trif on 06/12/2019.
//

#ifndef MSS_PUSHSIM_H
#define MSS_PUSHSIM_H

#include "Environment.h"
#include <string>
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace MASS {
    class WalkFSM {
    public:
        WalkFSM();
        void reset();
        void check(bool bool_l, bool bool_r);

        bool is_last_sw_r;
        int step_count;
        bool is_double_st;
    };


    class PushSim {
    public:
        PushSim(const std::string &meta_file, const std::string &nn_path);
        PushSim(const std::string &meta_file, const std::string &nn_path, const std::string &muscle_nn_path);
        ~PushSim();

        void Step();
        void Reset(bool RSI = true);

        Eigen::VectorXd GetActionFromNN();
        Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd &mt);

        p::object mm, mns, sys_module, nn_module, muscle_nn_module;
        Environment *mEnv;
        bool mNNLoaded;
        bool mMuscleNNLoaded;

        void simulate();
        void setParamedStepParams(int crouch_angle, double step_length_ratio, double walk_speed_ratio);
        void setPushParams(int push_step, double push_duration, double push_force, double push_start_timing);
        double getPushedLength();
        double getPushedStep();
        double getStepLength();
        double getWalkingSpeed();
        double getStartTimingFootIC();
        double getMidTimingFootIC();
        double getStartTimingTimeFL();
        double getMidTimingTimeFL();
        double getStartTimingFootFL();
        double getMidTimingFootFL();

        void SetWalkingParams(int crouch_angle, double stride_length, double walk_speed);
        void SetPushParams(int _push_step, double _push_duration, double _push_force, double _push_start_timing);
        void PrintWalkingParams();
        void PrintWalkingParamsSampled();
        double GetSimulationTime();

        bool IsBodyContact(const std::string &name);
        void AddBodyExtForce(const std::string &name, const Eigen::Vector3d &_force);
        Eigen::Vector3d GetBodyPosition(const std::string &name);
        double GetMotionHalfCycleDuration();


        double info_start_time;
        double info_end_time;
        std::vector<Eigen::Vector3d> info_root_pos;
        std::vector<Eigen::Vector3d> info_left_foot_pos;
        std::vector<Eigen::Vector3d> info_right_foot_pos;

        double push_start_time;
        double push_end_time;
        Eigen::Vector3d walking_dir;

        int pushed_step;
        double pushed_length;
        bool valid;

        double max_detour_length;
        int max_detour_step_count;

        WalkFSM walk_fsm;

        int push_step;
        double push_duration;
        double push_force;
        double push_start_timing;
        double step_length_ratio;
        double walk_speed_ratio;
        double duration_ratio;
    };

}


#endif //MSS_PUSHSIM_H
