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


    class PushSim {
    public:
        PushSim(const std::string &meta_file, const std::string &nn_path);
        PushSim(const std::string &meta_file, const std::string &nn_path, const std::string &muscle_nn_path);
        ~PushSim();

        void Step();
        void _PushStep();
        void PushStep();
        void PushStep_old();
        void Reset(bool RSI = true);

        Eigen::VectorXd GetActionFromNN();
        Eigen::VectorXd GetActivationFromNN(const Eigen::VectorXd &mt);

        p::object mm, mns, sys_module, nn_module, muscle_nn_module;
        Environment *mEnv;
        bool mNNLoaded;
        bool mMuscleNNLoaded;

        int simulate();
        void simulatePrepare();
        void setParamedStepParams(int crouch_angle, double step_length_ratio, double walk_speed_ratio);
        void setPushParams(int push_step, double push_duration, double push_force, double push_start_timing);
        double getPushedLength();
        double getPushedStep();
        double getStepLength();
        double getWalkingSpeed();
        double getStartTimingTimeIC();
        double getMidTimingTimeIC();
        double getStartTimingFootIC();
        double getMidTimingFootIC();
        double getStartTimingTimeFL();
        double getMidTimingTimeFL();
        double getStartTimingFootFL();
        double getMidTimingFootFL();

        double getMechanicalWork();
        double getTravelDistance();
        double getCostOfTransport();

        np::ndarray getPushedStanceFootPosition();
        np::ndarray getFootPlacementPosition();
        np::ndarray getCOMVelocityFootPlacement();

        void PrintWalkingParams();
        void PrintWalkingParamsSampled();
        double GetSimulationTime();

        bool IsBodyContact(const std::string &name);
        void AddBodyExtForce(const std::string &name, const Eigen::Vector3d &_force);
        Eigen::Vector3d GetBodyPosition(const std::string &name);
        double GetMotionHalfCycleDuration();
        bool IsValid(){return this->valid;}


        // simulation results
        double info_start_time;
        double info_end_time;
        std::vector<Eigen::Vector3d> info_root_pos;
        std::vector<Eigen::Vector3d> info_left_foot_pos;
        std::vector<Eigen::Vector3d> info_right_foot_pos;
        std::vector<Eigen::Vector3d> info_left_foot_pos_with_toe_off;
        std::vector<Eigen::Vector3d> info_right_foot_pos_with_toe_off;

        std::vector<Eigen::Vector3d> info_com_vel;

        double info_start_time_backup;
        Eigen::Vector3d info_root_pos_backup;

        double pushed_step_time;
        double pushed_next_step_time;

        double pushed_step_time_toe_off;
        double pushed_next_step_time_toe_off;

        double push_start_time;
        double push_mid_time;
        double push_end_time;
        Eigen::Vector3d walking_dir;
        Eigen::Vector3d push_force_vec;

        int pushed_step;
        double pushed_length;
        bool valid;
        int stopcode;

        Eigen::Vector3d max_detour_root_pos;
        Eigen::Vector3d max_detour_on_line;

        WalkFSM walk_fsm;

        bool push_ready;

        bool pushed_start;
        Eigen::Vector3d pushed_start_pos;
        Eigen::Vector3d pushed_start_foot_pos;
        Eigen::Vector3d pushed_start_toe_pos;

        bool pushed_mid;
        Eigen::Vector3d pushed_mid_pos;
        Eigen::Vector3d pushed_mid_foot_pos;
        Eigen::Vector3d pushed_mid_toe_pos;

        double travelDistance;

        Eigen::Vector3d last_root_pos;
        Eigen::Vector3d first_root_pos;

        // parameters
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
