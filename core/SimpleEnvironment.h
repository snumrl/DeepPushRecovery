//
// Created by trif on 22/01/2020.
//

#ifndef MSS_SIMPLEENVIRONMENT_H
#define MSS_SIMPLEENVIRONMENT_H
#include "dart/dart.hpp"
#include "Character.h"
namespace MASS
{
    class SimpleEnvironment
    {
    public:
        SimpleEnvironment();

        void SetControlHz(int con_hz) {mControlHz = con_hz;}
        void SetSimulationHz(int sim_hz) {mSimulationHz = sim_hz;}

        void SetCharacter(Character* character) {mCharacter = character;}
        void SetGround(const dart::dynamics::SkeletonPtr& ground) {mGround = ground;}

        void SetRewardParameters(double w_q,double w_v,double w_ee,double w_com){this->w_q = w_q;this->w_v = w_v;this->w_ee = w_ee;this->w_com = w_com;}
        void Initialize();
    public:
        void Step();
        void Reset(bool RSI = true);
        bool IsEndOfEpisode();
        Eigen::VectorXd GetState();
        void SetAction(const Eigen::VectorXd& a);
        double GetReward();

        Eigen::VectorXd GetDesiredTorques();

        const dart::simulation::WorldPtr& GetWorld(){return mWorld;}
        Character* GetCharacter(){return mCharacter;}
        const dart::dynamics::SkeletonPtr& GetGround(){return mGround;}
        int GetControlHz(){return mControlHz;}
        int GetSimulationHz(){return mSimulationHz;}
        int GetNumState(){return mNumState;}
        int GetNumAction(){return mNumActiveDof;}
        int GetNumSteps(){return mSimulationHz/mControlHz;}
        void SetWalkingParams(int _crouch_angle, double _stride_length, double _walk_speed);
    private:
        dart::simulation::WorldPtr mWorld;
        int mControlHz,mSimulationHz;
        Character* mCharacter;
        dart::dynamics::SkeletonPtr mGround;
        Eigen::VectorXd mAction;
        Eigen::VectorXd mTargetPositions,mTargetVelocities;

        double world_start_time;
        double bvh_start_time;

        int mNumState;
        int mNumActiveDof;
        int mRootJointDof;

        Eigen::VectorXd mDesiredTorque;
        int mSimCount;

        double w_q,w_v,w_ee,w_com;

        double phase; // for adaptive sampling
        int crouch_angle;
        int crouch_angle_index;
        double stride_length;
        double walk_speed;

        std::vector<double> stride_length_mean_vec;
        std::vector<double> stride_length_var_vec;
        std::vector<double> walk_speed_mean_vec;
        std::vector<double> walk_speed_var_vec;
    };
};


#endif //MSS_SIMPLEENVIRONMENT_H
