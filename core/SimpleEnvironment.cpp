//
// Created by trif on 22/01/2020.
//

#include "SimpleEnvironment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "dart/collision/bullet/bullet.hpp"
using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;

SimpleEnvironment::
SimpleEnvironment()
        :mControlHz(30),mSimulationHz(600),mWorld(std::make_shared<World>()),w_q(0.75),w_v(0.1),w_ee(0.0),w_com(0.15)
{
    world_start_time = 0.;
    bvh_start_time = 0.;

    crouch_angle = 0;
    crouch_angle_index = 0;
    stride_length = 1.12620703;
    walk_speed = 0.9943359644;

    stride_length_mean_vec.clear();
    stride_length_var_vec.clear();
    walk_speed_mean_vec.clear();
    walk_speed_var_vec.clear();

    stride_length_mean_vec.push_back(1.12620703);
    stride_length_mean_vec.push_back(0.9529737358);
    stride_length_mean_vec.push_back(0.9158506655);
    stride_length_mean_vec.push_back(0.8755451448);

    walk_speed_mean_vec.push_back(0.994335964);
    walk_speed_mean_vec.push_back(0.8080297151);
    walk_speed_mean_vec.push_back(0.7880050552);
    walk_speed_mean_vec.push_back(0.7435198328);

    stride_length_var_vec.push_back(0.0323409929);
    stride_length_var_vec.push_back(0.02508595114);
    stride_length_var_vec.push_back(0.02772452640);
    stride_length_var_vec.push_back(0.02817863267);

    walk_speed_var_vec.push_back(0.0692930964);
    walk_speed_var_vec.push_back(0.04421889347);
    walk_speed_var_vec.push_back(0.04899931048);
    walk_speed_var_vec.push_back(0.05194827755);
}

void
SimpleEnvironment::
Initialize()
{
    MASS::Character* character = new MASS::Character();
    character->LoadSkeleton(std::string(MASS_ROOT_DIR)+std::string("/data/human.xml"), false);

    double kp = 300.0;
    character->SetPDParameters(kp,sqrt(2*kp));
    this->SetCharacter(character);
    this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground_app.xml")));

    if(mCharacter->GetSkeleton()==nullptr){
        std::cout<<"Initialize character First"<<std::endl;
        exit(0);
    }
    if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="FreeJoint")
        mRootJointDof = 6;
    else if(mCharacter->GetSkeleton()->getRootBodyNode()->getParentJoint()->getType()=="PlanarJoint")
        mRootJointDof = 3;
    else
        mRootJointDof = 0;
    mNumActiveDof = mCharacter->GetSkeleton()->getNumDofs()-mRootJointDof;
    mWorld->setGravity(Eigen::Vector3d(0,-9.8,0.0));
    mWorld->setTimeStep(1.0/mSimulationHz);
    mWorld->getConstraintSolver()->setCollisionDetector(dart::collision::BulletCollisionDetector::create());
    mWorld->addSkeleton(mCharacter->GetSkeleton());
    mWorld->addSkeleton(mGround);
    mAction = Eigen::VectorXd::Zero(mNumActiveDof);

    Reset(false);
    mNumState = GetState().rows();
}

void
SimpleEnvironment::
Reset(bool RSI)
{
    mWorld->reset();

    mCharacter->GetSkeleton()->clearConstraintImpulses();
    mCharacter->GetSkeleton()->clearInternalForces();
    mCharacter->GetSkeleton()->clearExternalForces();

    crouch_angle = 0;
    crouch_angle_index = 0;
    stride_length = 1.12620703;
    walk_speed = 0.9943359644;

    bvh_start_time = 0.;

    mCharacter->GenerateBvhForPushExp(crouch_angle, stride_length, walk_speed);

    double t = 0.2;

    world_start_time = t;
    mWorld->setTime(t);
    mCharacter->Reset();

    mAction.setZero();

    std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t - bvh_start_time, 1.0/mControlHz);
    mTargetPositions = pv.first;
    mTargetVelocities = pv.second;

    mCharacter->GetSkeleton()->setPositions(mTargetPositions);
    mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);
    mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);
}

void
SimpleEnvironment::
Step()
{
    GetDesiredTorques();
    mCharacter->GetSkeleton()->setForces(mDesiredTorque);

    mWorld->step();

    mSimCount++;
}


Eigen::VectorXd
SimpleEnvironment::
GetDesiredTorques()
{
    Eigen::VectorXd p_des = mTargetPositions;
    p_des.tail(mTargetPositions.rows()-mRootJointDof) += mAction;
    mDesiredTorque = mCharacter->GetSPDForces(p_des);
    return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}
static double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
    return exp(-w*vec.squaredNorm());
}
static double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
    return exp(-w*vec.squaredNorm());
}
static double exp_of_squared(double val,double w)
{
    return exp(-w*val*val);
}


bool
SimpleEnvironment::
IsEndOfEpisode()
{
    bool isTerminal = false;

    Eigen::VectorXd p = mCharacter->GetSkeleton()->getPositions();
    Eigen::VectorXd v = mCharacter->GetSkeleton()->getVelocities();

    double root_y = mCharacter->GetSkeleton()->getBodyNode(0)->getTransform().translation()[1] - mGround->getRootBodyNode()->getCOM()[1];
    if(root_y < 0.9)
        isTerminal =true;
    else if (dart::math::isNan(p) || dart::math::isNan(v))
        isTerminal =true;
    else if(mWorld->getTime() - world_start_time > 10.0)
        isTerminal =true;

    return isTerminal;
}

Eigen::VectorXd
SimpleEnvironment::
GetState()
{
    auto& skel = mCharacter->GetSkeleton();
    dart::dynamics::BodyNode* root = skel->getBodyNode(0);
    int num_body_nodes = skel->getNumBodyNodes() - 1;
    Eigen::VectorXd p,v;

    p.resize( (num_body_nodes-1)*3);
    v.resize((num_body_nodes)*3);

    for(int i = 1;i<num_body_nodes;i++)
    {
        p.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOM(root);
        v.segment<3>(3*(i-1)) = skel->getBodyNode(i)->getCOMLinearVelocity();
    }

    v.tail<3>() = root->getCOMLinearVelocity();

    double t_phase = mCharacter->GetBVH()->GetMaxTime();
    double phi = std::fmod(mWorld->getTime() - bvh_start_time, t_phase)/t_phase;

    p *= 0.8;
    v *= 0.2;

    double crouch_angle_normalized = sin(btRadians((double)crouch_angle)) * 2. / sqrt(3.);
    double stride_length_normalized = (stride_length - stride_length_mean_vec[crouch_angle_index]) / sqrt(stride_length_var_vec[crouch_angle_index]);
    double walk_speed_normalized = (walk_speed - walk_speed_mean_vec[crouch_angle_index]) / sqrt(walk_speed_var_vec[crouch_angle_index]);

    Eigen::VectorXd state(p.rows()+v.rows()+1+3);

    state << p, v, phi, crouch_angle_normalized, stride_length_normalized, walk_speed_normalized;

    return state;
}

void
SimpleEnvironment::
SetAction(const Eigen::VectorXd& a)
{
    mAction = a*0.1;

    double t = mWorld->getTime();

    std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t - bvh_start_time, 1.0/mControlHz);
    mTargetPositions = pv.first;
    mTargetVelocities = pv.second;

    mSimCount = 0;
}

double
SimpleEnvironment::
GetReward()
{
    auto& skel = mCharacter->GetSkeleton();

    Eigen::VectorXd cur_pos = skel->getPositions();
    Eigen::VectorXd cur_vel = skel->getVelocities();

    Eigen::VectorXd p_diff_all = skel->getPositionDifferences(mTargetPositions,cur_pos);
    Eigen::VectorXd v_diff_all = skel->getPositionDifferences(mTargetVelocities,cur_vel);

    Eigen::VectorXd p_diff = Eigen::VectorXd::Zero(skel->getNumDofs());
    Eigen::VectorXd v_diff = Eigen::VectorXd::Zero(skel->getNumDofs());

    const auto& bvh_map = mCharacter->GetBVH()->GetBVHMap();

    for(auto ss : bvh_map)
    {
        auto joint = mCharacter->GetSkeleton()->getBodyNode(ss.first)->getParentJoint();
        int idx = joint->getIndexInSkeleton(0);
        if(joint->getType()=="FreeJoint")
            continue;
        else if(joint->getType()=="RevoluteJoint")
            p_diff[idx] = p_diff_all[idx];
        else if(joint->getType()=="BallJoint")
            p_diff.segment<3>(idx) = p_diff_all.segment<3>(idx);
    }

    auto ees = mCharacter->GetEndEffectors();
    Eigen::VectorXd ee_diff(ees.size()*3);
    Eigen::VectorXd com_diff;

    for(int i =0;i<ees.size();i++)
        ee_diff.segment<3>(i*3) = ees[i]->getCOM();
    com_diff = skel->getCOM();

    skel->setPositions(mTargetPositions);
    skel->computeForwardKinematics(true,false,false);

    com_diff -= skel->getCOM();
    for(int i=0;i<ees.size();i++)
        ee_diff.segment<3>(i*3) -= ees[i]->getCOM()+com_diff;

    skel->setPositions(cur_pos);
    skel->computeForwardKinematics(true,false,false);

    double r_q = exp_of_squared(p_diff,2.0);
    double r_v = exp_of_squared(v_diff,0.1);
    double r_ee = exp_of_squared(ee_diff,40.0);
    double r_com = exp_of_squared(com_diff,10.0);

    double r = r_ee*(w_q*r_q + w_v*r_v);

    return r;
}

void
SimpleEnvironment::
SetWalkingParams(int _crouch_angle, double _stride_length, double _walk_speed)
{
    double t_phase = mCharacter->GetBVH()->GetMaxTime();
    double phi = std::fmod(mWorld->getTime() - bvh_start_time, t_phase)/t_phase;

    crouch_angle = _crouch_angle;
    stride_length = _stride_length;
    walk_speed = _walk_speed;
    
    if (crouch_angle == 0) crouch_angle_index = 0;
    else if (crouch_angle == 20) crouch_angle_index = 1;
    else if (crouch_angle == 30) crouch_angle_index = 2;
    else if (crouch_angle == 60) crouch_angle_index = 3;

    mCharacter->GenerateBvhForPushExp(crouch_angle, stride_length, walk_speed);
    t_phase = mCharacter->GetBVH()->GetMaxTime();
    bvh_start_time = mWorld->getTime() - phi * t_phase;
}
