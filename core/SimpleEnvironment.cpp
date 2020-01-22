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
        :mControlHz(30),mSimulationHz(900),mWorld(std::make_shared<World>()),w_q(0.65),w_v(0.1),w_ee(0.15),w_com(0.1)
{
}

void
SimpleEnvironment::
Initialize(const std::string& meta_file,bool load_obj)
{
    std::ifstream ifs(meta_file);
    if(!(ifs.is_open()))
    {
        std::cout<<"Can't read file "<<meta_file<<std::endl;
        return;
    }
    std::string str;
    std::string index;
    std::stringstream ss;
    MASS::Character* character = new MASS::Character();
    while(!ifs.eof())
    {
        str.clear();
        index.clear();
        ss.clear();

        std::getline(ifs,str);
        ss.str(str);
        ss>>index;
        if(!index.compare("use_muscle"))
        {
            std::string str2;
            ss>>str2;
            if(!str2.compare("true"))
                this->SetUseMuscle(true);
            else
                this->SetUseMuscle(false);
        }
        else if(!index.compare("con_hz")){
            int hz;
            ss>>hz;
            this->SetControlHz(hz);
        }
        else if(!index.compare("sim_hz")){
            int hz;
            ss>>hz;
            this->SetSimulationHz(hz);
        }
        else if(!index.compare("sim_hz")){
            int hz;
            ss>>hz;
            this->SetSimulationHz(hz);
        }
        else if(!index.compare("skel_file")){
            std::string str2;
            ss>>str2;

            character->LoadSkeleton(std::string(MASS_ROOT_DIR)+str2,load_obj);
        }
        else if(!index.compare("muscle_file")){
            std::string str2;
            ss>>str2;
            if(this->GetUseMuscle())
                character->LoadMuscles(std::string(MASS_ROOT_DIR)+str2);
        }
        else if(!index.compare("bvh_file")){
            std::string str2,str3;

            ss>>str2>>str3;
            bool cyclic = false;
            if(!str3.compare("true"))
                cyclic = true;
            character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
        }
        else if(!index.compare("reward_param")){
            double a,b,c,d;
            ss>>a>>b>>c>>d;
            this->SetRewardParameters(a,b,c,d);

        }


    }
    ifs.close();


    double kp = 300.0;
    character->SetPDParameters(kp,sqrt(2*kp));
    this->SetCharacter(character);
    this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

    this->Initialize();
}
void
SimpleEnvironment::
Initialize()
{
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

    double t = 0.0;

    if(RSI)
        t = dart::math::random(0.0,mCharacter->GetBVH()->GetMaxTime()*0.9);
    mWorld->setTime(t);
    mCharacter->Reset();

    mAction.setZero();

    std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
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
double exp_of_squared(const Eigen::VectorXd& vec,double w)
{
    return exp(-w*vec.squaredNorm());
}
double exp_of_squared(const Eigen::Vector3d& vec,double w)
{
    return exp(-w*vec.squaredNorm());
}
double exp_of_squared(double val,double w)
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
    if(root_y<1.3)
        isTerminal =true;
    else if (dart::math::isNan(p) || dart::math::isNan(v))
        isTerminal =true;
    else if(mWorld->getTime()>10.0)
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
    double phi = std::fmod(mWorld->getTime(),t_phase)/t_phase;

    p *= 0.8;
    v *= 0.2;

    Eigen::VectorXd state(p.rows()+v.rows()+1);

    state<<p,v,phi;
    return state;
}
void
SimpleEnvironment::
SetAction(const Eigen::VectorXd& a)
{
    mAction = a*0.1;

    double t = mWorld->getTime();

    std::pair<Eigen::VectorXd,Eigen::VectorXd> pv = mCharacter->GetTargetPosAndVel(t,1.0/mControlHz);
    mTargetPositions = pv.first;
    mTargetVelocities = pv.second;

    mSimCount = 0;
    mRandomSampleIndex = rand()%(mSimulationHz/mControlHz);
    mAverageActivationLevels.setZero();
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
