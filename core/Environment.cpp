#include "Environment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
#include "dart/collision/bullet/bullet.hpp"

#include <chrono>
#include <boost/algorithm/clamp.hpp>

using namespace dart;
using namespace dart::simulation;
using namespace dart::dynamics;
using namespace MASS;

Environment::
Environment()
	:mControlHz(30),mSimulationHz(900),mWorld(std::make_shared<World>()),mUseMuscle(true),w_q(0.65),w_v(0.1),w_ee(0.15),w_com(0.1)
{
    index = 0;
    crouch_angle_index = 0;
    crouch_angle = 0;
    stride_length = 1.12620703;
    walk_speed = 0.9943359644;
    crouch_angle_set.clear();
    stride_length_mean_vec.clear();
    stride_length_var_vec.clear();
    walk_speed_mean_vec.clear();
    walk_speed_var_vec.clear();
    stride_speed_covar_vec.clear();
    sample_param_as_normal = false;
    walking_param_change = false;

    push_enable = false;
    push_step = 8;
    push_duration = .2;
    push_force = 50.;
    push_start_timing = 50.;
}

Environment::
Environment(int _index)
        :mControlHz(30),mSimulationHz(900),mWorld(std::make_shared<World>()),mUseMuscle(true),w_q(0.65),w_v(0.1),w_ee(0.15),w_com(0.1)
{
    index = _index;
    crouch_angle_index = 0;
    crouch_angle = 0;
    stride_length = 1.12620703;
    walk_speed = 0.9943359644;
    crouch_angle_set.clear();
    stride_length_mean_vec.clear();
    stride_length_var_vec.clear();
    walk_speed_mean_vec.clear();
    walk_speed_var_vec.clear();
    stride_speed_covar_vec.clear();
    sample_param_as_normal = false;
    walking_param_change = false;

    push_enable = false;
    push_step = 8;
    push_duration = .2;
    push_force = 50.;
    push_start_timing = 50.;
}


void
Environment::
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
	MASS::Character* character = new MASS::Character(this->index);
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
			this->SetUseMuscle(!str2.compare("true"));
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
//			character->LoadBVH(std::string(MASS_ROOT_DIR)+str2,cyclic);
		}
		else if(!index.compare("reward_param")){
			double a,b,c,d;
			ss>>a>>b>>c>>d;
			this->SetRewardParameters(a,b,c,d);

		}
		else if(index == "walking_param") {
		    std::string str1;
		    ss >> str1;
            crouch_angle_set.push_back(atoi(str1.c_str()));
		    double a, b, c, d, e;
            ss>>a>>b>>c>>d>>e;
		    stride_length_mean_vec.push_back(a);
            walk_speed_mean_vec.push_back(b);
            stride_length_var_vec.push_back(c);
            stride_speed_covar_vec.push_back(d);
            walk_speed_var_vec.push_back(e);

		}
		else if(index == "walking_sample") {
		    std::string str1;
		    ss >> str1;
		    this->sample_param_as_normal = (str1 == "normal");
		}
	}
	ifs.close();

	if (crouch_angle_set.empty()) {
	    crouch_angle_set.push_back(0);
        stride_length_mean_vec.push_back(1.12620703);
        walk_speed_mean_vec.push_back(0.9943359644);
        stride_length_var_vec.push_back(0.);
        walk_speed_var_vec.push_back(0.);
        stride_speed_covar_vec.push_back(0.);
    }

	crouch_angle_index = 0;
    crouch_angle = crouch_angle_set[0];
    stride_length = stride_length_mean_vec[0];
    walk_speed = walk_speed_mean_vec[0];

    walking_param_change = crouch_angle_set.size() > 1
            || stride_length_var_vec[0] > DBL_EPSILON
            || walk_speed_var_vec[0] > DBL_EPSILON;

    character->GenerateBvhForPushExp(crouch_angle, stride_length, walk_speed);
	
	double kp = 300.0;
	character->SetPDParameters(kp,sqrt(2*kp));
	this->SetCharacter(character);
	this->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

	this->Initialize();
}
void
Environment::
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
	if(mUseMuscle)
	{
		int num_total_related_dofs = 0;
		for(auto m : mCharacter->GetMuscles()){
			m->Update();
			num_total_related_dofs += m->GetNumRelatedDofs();
		}
		mCurrentMuscleTuple.JtA = Eigen::VectorXd::Zero(num_total_related_dofs);
		mCurrentMuscleTuple.L = Eigen::MatrixXd::Zero(mNumActiveDof,mCharacter->GetMuscles().size());
		mCurrentMuscleTuple.b = Eigen::VectorXd::Zero(mNumActiveDof);
		mCurrentMuscleTuple.tau_des = Eigen::VectorXd::Zero(mNumActiveDof);
		mActivationLevels = Eigen::VectorXd::Zero(mCharacter->GetMuscles().size());
	}
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
Environment::
Reset(bool RSI)
{
	mWorld->reset();

	mCharacter->GetSkeleton()->clearConstraintImpulses();
	mCharacter->GetSkeleton()->clearInternalForces();
	mCharacter->GetSkeleton()->clearExternalForces();
	
	double t = 0.0;

    if(walking_param_change) {
        SampleWalkingParams();
        mCharacter->GenerateBvhForPushExp(crouch_angle, stride_length, walk_speed);
    }

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
Environment::
Step()
{	
	if(mUseMuscle)
	{
		int count = 0;
		for(auto muscle : mCharacter->GetMuscles())
		{
			muscle->activation = mActivationLevels[count++];
			muscle->Update();
			muscle->ApplyForceToBody();
		}
		if(mSimCount == mRandomSampleIndex)
		{
			auto& skel = mCharacter->GetSkeleton();
			auto& muscles = mCharacter->GetMuscles();

			int n = skel->getNumDofs();
			int m = muscles.size();
			Eigen::MatrixXd JtA = Eigen::MatrixXd::Zero(n,m);
			Eigen::VectorXd Jtp = Eigen::VectorXd::Zero(n);

			for(int i=0;i<muscles.size();i++)
			{
				auto muscle = muscles[i];
				// muscle->Update();
				Eigen::MatrixXd Jt = muscle->GetJacobianTranspose();
				auto Ap = muscle->GetForceJacobianAndPassive();

				JtA.block(0,i,n,1) = Jt*Ap.first;
				Jtp += Jt*Ap.second;
			}

			mCurrentMuscleTuple.JtA = GetMuscleTorques();
			mCurrentMuscleTuple.L = JtA.block(mRootJointDof,0,n-mRootJointDof,m);
			mCurrentMuscleTuple.b = Jtp.segment(mRootJointDof,n-mRootJointDof);
			mCurrentMuscleTuple.tau_des = mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
			mMuscleTuples.push_back(mCurrentMuscleTuple);
		}
	}
	else
	{
		GetDesiredTorques();
		mCharacter->GetSkeleton()->setForces(mDesiredTorque);
	}

	mWorld->step();
	// Eigen::VectorXd p_des = mTargetPositions;
	// //p_des.tail(mAction.rows()) += mAction;
	// mCharacter->GetSkeleton()->setPositions(p_des);
	// mCharacter->GetSkeleton()->setVelocities(mTargetVelocities);
	// mCharacter->GetSkeleton()->computeForwardKinematics(true,false,false);
	// mWorld->setTime(mWorld->getTime()+mWorld->getTimeStep());

	mSimCount++;
}


Eigen::VectorXd
Environment::
GetDesiredTorques()
{
	Eigen::VectorXd p_des = mTargetPositions;
	p_des.tail(mTargetPositions.rows()-mRootJointDof) += mAction;
	mDesiredTorque = mCharacter->GetSPDForces(p_des);
	return mDesiredTorque.tail(mDesiredTorque.rows()-mRootJointDof);
}
Eigen::VectorXd
Environment::
GetMuscleTorques()
{
	int _index = 0;
	mCurrentMuscleTuple.JtA.setZero();
	for(auto muscle : mCharacter->GetMuscles())
	{
		muscle->Update();
		Eigen::VectorXd JtA_i = muscle->GetRelatedJtA();
		mCurrentMuscleTuple.JtA.segment(_index,JtA_i.rows()) = JtA_i;
		_index += JtA_i.rows();
	}
	
	return mCurrentMuscleTuple.JtA;
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
Environment::
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
	else if(mWorld->getTime()>10.0)
		isTerminal =true;
	
	return isTerminal;
}
Eigen::VectorXd 
Environment::
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

	std::vector<double> normalized_walking_parameters;
	normalized_walking_parameters.clear();
	if (crouch_angle_set.size() > 1) {
        double crouch_angle_normalized = sin(btRadians((double)crouch_angle)) * 2. / sqrt(3.);
        normalized_walking_parameters.push_back(crouch_angle_normalized);
	}

    if (stride_length_var_vec[crouch_angle_index] > DBL_EPSILON) {
        double stride_length_normalized = (stride_length - stride_length_mean_vec[crouch_angle_index]) / sqrt(stride_length_var_vec[crouch_angle_index]);
        normalized_walking_parameters.push_back(stride_length_normalized);
    }

    if (walk_speed_var_vec[crouch_angle_index] > DBL_EPSILON) {
        double walk_speed_normalized = (walk_speed - walk_speed_mean_vec[crouch_angle_index]) / sqrt(walk_speed_var_vec[crouch_angle_index]);
        normalized_walking_parameters.push_back(walk_speed_normalized);
    }

    Eigen::VectorXd state(p.rows()+v.rows()+1 + normalized_walking_parameters.size());

    switch(normalized_walking_parameters.size()){
        case 0:
            state << p, v, phi;
            break;
        case 1:
            state << p, v, phi, normalized_walking_parameters[0];
            break;
        case 2:
            state << p, v, phi, normalized_walking_parameters[0], normalized_walking_parameters[1];
            break;
        case 3:
            state << p, v, phi, normalized_walking_parameters[0], normalized_walking_parameters[1], normalized_walking_parameters[2];
            break;
        default:
            state << p, v, phi;
            break;
    }
	return state;
}
void 
Environment::
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
Environment::
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
Environment::
SampleWalkingParams()
{
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    crouch_angle_index = 0;
    if (crouch_angle_set.size() > 1) {
        std::uniform_int_distribution<int> crouch_angle_distribution(0, crouch_angle_set.size()-1);
        crouch_angle_index = crouch_angle_distribution(generator);
    }
    crouch_angle = crouch_angle_set[crouch_angle_index];

    double stride_length_mean = stride_length_mean_vec[crouch_angle_index];
    double stride_length_var = stride_length_var_vec[crouch_angle_index];
    double walk_speed_mean = walk_speed_mean_vec[crouch_angle_index];
    double walk_speed_var = walk_speed_var_vec[crouch_angle_index];
    double stride_speed_covar = stride_speed_covar_vec[crouch_angle_index];

    if (sample_param_as_normal) {
        // normal sampling
        if (stride_length_var > DBL_EPSILON && walk_speed_var > DBL_EPSILON) {
            // using multivariate gaussian
            Eigen::Vector2d mean;
            mean << stride_length_mean, walk_speed_mean;
            Eigen::Matrix2d covar;
            covar << stride_length_var, stride_speed_covar, stride_speed_covar, walk_speed_var;

            Eigen::Vector2d normalized_val;
            normalized_val.setRandom();
            normalized_val.normalize();
            std::normal_distribution<double> normal(0, 1);
            do {
                normalized_val *= normal(generator);
            }
            while (normalized_val.norm() > 2.4477468307);
            // Mahalanobis distance r < sqrt(-2*ln(1-p)) -> p = 0.95

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
            Eigen::Matrix2d normTransform = eigenSolver.eigenvectors()
                                            * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();

            Eigen::Vector2d sample = mean + normTransform * normalized_val;
            stride_length = sample[0];
            walk_speed = sample[1];
        }

        else if (stride_length_var > DBL_EPSILON) {
            std::normal_distribution<double> stride_length_distribution(stride_length_mean, sqrt(stride_length_var));
            do {
                stride_length = stride_length_distribution(generator);
            } while( abs(stride_length - stride_length_mean) > 2*sqrt(stride_length_var) );
        }

        else if (walk_speed_var > DBL_EPSILON) {
            std::normal_distribution<double> walk_speed_distribution(walk_speed_mean, sqrt(walk_speed_var));
            do {
                walk_speed = walk_speed_distribution(generator);
            } while( abs(walk_speed - walk_speed_mean) > 2*sqrt(walk_speed_var) );
        }
    } else {
        // uniform sampling
        if (stride_length_var > DBL_EPSILON && walk_speed_var > DBL_EPSILON) {
            // using multivariate gaussian
            Eigen::Vector2d mean;
            mean << stride_length_mean, walk_speed_mean;
            Eigen::Matrix2d covar;
            covar << stride_length_var, stride_speed_covar, stride_speed_covar, walk_speed_var;

            Eigen::Vector2d normalized_val;
            normalized_val.setRandom();
            normalized_val.normalize();
            // Mahalanobis distance r < sqrt(-2*ln(1-p)) -> p = 0.95
            std::uniform_real_distribution<double> uniform(-2.4477468307, 2.4477468307);
            normalized_val *= uniform(generator);

            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigenSolver(covar);
            Eigen::Matrix2d normTransform = eigenSolver.eigenvectors()
                                            * eigenSolver.eigenvalues().cwiseSqrt().asDiagonal();

            Eigen::Vector2d sample = mean + normTransform * normalized_val;
            stride_length = sample[0];
            walk_speed = sample[1];

        }
        else if (stride_length_var > DBL_EPSILON) {
            std::uniform_real_distribution<double>
                    stride_length_distribution(stride_length_mean - 2 * sqrt(stride_length_var),
                                               stride_length_mean + 2 * sqrt(stride_length_var));
            stride_length = stride_length_distribution(generator);
        }
        else if (walk_speed_var > DBL_EPSILON) {
            std::uniform_real_distribution<double>
                    walk_speed_distribution(walk_speed_mean - 2 * sqrt(walk_speed_var),
                                            walk_speed_mean + 2 * sqrt(walk_speed_var));
            walk_speed = walk_speed_distribution(generator);
        }
    }
    // std::cout << "Environment::SampleWalkingParams: " << crouch_angle << " " << stride_length << " " << walk_speed << std::endl;
}

void
Environment::
SetWalkingParams(int _crouch_angle, double _stride_length, double _walk_speed)
{
    crouch_angle = _crouch_angle;
    stride_length = _stride_length;
    walk_speed = _walk_speed;
    walking_param_change = false;
    mCharacter->GenerateBvhForPushExp(crouch_angle, stride_length, walk_speed);
    this->Reset(false);
}
void
Environment::
SetPushParams(int _push_step, double _push_duration, double _push_force, double _push_start_timing)
{
    push_step = _push_step;
    push_duration = _push_duration;
    push_force = _push_force;
    push_start_timing = _push_start_timing;
    push_enable = true;
}

void
Environment::
PrintPushParamsSampled()
{
    std::cout << "push step: " << push_step << " steps" << std::endl;
    std::cout << "push duration: " << push_duration << " s" << std::endl;
    std::cout << "push force: " << push_force << " N" << std::endl;
    std::cout << "push start timing: " << push_start_timing << " %" << std::endl;
    std::cout << std::endl;
}

void
Environment::
PrintWalkingParamsSampled()
{
    std::cout << "crouch angle: " << crouch_angle << " degree" << std::endl;
    std::cout << "stride length: " << stride_length << " m" << std::endl;
    std::cout << "walking speed: " << walk_speed << " m/s" << std::endl;
    std::cout << std::endl;
}

void
Environment::
PrintWalkingParams() {
    std::cout << "Walking parameter info: ";
    for (int i = 0; i < crouch_angle_set.size(); i++) {
        std::cout << "crouch " << crouch_angle_set[i] << std::endl;
        std::cout << "stride length mean: " << stride_length_mean_vec[i] << " m" << std::endl;
        std::cout << "stride length std: " << sqrt(stride_length_var_vec[i]) << " m" << std::endl;
        std::cout << "walking speed mean: " << walk_speed_mean_vec[i] << " m/s" << std::endl;
        std::cout << "walking speed std: " << sqrt(walk_speed_var_vec[i]) << " m/s" << std::endl;
        if (sample_param_as_normal)
            std::cout << "stride_speed corel: " << stride_speed_covar_vec[i]/(sqrt(stride_length_var_vec[i])*sqrt(walk_speed_var_vec[i])) << " m/s" << std::endl;
        std::cout << std::endl;
    }
    if (sample_param_as_normal) {
        std::cout << "normal distribution" << std::endl;
    }
    else {
        std::cout << "uniform distribution" << std::endl;
    }
}


void
Environment::
SampleStrategy(int flag)
{
    if (flag == 0)
        // uniform sampling
        sample_param_as_normal = false;
    if (flag == 1)
        // normal sampling
        sample_param_as_normal = true;

    // TODO:
    // adaptive sampling...to be implemented
    if (flag == 2);
}
