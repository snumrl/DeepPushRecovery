#include "EnvManager.h"
#include "DARTHelper.h"
#include <tuple>
#include <random>
#include <omp.h>

EnvManager::
EnvManager(std::string meta_file, int num_envs)
	:mNumEnvs(num_envs)
{
	dart::math::seedRand();
#ifdef _OPENMP
	omp_set_num_threads(mNumEnvs);
#endif
	for(int i = 0;i<mNumEnvs;i++){
		mEnvs.push_back(new MASS::Environment(i));
		MASS::Environment* env = mEnvs.back();

		env->Initialize(meta_file,false);
		// env->SetUseMuscle(false);
		// env->SetControlHz(30);
		// env->SetSimulationHz(600);
		// env->SetRewardParameters(0.65,0.1,0.15,0.1);

		// MASS::Character* character = new MASS::Character();
		// character->LoadSkeleton(std::string(MASS_ROOT_DIR)+std::string("/data/human.xml"),false);
		// if(env->GetUseMuscle())
		// 	character->LoadMuscles(std::string(MASS_ROOT_DIR)+std::string("/data/muscle.xml"));

		// character->LoadBVH(std::string(MASS_ROOT_DIR)+std::string("/data/motion/walk.bvh"),true);
		
		// double kp = 300.0;
		// character->SetPDParameters(kp,sqrt(2*kp));
		// env->SetCharacter(character);
		// env->SetGround(MASS::BuildFromFile(std::string(MASS_ROOT_DIR)+std::string("/data/ground.xml")));

		// env->Initialize();
	}
}
int
EnvManager::
GetNumState()
{
	return mEnvs[0]->GetNumState();
}
int
EnvManager::
GetNumAction()
{
	return mEnvs[0]->GetNumAction();
}
int
EnvManager::
GetSimulationHz()
{
	return mEnvs[0]->GetSimulationHz();
}
int
EnvManager::
GetControlHz()
{
	return mEnvs[0]->GetControlHz();
}
int
EnvManager::
GetNumSteps()
{
	return mEnvs[0]->GetNumSteps();
}
bool
EnvManager::
UseMuscle()
{
	return mEnvs[0]->GetUseMuscle();
}

void
EnvManager::
Step(int id)
{
	mEnvs[id]->Step();
}
void
EnvManager::
Reset(bool RSI,int id)
{
	mEnvs[id]->Reset(RSI);
}
bool
EnvManager::
IsEndOfEpisode(int id)
{
	return mEnvs[id]->IsEndOfEpisode();
}
np::ndarray 
EnvManager::
GetState(int id)
{
	return toNumPyArray(mEnvs[id]->GetState());
}
void 
EnvManager::
SetAction(np::ndarray np_array, int id)
{
	mEnvs[id]->SetAction(toEigenVector(np_array));
}
double 
EnvManager::
GetReward(int id)
{
	return mEnvs[id]->GetReward();
}

void
EnvManager::
Steps(int num)
{
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
			mEnvs[id]->Step();
	}
}
void
EnvManager::
StepsAtOnce()
{
	int num = this->GetNumSteps();
#pragma omp parallel for
	for (int id = 0;id<mNumEnvs;++id)
	{
		for(int j=0;j<num;j++)
			mEnvs[id]->Step();
	}
}
void
EnvManager::
Resets(bool RSI)
{
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->Reset(RSI);
	}
}
np::ndarray
EnvManager::
IsEndOfEpisodes()
{
	std::vector<bool> is_end_vector(mNumEnvs);
	for (int id = 0;id<mNumEnvs;++id)
	{
		is_end_vector[id] = mEnvs[id]->IsEndOfEpisode();
	}

	return toNumPyArray(is_end_vector);
}
np::ndarray
EnvManager::
GetStates()
{
	Eigen::MatrixXd states(mNumEnvs,this->GetNumState());
	for (int id = 0;id<mNumEnvs;++id)
	{
		states.row(id) = mEnvs[id]->GetState().transpose();
	}

	return toNumPyArray(states);
}
void
EnvManager::
SetActions(np::ndarray np_array)
{
	Eigen::MatrixXd action = toEigenMatrix(np_array);
	for (int id = 0;id<mNumEnvs;++id)
	{
		mEnvs[id]->SetAction(action.row(id).transpose());
	}
}
np::ndarray
EnvManager::
GetRewards()
{
	std::vector<float> rewards(mNumEnvs);
	for (int id = 0;id<mNumEnvs;++id)
	{
		rewards[id] = mEnvs[id]->GetReward();
	}
	return toNumPyArray(rewards);
}
np::ndarray
EnvManager::
GetMuscleTorques()
{
	std::vector<Eigen::VectorXd> mt(mNumEnvs);

#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		mt[id] = mEnvs[id]->GetMuscleTorques();
	}
	return toNumPyArray(mt);
}
np::ndarray
EnvManager::
GetDesiredTorques()
{
	std::vector<Eigen::VectorXd> tau_des(mNumEnvs);
	
#pragma omp parallel for
	for (int id = 0; id < mNumEnvs; ++id)
	{
		tau_des[id] = mEnvs[id]->GetDesiredTorques();
	}
	return toNumPyArray(tau_des);
}

void
EnvManager::
SetActivationLevels(np::ndarray np_array)
{
	std::vector<Eigen::VectorXd> activations =toEigenVectorVector(np_array);
	for (int id = 0; id < mNumEnvs; ++id)
		mEnvs[id]->SetActivationLevels(activations[id]);
}

p::list
EnvManager::
GetMuscleTuples()
{
	p::list all;
	for (int id = 0; id < mNumEnvs; ++id)
	{
		auto& tps = mEnvs[id]->GetMuscleTuples();
		for(int j=0;j<tps.size();j++)
		{
			p::list t;
			t.append(toNumPyArray(tps[j].JtA));
			t.append(toNumPyArray(tps[j].tau_des));
			t.append(toNumPyArray(tps[j].L));
			t.append(toNumPyArray(tps[j].b));
			all.append(t);
		}
		tps.clear();
	}

	return all;
}

bool
EnvManager::
UseAdaptiveSampling()
{
    return mEnvs[0]->GetUseAdaptiveSampling();
}

np::ndarray
EnvManager::
SampleMarginalState()
{
	double crouch_angle, stride_length, walking_speed;
	mEnvs[0]->SampleWalkingParams();
	std::tie(crouch_angle, stride_length, walking_speed) = mEnvs[0]->GetNormalizedWalkingParams();

	if (mEnvs[0]->GetMarginalStateNum() == 4) {
        Eigen::VectorXd marginal_state(4);
        marginal_state << dart::math::random(0., 1.), crouch_angle, stride_length, walking_speed;

        return toNumPyArray(marginal_state);
    }
	else {
        Eigen::VectorXd marginal_state(3);
        marginal_state << dart::math::random(0., 1.), stride_length, walking_speed;

        return toNumPyArray(marginal_state);
	}
}

void
EnvManager::
SetMarginalSampled(const np::ndarray &_marginal_samples, const p::list &_marginal_sample_cumulative_prob)
{
    std::vector<Eigen::VectorXd> marginal_samples = toEigenVectorVector(_marginal_samples);
    std::vector<double> marginal_cumulative_probs(p::len(_marginal_sample_cumulative_prob));
    for (int i=0; i < marginal_cumulative_probs.size(); i++)
        marginal_cumulative_probs[i] = p::extract<double>(_marginal_sample_cumulative_prob[i]);

    for (int id = 0; id < mNumEnvs; ++id)
        mEnvs[id]->SetMarginalSampled(marginal_samples, marginal_cumulative_probs);
}