#ifndef __ENV_MANAGER_H__
#define __ENV_MANAGER_H__
#include "Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "NumPyHelper.h"
class EnvManager
{
public:
	EnvManager(std::string meta_file, int num_envs);

	int GetNumState();
	int GetNumAction();
	int GetSimulationHz();
	int GetControlHz();
	int GetNumSteps();
	bool UseMuscle();

	void Step(int id);
	void Reset(bool RSI,int id);
	bool IsEndOfEpisode(int id);
	np::ndarray GetState(int id);
	void SetAction(np::ndarray np_array, int id);
	double GetReward(int id);

	void Steps(int num);
	void StepsAtOnce();
	void Resets(bool RSI);
	np::ndarray IsEndOfEpisodes();
	np::ndarray GetStates();
	void SetActions(np::ndarray np_array);
	np::ndarray GetRewards();

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs(){return mEnvs[0]->GetNumTotalRelatedDofs();};
	int GetNumMuscles(){return mEnvs[0]->GetCharacter()->GetMuscles().size();}
	np::ndarray GetMuscleTorques();
	np::ndarray GetDesiredTorques();
	void SetActivationLevels(np::ndarray np_array);
	
	p::list GetMuscleTuples();

	// for push experi
    bool HasCrouchVariation(){return mEnvs[0]->HasCrouchVariation();}

	// for adaptive sampling
	double GetMarginalParameter(){return mEnvs[0]->GetMarginalParameter();}
	bool UseAdaptiveSampling();
	int GetMarginalStateNum(){return mEnvs[0]->GetMarginalStateNum();}
	np::ndarray SampleMarginalState();
	void SetMarginalSampled(const np::ndarray &_marginal_samples, const p::list &_marginal_sample_cumulative_prob);

private:
	std::vector<MASS::Environment*> mEnvs;

	int mNumEnvs;
};

#endif