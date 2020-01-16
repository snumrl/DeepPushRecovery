#ifndef __ENV_WRAPPER_H__
#define __ENV_WRAPPER_H__

#include "Environment.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>
#include "NumPyHelper.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

class EnvWrapper
{
public:
	EnvWrapper(std::string meta_file, int index);
	~EnvWrapper();

	int GetNumState();
	int GetNumAction();
	int GetSimulationHz();
	int GetControlHz();
	int GetNumSteps();
	bool UseMuscle();

	void Step();
	void Reset(bool RSI);
	bool IsEndOfEpisode();
	np::ndarray GetState();
	void SetAction(np::ndarray np_array);
	double GetReward();

    void Steps(int num);
    void StepsAtOnce();

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs();
	int GetNumMuscles();
	np::ndarray GetMuscleTorques();
	np::ndarray GetDesiredTorques();
	void SetActivationLevels(np::ndarray np_array);
	
	p::list GetMuscleTuples();

    // for push experi
    bool HasCrouchVariation(){return mEnv->HasCrouchVariation();}

    // for adaptive sampling
    double GetMarginalParameter(){return mEnv->GetMarginalParameter();}
    bool UseAdaptiveSampling(){return mEnv->GetUseAdaptiveSampling();}
    int GetMarginalStateNum(){return mEnv->GetMarginalStateNum();}
    np::ndarray SampleMarginalState();
    void SetMarginalSampled(const np::ndarray &_marginal_samples, const p::list &_marginal_sample_cumulative_prob);

    // for push experiments
	void SetWalkingParams(int crouch_angle, double stride_length, double walk_speed);
	void SetPushParams(int _push_step, double _push_duration, double _push_force, double _push_start_timing);
	void PrintWalkingParams();
    void PrintWalkingParamsSampled();
    double GetSimulationTime();

    bool IsBodyContact(std::string name);
    void AddBodyExtForce(std::string name, np::ndarray &_force);
    np::ndarray GetBodyPosition(std::string name);
    double GetMotionHalfCycleDuration();

private:
	MASS::Environment* mEnv;
};

#endif  // __ENV_WRAPPER_H__