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
	explicit EnvWrapper(std::string meta_file);
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

	//For Muscle Transitions
	int GetNumTotalMuscleRelatedDofs();
	int GetNumMuscles();
	np::ndarray GetMuscleTorques();
	np::ndarray GetDesiredTorques();
	void SetActivationLevels(np::ndarray np_array);
	
	p::list GetMuscleTuples();

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