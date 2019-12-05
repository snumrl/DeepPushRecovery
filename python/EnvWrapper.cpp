#include "EnvWrapper.h"
#include "DARTHelper.h"

EnvWrapper::
EnvWrapper(std::string meta_file)
{
    mEnv = new MASS::Environment(0);
	dart::math::seedRand();
	mEnv->Initialize(meta_file);
}

EnvWrapper::
~EnvWrapper()
{
    delete mEnv;
}

int
EnvWrapper::
GetNumState()
{
	return mEnv->GetNumState();
}
int
EnvWrapper::
GetNumAction()
{
	return mEnv->GetNumAction();
}
int
EnvWrapper::
GetSimulationHz()
{
	return mEnv->GetSimulationHz();
}
int
EnvWrapper::
GetControlHz()
{
	return mEnv->GetControlHz();
}
int
EnvWrapper::
GetNumSteps()
{
	return mEnv->GetNumSteps();
}
bool
EnvWrapper::
UseMuscle()
{
	return mEnv->GetUseMuscle();
}
void
EnvWrapper::
Step()
{
	mEnv->Step();
}
void
EnvWrapper::
Reset(bool RSI)
{
	mEnv->Reset(RSI);
}
bool
EnvWrapper::
IsEndOfEpisode()
{
	return mEnv->IsEndOfEpisode();
}
np::ndarray 
EnvWrapper::
GetState()
{
	return toNumPyArray(mEnv->GetState());
}
void 
EnvWrapper::
SetAction(np::ndarray np_array)
{
	mEnv->SetAction(toEigenVector(np_array));
}
double 
EnvWrapper::
GetReward()
{
	return mEnv->GetReward();
}

np::ndarray
EnvWrapper::
GetMuscleTorques()
{
	Eigen::VectorXd mt = mEnv->GetMuscleTorques();
	return toNumPyArray(mt);
}
np::ndarray
EnvWrapper::
GetDesiredTorques()
{
	Eigen::VectorXd tau_des = mEnv->GetDesiredTorques();
	return toNumPyArray(tau_des);
}

void
EnvWrapper::
SetActivationLevels(np::ndarray np_array)
{
	Eigen::VectorXd activation = toEigenVector(np_array);
    mEnv->SetActivationLevels(activation);
}

p::list
EnvWrapper::
GetMuscleTuples()
{
	p::list all;
    auto& tps = mEnv->GetMuscleTuples();
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

	return all;
}

int
EnvWrapper::
GetNumTotalMuscleRelatedDofs()
{
    return mEnv->GetNumTotalRelatedDofs();
}

int
EnvWrapper::
GetNumMuscles()
{
    return mEnv->GetCharacter()->GetMuscles().size();
}

void
EnvWrapper::
PrintWalkingParams()
{
    mEnv->PrintWalkingParams();
}

void
EnvWrapper::
PrintWalkingParamsSampled()
{
    mEnv->PrintWalkingParamsSampled();
}

void
EnvWrapper::
SetWalkingParams(int crouch_angle, double stride_length, double walk_speed)
{
    mEnv->SetWalkingParams(crouch_angle, stride_length, walk_speed);
}