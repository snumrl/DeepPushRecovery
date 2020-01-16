#include "EnvWrapper.h"
#include "DARTHelper.h"

EnvWrapper::
EnvWrapper(std::string meta_file, int index)
{
    mEnv = new MASS::Environment(index);
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

void
EnvWrapper::
Steps(int num)
{
    for(int j=0;j<num;j++)
        mEnv->Step();
}
void
EnvWrapper::
StepsAtOnce()
{
    int num = this->GetNumSteps();
    for(int j=0;j<num;j++)
        mEnv->Step();
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

p::list
EnvWrapper::
GetWalkingParams() {
    p::list param;
    auto tp = mEnv->GetWalkingParams();
    param.append(std::get<0>(tp));
    param.append(std::get<1>(tp));
    param.append(std::get<2>(tp));
    return param;
}
void
EnvWrapper::
SetBvhStr(std::string str)
{
    mEnv->SetBvhStr(str);
}

np::ndarray
EnvWrapper::
SampleMarginalState()
{
    double crouch_angle, stride_length, walking_speed;
    mEnv->SampleWalkingParams();
    std::tie(crouch_angle, stride_length, walking_speed) = mEnv->GetNormalizedWalkingParams();

    if (mEnv->GetMarginalStateNum() == 4) {
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
EnvWrapper::
SetMarginalSampled(const np::ndarray &_marginal_samples, const p::list &_marginal_sample_cumulative_prob)
{
    std::vector<Eigen::VectorXd> marginal_samples = toEigenVectorVector(_marginal_samples);
    std::vector<double> marginal_cumulative_probs(p::len(_marginal_sample_cumulative_prob));
    for (int i=0; i < marginal_cumulative_probs.size(); i++)
        marginal_cumulative_probs[i] = p::extract<double>(_marginal_sample_cumulative_prob[i]);

    mEnv->SetMarginalSampled(marginal_samples, marginal_cumulative_probs);
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

void
EnvWrapper::
SetPushParams(int _push_step, double _push_duration, double _push_force, double _push_start_timing)
{
    mEnv->SetPushParams(_push_step, _push_duration, _push_force, 0., _push_start_timing, 0.);
}

double
EnvWrapper::
GetSimulationTime()
{
    return mEnv->GetWorld()->getTime();
}

bool
EnvWrapper::
IsBodyContact(std::string name)
{
    return mEnv->GetWorld()->getLastCollisionResult().inCollision(
    mEnv->GetCharacter()->GetSkeleton()->getBodyNode(name)
    );
}

void
EnvWrapper::
AddBodyExtForce(std::string name, np::ndarray &_force)
{
    mEnv->GetCharacter()->GetSkeleton()->getBodyNode(name)->addExtForce(toEigenVector(_force));
}

np::ndarray
EnvWrapper::
GetBodyPosition(std::string name)
{
    Eigen::VectorXd translation = mEnv->GetCharacter()->GetSkeleton()->getBodyNode(name)->getTransform().translation();
    return toNumPyArray(translation);
}
double
EnvWrapper::
GetMotionHalfCycleDuration()
{
    return mEnv->GetCharacter()->GetBVH()->GetMaxTime()/2.;
}
