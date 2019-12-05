//
// Created by trif on 05/12/2019.
//
#include "EnvManager.h"
#include "EnvWrapper.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(pymss)
{
    Py_Initialize();
    np::initialize();

    class_<EnvManager>("EnvManager",init<std::string,int>())
        .def("GetNumState",&EnvManager::GetNumState)
        .def("GetNumAction",&EnvManager::GetNumAction)
        .def("GetSimulationHz",&EnvManager::GetSimulationHz)
        .def("GetControlHz",&EnvManager::GetControlHz)
        .def("GetNumSteps",&EnvManager::GetNumSteps)
        .def("UseMuscle",&EnvManager::UseMuscle)
        .def("Step",&EnvManager::Step)
        .def("Reset",&EnvManager::Reset)
        .def("IsEndOfEpisode",&EnvManager::IsEndOfEpisode)
        .def("GetState",&EnvManager::GetState)
        .def("SetAction",&EnvManager::SetAction)
        .def("GetReward",&EnvManager::GetReward)
        .def("Steps",&EnvManager::Steps)
        .def("StepsAtOnce",&EnvManager::StepsAtOnce)
        .def("Resets",&EnvManager::Resets)
        .def("IsEndOfEpisodes",&EnvManager::IsEndOfEpisodes)
        .def("GetStates",&EnvManager::GetStates)
        .def("SetActions",&EnvManager::SetActions)
        .def("GetRewards",&EnvManager::GetRewards)
        .def("GetNumTotalMuscleRelatedDofs",&EnvManager::GetNumTotalMuscleRelatedDofs)
        .def("GetNumMuscles",&EnvManager::GetNumMuscles)
        .def("GetMuscleTorques",&EnvManager::GetMuscleTorques)
        .def("GetDesiredTorques",&EnvManager::GetDesiredTorques)
        .def("SetActivationLevels",&EnvManager::SetActivationLevels)
        .def("GetMuscleTuples",&EnvManager::GetMuscleTuples);

    class_<EnvWrapper>("EnvWrapper",init<std::string>())
        .def("GetNumState",&EnvWrapper::GetNumState)
        .def("GetNumAction",&EnvWrapper::GetNumAction)
        .def("GetSimulationHz",&EnvWrapper::GetSimulationHz)
        .def("GetControlHz",&EnvWrapper::GetControlHz)
        .def("GetNumSteps",&EnvWrapper::GetNumSteps)
        .def("UseMuscle",&EnvWrapper::UseMuscle)
        .def("Step",&EnvWrapper::Step)
        .def("Reset",&EnvWrapper::Reset)
        .def("IsEndOfEpisode",&EnvWrapper::IsEndOfEpisode)
        .def("GetState",&EnvWrapper::GetState)
        .def("SetAction",&EnvWrapper::SetAction)
        .def("GetReward",&EnvWrapper::GetReward)
        .def("GetNumTotalMuscleRelatedDofs",&EnvWrapper::GetNumTotalMuscleRelatedDofs)
        .def("GetNumMuscles",&EnvWrapper::GetNumMuscles)
        .def("GetMuscleTorques",&EnvWrapper::GetMuscleTorques)
        .def("GetDesiredTorques",&EnvWrapper::GetDesiredTorques)
        .def("SetActivationLevels",&EnvWrapper::SetActivationLevels)
        .def("GetMuscleTuples",&EnvWrapper::GetMuscleTuples)
        .def("SetWalkingParams", &EnvWrapper::SetWalkingParams)
        .def("PrintWalkingParams", &EnvWrapper::PrintWalkingParams)
        .def("PrintWalkingParamsSampled", &EnvWrapper::PrintWalkingParamsSampled);
}