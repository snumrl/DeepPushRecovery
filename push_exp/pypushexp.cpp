//
// Created by trif on 06/12/2019.
//

#include "PushSim.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using MASS::PushSim;
using namespace boost::python;

BOOST_PYTHON_MODULE(pypushexp)
{
    Py_Initialize();
    np::initialize();

    class_<PushSim>("PushSim",init<std::string, std::string>())
        .def(init<std::string, std::string, std::string>())
        .def("simulate_motion", &PushSim::simulate_motion)
        .def("simulate", &PushSim::simulate)
        .def("setParamedStepParams", &PushSim::setParamedStepParams)
        .def("setPushParams", &PushSim::setPushParams)
        .def("IsValid", &PushSim::IsValid)
        .def("getPushedLength", &PushSim::getPushedLength)
        .def("getPushedStep", &PushSim::getPushedStep)
        .def("getStepLength", &PushSim::getStepLength)
        .def("getWalkingSpeed", &PushSim::getWalkingSpeed)
        .def("getStartTimingTimeIC", &PushSim::getStartTimingTimeIC)
        .def("getMidTimingTimeIC", &PushSim::getMidTimingTimeIC)
        .def("getStartTimingFootIC", &PushSim::getStartTimingFootIC)
        .def("getMidTimingFootIC", &PushSim::getMidTimingFootIC)
        .def("getStartTimingTimeFL", &PushSim::getStartTimingTimeFL)
        .def("getMidTimingTimeFL", &PushSim::getMidTimingTimeFL)
        .def("getStartTimingFootFL", &PushSim::getStartTimingFootFL)
        .def("getMidTimingFootFL", &PushSim::getMidTimingFootFL)
        .def("getMechanicalWork", &PushSim::getMechanicalWork)
        .def("getTravelDistance", &PushSim::getTravelDistance)
        .def("getCostOfTransport", &PushSim::getCostOfTransport)
        .def("getPushedStanceFootPosition", &PushSim::getPushedStanceFootPosition)
        .def("getFootPlacementPosition", &PushSim::getFootPlacementPosition)
        .def("getCOMVelocityFootPlacement", &PushSim::getCOMVelocityFootPlacement)
        .def("getMotions", &PushSim::getMotions)
        .def("getPushStartFrame", &PushSim::getPushStartFrame)
        .def("getPushEndFrame", &PushSim::getPushEndFrame)
    ;
}
