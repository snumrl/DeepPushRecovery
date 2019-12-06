//
// Created by trif on 06/12/2019.
//

#include "PushSim.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

using namespace boost::python;

BOOST_PYTHON_MODULE(pypushexp)
{
    Py_Initialize();
    np::initialize();

    class_<PushSim>("PushSim",init<const std::string &, const std::string &>())
        .def("simulate", &PushSim::simulate)
        .def("setParamedStepParams", &PushSim::setParamedStepParams)
        .def("setPushParams", &PushSim::setPushParams)
        .def("getPushedLength", &PushSim::getPushedLength)
        .def("getPushedStep", &PushSim::getPushedStep)
        .def("getStepLength", &PushSim::getStepLength)
        .def("getWalkingSpeed", &PushSim::getWalkingSpeed)
        .def("getStartTimingFootIC", &PushSim::getStartTimingFootIC)
        .def("getMidTimingFootIC", &PushSim::getMidTimingFootIC)
        .def("getStartTimingTimeFL", &PushSim::getStartTimingTimeFL)
        .def("getMidTimingTimeFL", &PushSim::getMidTimingTimeFL)
        .def("getStartTimingFootFL", &PushSim::getStartTimingFootFL)
        .def("getMidTimingFootFL", &PushSim::getMidTimingFootFL)
    ;
}
