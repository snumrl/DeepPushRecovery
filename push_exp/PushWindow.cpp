#include "PushWindow.h"
#include "Muscle.h"
#include <iostream>
using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

PushWindow::PushWindow(const std::string &meta_file, const std::string &nn_path)
	:PushSim(meta_file, nn_path),mFocus(true),mSimulating(false),mDrawOBJ(false),mDrawShadow(true),mBVHPlaying(false)
{
	mEnv->PrintWalkingParams();
	// normally sample
    this->mEnv->SetSampleStrategy(1);

	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 1.0;
	mBackground[3] = 1.0;
	SetFocusing();
	mZoom = 0.1;
	mFocus = false;
	mRootTrajectory.clear();
}

PushWindow::PushWindow(const std::string &meta_file, const std::string &nn_path, const std::string &muscle_nn_path)
    :PushSim(meta_file, nn_path, muscle_nn_path),mFocus(true),mSimulating(false),mDrawOBJ(false),mDrawShadow(true),mBVHPlaying(false)
{
    mEnv->PrintWalkingParams();

    mBackground[0] = 1.0;
    mBackground[1] = 1.0;
    mBackground[2] = 1.0;
    mBackground[3] = 1.0;
    SetFocusing();
    mZoom = 0.1;
    mFocus = false;
    mRootTrajectory.clear();
}

void PushWindow::keyboard(unsigned char _key, int _x, int _y) {
	switch (_key)
	{
	case 's':
	    this->Step();
        mRootTrajectory.push_back(GetBodyPosition("Pelvis"));
	    break;

	case 'f':
	    mFocus = !mFocus;
	    break;

	case 'z':
	    mSimulating = !mSimulating;
	    break;

	case 'o':
	    mDrawOBJ = !mDrawOBJ;
	    break;

    case 'S':
        this->StepMotion();
        mRootTrajectory.push_back(GetBodyPosition("Pelvis"));
        break;

	case 'p':
	    mBVHPlaying = !mBVHPlaying;
	    break;

	case 'm':
	    int a;
	    double b, c, d, e;
	    std::cin >> a >> b >> c >> d >> e;
        this->mEnv->SetWalkingParams(a, b, c);
        this->mEnv->PrintWalkingParamsSampled();
//	    this->setParamedStepParams(a, b, c);
        this->setPushParams(8, 0.2, d, e);
    case 'r':
        this->Reset(false);
        mRootTrajectory.clear();
        simulatePrepare();
        std::cout << "Reset!" << std::endl;
        break;

    case ' ':
    case 'R':
        this->Reset(false);
        mRootTrajectory.clear();
        this->mEnv->PrintWalkingParamsSampled();
        this->SamplePushForce();
        simulatePrepare();
        std::cout << "Reset!" << std::endl;
        mSimulating = true;
        break;

	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}

}

void PushWindow::displayTimer(int _val) {
	if(this->valid && mSimulating) {
        PushStep();
        mRootTrajectory.push_back(GetBodyPosition("Pelvis"));
        if(this->GetSimulationTime() >= push_start_time + 10.)
        {
            if(pushed_step == 0) {
                this->valid = false;
                this->stopcode = 4;
            }
            mSimulating = !mSimulating;
            if(this->valid){
                std::cout << "PushedLength: " << getPushedLength() << std::endl;
                std::cout << "PushedStep: " << getPushedStep() << std::endl;
                std::cout << "StepLength: " << getStepLength() << std::endl;
                std::cout << "WalkingSpeed: " << getWalkingSpeed() << std::endl;
                std::cout << "StartTimingTimeIC: " << getStartTimingTimeIC() << std::endl;
                std::cout << "MidTimingTimeIC: " << getMidTimingTimeIC() << std::endl;
                std::cout << "StartTimingFootIC: " << getStartTimingFootIC() << std::endl;
                std::cout << "MidTimingFootIC: " << getMidTimingFootIC() << std::endl;
                std::cout << "StartTimingTimeFL: " << getStartTimingTimeFL() << std::endl;
                std::cout << "MidTimingTimeFL: " << getMidTimingTimeFL() << std::endl;
                std::cout << "StartTimingFootFL: " << getStartTimingFootFL() << std::endl;
                std::cout << "MidTimingFootFL: " << getMidTimingFootFL() << std::endl;
                std::cout << "MechanicalWork: " << getMechanicalWork() << std::endl;
                std::cout << "TravelDistance: " << getTravelDistance() << std::endl;
                std::cout << "CostOfTransport: " << getCostOfTransport() << std::endl;
            }
            std::cout << "end!" << this->stopcode << " " << this->walk_fsm.step_count << " steps"<< std::endl;

        }
        if (this->GetBodyPosition("Pelvis")[1] < 0.3) {
            this->valid = false;
            if (this->pushed_start)
                this->stopcode = 2; // falling down after push
            else
                this->stopcode = 1; // falling down before push
            mSimulating = !mSimulating;
            std::cout << "end!" << this->stopcode << " " << this->walk_fsm.step_count << " steps"<< std::endl;
        }
    }
	else if(!this->valid && mSimulating){
	    mSimulating = !mSimulating;
        std::cout << "end!" << this->stopcode << " " << this->walk_fsm.step_count << " steps"<< std::endl;
    }


    glutPostRedisplay();
	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
}

void PushWindow::SamplePushForce() {
    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<double> push_force_dist(.535, .096);
    std::normal_distribution<double> push_timing_dist(34, 21);
    this->setPushParams(8, .2, -abs(push_force_dist(generator) * 72.*5.), abs(push_timing_dist(generator)));
}
