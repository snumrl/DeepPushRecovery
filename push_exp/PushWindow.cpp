#include "PushWindow.h"
#include "Muscle.h"
#include <iostream>
#include "DARTHelper.h"
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
    mEnv->SetPushEnable(false);

    balls.clear();
    balls_in_world.clear();

	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 1.0;
	mBackground[3] = 1.0;
	SetFocusing();
	mZoom = 0.1;
	mFocus = false;
	mRootTrajectory.clear();
    mRightFootTrajectory.clear();

    human_pos_temp.clear();
    human_pos_temp_idx = 0;
//    std::ifstream file("/Users/trif/works/mass/hihi.txt");
//    std::string str;
//    std::stringstream ss;
//    while (std::getline(file, str)) {
//        ss.str(str);
//        Eigen::VectorXd v(56);
//        for(int i=0; i<56; i++)
//            ss >> v[i];
//        human_pos_temp.push_back(v);
//    }
//    file.close();
}

PushWindow::PushWindow(const std::string &meta_file, const std::string &nn_path, const std::string &muscle_nn_path)
    :PushSim(meta_file, nn_path, muscle_nn_path),mFocus(true),mSimulating(false),mDrawOBJ(false),mDrawShadow(true),mBVHPlaying(false)
{
    mEnv->PrintWalkingParams();
    this->mEnv->SetSampleStrategy(1);
    mEnv->SetPushEnable(false);

    mBackground[0] = 1.0;
    mBackground[1] = 1.0;
    mBackground[2] = 1.0;
    mBackground[3] = 1.0;
    SetFocusing();
    mZoom = 0.1;
    mFocus = false;
    mRootTrajectory.clear();
    mRightFootTrajectory.clear();
}

void PushWindow::keyboard(unsigned char _key, int _x, int _y) {
    std::ofstream fout;
    Eigen::VectorXd p;
    switch (_key)
	{
	case 's':
	    mSimulating = false;
        PushStep();
        mRootTrajectory.push_back(GetBodyPosition("Pelvis"));
        mRightFootTrajectory.push_back(GetBodyPosition("TalusR"));
        if(this->GetSimulationTime() >= push_start_time + 10.)
        {
            if (this->valid)
                CheckPushedStep();

            if(pushed_step == 0) {
                this->valid = false;
                this->stopcode = 4;
            }
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
            std::cout << "end!" << this->stopcode << " " << this->walk_fsm.step_count << " steps"<< std::endl;
        }
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
        mRightFootTrajectory.push_back(GetBodyPosition("TalusR"));
        break;

	case 'p':
	    mBVHPlaying = !mBVHPlaying;
	    break;

    case 'b':
        this->AddBall();
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
        mRightFootTrajectory.clear();
        simulatePrepare();
        std::cout << "Reset!" << std::endl;
        std::cout << std::endl;
        break;

    case ' ':
        ///////////////////////////////////
        mEnv->SetWalkingParams(30, 0.9158506655, 0.7880050552);
//        mEnv->SetWalkingParams(30, 0.66609, 0.456);
//          mEnv->SetWalkingParams(30, 1.2488643372914146, 1.2307208126787688);
//        mEnv->SetWalkingParams(0, 1.1818643660084964,1.1818643660084964/0.9182594956257297);
//        mEnv->SetWalkingParams(0, 1.12620703, 0.994335964);
        ///////////////////////////////////
        this->Reset(false);
        mRootTrajectory.clear();
        mRightFootTrajectory.clear();
        this->mEnv->PrintWalkingParamsSampled();
        this->SamplePushForce();
        this->PrintPushParamsSampled();
        simulatePrepare();
        std::cout << "Reset!" << std::endl;
        fout.open("bvh.txt", std::ios::out);
        p = getPoseForBvh();
        for(int i=0; i<p.rows(); i++)
            fout << p[i] << " ";
        fout << std::endl;
        fout.close();
//        fout.open("ball_motion.txt", std::ios::out);
//        fout << 0;
//        fout << std::endl;
//        fout.close();
        mSimulating = true;
        break;
    case 'R':
        ///////////////////////////////////
//        mEnv->SetWalkingParams(30, 0.9158506655, 0.7880050552);
        ///////////////////////////////////
        this->Reset(false);
        mRootTrajectory.clear();
        mRightFootTrajectory.clear();
        this->mEnv->PrintWalkingParamsSampled();
        this->SamplePushForce();
        this->PrintPushParamsSampled();
        simulatePrepare();
        std::cout << "Reset!" << std::endl;
        mSimulating = false;
        break;

	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}

}

void PushWindow::displayTimer(int _val) {
	if(this->valid && mSimulating) {
//        auto world = mEnv->GetWorld();
//        for (int i =10;i<=25;i++) {
//            if (world->getTime() > i && world->getTime() < i + 0.033)
//                AddBall();
//        }
//        for (int i =15;i<=25;i++) {
//            if (world->getTime() > i+0.5 && world->getTime() < i + 0.533)
//                AddBall();
//        }
//        for (int i =20;i<=25;i++) {
//            if (world->getTime() > i+0.125 && world->getTime() < i + 0.158)
//                AddBall();
//            if (world->getTime() > i+0.25 && world->getTime() < i + 0.283)
//                AddBall();
//            if (world->getTime() > i+0.375 && world->getTime() < i + 0.408)
//                AddBall();
//            if (world->getTime() > i+0.625 && world->getTime() < i + 0.658)
//                AddBall();
//            if (world->getTime() > i+0.75 && world->getTime() < i + 0.783)
//                AddBall();
//            if (world->getTime() > i+0.875 && world->getTime() < i + 0.908)
//                AddBall();
//        }
        PushStep();
        std::ofstream fout;
        fout.open("bvh.txt", std::ios::out | std::ios::app);
        auto p = getPoseForBvh();
        for(int i=0; i<p.rows(); i++)
            fout << p[i] << " ";
        fout << std::endl;
        fout.close();
//
//        fout.open("ball_motion.txt", std::ios::out | std::ios::app);
//        int ball_count = 0;
//        for(int i=0; i<balls_in_world.size(); i++)
//            if(balls_in_world[i])
//                ball_count++;
//        fout << ball_count << " ";
//        for(int i=0; i<balls_in_world.size(); i++) {
//            if (balls_in_world[i]) {
//                auto com = balls[i]->getRootBodyNode()->getCOM();
//                fout << com[0] << " " << com[1] << " " << com[2] << " ";
//            }
//        }
//        fout << std::endl;
//        fout.close();
//
//        auto collision_result = world->getLastCollisionResult();
//        auto &contacts = collision_result.getContacts();
//        for (auto &contact : contacts) {
//            auto shape1_name = contact.collisionObject1->getShapeFrame()->asShapeNode()->getBodyNodePtr()->getName();
//            auto shape2_name = contact.collisionObject2->getShapeFrame()->asShapeNode()->getBodyNodePtr()->getName();
////            std::cout << shape1_name << " " << shape2_name << std::endl;
//            if (shape1_name == "ground" && shape2_name.find(std::string("Ball")) != std::string::npos) {
//                int idx = atoi(&shape2_name.c_str()[4]);
//                world->removeSkeleton(balls[idx]);
//                balls_in_world[idx] = false;
//            }
//            else if (shape2_name == "ground" && shape1_name.find(std::string("Ball")) != std::string::npos) {
//                int idx = atoi(&shape1_name.c_str()[4]);
//                world->removeSkeleton(balls[idx]);
//                balls_in_world[idx] = false;
//            }
//        }
        // mRootTrajectory.push_back(GetBodyPosition("Pelvis"));
        mRootTrajectory.push_back(this->mEnv->GetCharacter()->GetSkeleton()->getCOM());
        mRightFootTrajectory.push_back(GetBodyPosition("TalusR"));
        if(this->GetSimulationTime() >= 30.)
//        if(this->GetSimulationTime() >= push_start_time + 10.)
        {
            if (this->valid)
                CheckPushedStep();
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
/////////////////////////////
//    defalut
//    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
//    std::normal_distribution<double> push_force_dist(.535, .096);
//    std::normal_distribution<double> push_timing_dist(34, 21);
//    this->setPushParams(8, .2, -abs(push_force_dist(generator) * 72.*5.), abs(push_timing_dist(generator)));
/////////////////////////////
//    specific
//    std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
//    std::normal_distribution<double> push_force_dist(.535, .096);
//    std::normal_distribution<double> push_force_dist(.500, 0);
//    std::normal_distribution<double> push_timing_dist(10, 0);
//    this->setPushParams(8, .2, -abs(push_force_dist(generator) * 72.*5.), abs(push_timing_dist(generator)));
/////////////////////////////
//    user input
//    double push_force;
//    double push_timing;
//    std::cout << "Please Provide force" << std::endl;
//    std::cin >> push_force;
//    push_timing = 34;
//    this->setPushParams(8, .2, -abs(push_force), abs(push_timing));
/////////////////////////////
//    zero
    this->setPushParams(8, .2, 0., 0.);
}

void PushWindow::PrintPushParamsSampled() {
    std::cout << "push_step: " << this->push_step << std::endl;
    std::cout << "push_duration: " << this->push_duration << std::endl;
    std::cout << "push_force: " << this->push_force << std::endl;
    std::cout << "push_start_timing: " << this->push_start_timing << std::endl;

}

void
PushWindow::AddBall()
{
    char buf[32];
    sprintf(buf, "Ball%04lu", this->balls.size());
    this->balls.push_back(Skeleton::create(std::string(buf)));
    SkeletonPtr skel = this->balls.back();
    ShapePtr shape = MASS::MakeSphereShape(0.1);
    double mass = 1.;
    dart::dynamics::Inertia inertia = MakeInertia(shape,mass);
    Joint::Properties * props = MASS::MakeFreeJointProperties(std::string(buf));
    auto bn = MASS::MakeBodyNode(skel, nullptr, props, std::string("Free"), inertia);
    bn->createShapeNodeWith<VisualAspect,CollisionAspect,DynamicsAspect>(shape);
    // bn->getShapeNodesWith<VisualAspect>().back()->getVisualAspect()->setColor(color);

    Eigen::Vector3d torso_pos = this->GetBodyPosition(std::string("Torso"));

    Eigen::Vector3d ball_vec;
    ball_vec.setRandom();
    ball_vec.normalize();
    if (ball_vec[1] < 0.) ball_vec[1] = -ball_vec[1];

    Eigen::VectorXd pos(6);
    Eigen::VectorXd vel(6);
    pos << 0., 0., 0., torso_pos[0], torso_pos[1], torso_pos[2];
    pos.tail(3) += 1.5 * ball_vec;
    vel << 0., 0., 0., 0., 0., 0.;
    vel.tail(3) -= 10.*ball_vec;
    skel->setPositions(pos);
    skel->setVelocities(vel);
    mEnv->GetWorld()->addSkeleton(skel);
    this->balls_in_world.push_back(true);
}

