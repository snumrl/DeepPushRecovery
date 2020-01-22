#include "AppWindow.h"
#include "SimpleEnvironment.h"
#include "Character.h"
#include "BVH.h"
#include <iostream>
using namespace MASS;
using namespace dart;
using namespace dart::dynamics;
using namespace dart::simulation;
using namespace dart::gui;

AppWindow::
AppWindow(SimpleEnvironment* env,const std::string& nn_path)
	:mEnv(env),mFocus(true),mSimulating(false),mDrawOBJ(false),mDrawShadow(true),mBVHPlaying(false)
{
	mBackground[0] = 1.0;
	mBackground[1] = 1.0;
	mBackground[2] = 1.0;
	mBackground[3] = 1.0;
	SetFocusing();
	mZoom = 0.1;
	mFocus = false;
	mNNLoaded = false;

	mm = p::import("__main__");
	mns = mm.attr("__dict__");
	sys_module = p::import("sys");

	p::str module_dir = (std::string(MASS_ROOT_DIR)+"/python").c_str();
	sys_module.attr("path").attr("insert")(1, module_dir);
	p::exec("import torch",mns);
	p::exec("import torch.nn as nn",mns);
	p::exec("import torch.optim as optim",mns);
	p::exec("import torch.nn.functional as F",mns);
	p::exec("import torchvision.transforms as T",mns);
	p::exec("import numpy as np",mns);
	p::exec("from Model_depth3 import *",mns);
    isCudaAvaliable = p::extract<bool> (p::eval("torch.cuda.is_available()", mns));

	mNNLoaded = true;

	boost::python::str str = ("num_state = "+std::to_string(mEnv->GetNumState())).c_str();
	p::exec(str,mns);
	str = ("num_action = "+std::to_string(mEnv->GetNumAction())).c_str();
	p::exec(str,mns);

	nn_module = p::eval("SimulationNN(num_state,num_action)",mns);
	p::object load = nn_module.attr("load");
    load(nn_path);

    motion.clear();

}

void
AppWindow::
draw()
{	
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];
	mViewMatrix.linear() = A;
	mViewMatrix.translation() = b;

	auto ground = mEnv->GetGround();
	float y = ground->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(ground->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
	
	DrawGround(y);
	DrawSkeleton(mEnv->GetCharacter()->GetSkeleton());

	SetFocusing();
}
void
AppWindow::
keyboard(unsigned char _key, int _x, int _y)
{
	switch (_key)
	{
	case 's': this->Step();break;
	case 'f': mFocus = !mFocus;break;
	case 'r': this->Reset();break;
	case ' ': mSimulating = !mSimulating;break;
	case 'o': mDrawOBJ = !mDrawOBJ;break;

	case 'R': this->Reset(false);break;
    case 'S': this->StepMotion();break;
	case 'p': mBVHPlaying = !mBVHPlaying;break;
	case 'm':
	    int a;
	    double b, c;
	    std::cin >> a >> b >> c;
	    mEnv->SetWalkingParams(a, b, c);
	    break;

	case 27 : exit(0);break;
	default:
		Win3D::keyboard(_key,_x,_y);break;
	}

}
void
AppWindow::
displayTimer(int _val)
{
	if(mSimulating)
		Step();
	else if(mBVHPlaying)
	    StepMotion();
	glutPostRedisplay();
	glutTimerFunc(mDisplayTimeout, refreshTimer, _val);
}
void
AppWindow::
StepMotion()
{
    double t = mEnv->GetWorld()->getTime();
    double dt = 1./mEnv->GetControlHz();
    Eigen::VectorXd p = mEnv->GetCharacter()->GetTargetPositions(t, 1./mEnv->GetControlHz());
    mEnv->GetCharacter()->GetSkeleton()->setPositions(p);
    mEnv->GetCharacter()->GetSkeleton()->computeForwardKinematics(true,false,false);
    mEnv->GetWorld()->setTime(t + dt);
}

void
AppWindow::
Step()
{	
	int num = mEnv->GetSimulationHz()/mEnv->GetControlHz();
	Eigen::VectorXd action;
	if(mNNLoaded)
		action = GetActionFromNN();
	else
		action = Eigen::VectorXd::Zero(mEnv->GetNumAction());
	mEnv->SetAction(action);

	for(int i=0;i<num;i++) {
        mEnv->Step();
    }
    motion.push_back(this->getPoseForBvh());
}
void
AppWindow::
Reset(bool RSI)
{
	mEnv->Reset(RSI);
}
void
AppWindow::
SetFocusing()
{
	if(mFocus)
	{
		mTrans = -mEnv->GetWorld()->getSkeleton("Human")->getRootBodyNode()->getCOM();
		mTrans[1] -= 0.3;

		mTrans *=1000.0;
		
	}
}

static np::ndarray toNumPyArray(const Eigen::VectorXd& vec)
{
	int n = vec.rows();
	p::tuple shape = p::make_tuple(n);
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray array = np::empty(shape,dtype);

	float* dest = reinterpret_cast<float*>(array.get_data());
	for(int i =0;i<n;i++)
	{
		dest[i] = vec[i];
	}

	return array;
}


Eigen::VectorXd
AppWindow::
GetActionFromNN()
{
	p::object get_action;
	get_action= nn_module.attr("get_action");
	Eigen::VectorXd state = mEnv->GetState();
	p::tuple shape = p::make_tuple(state.rows());
	np::dtype dtype = np::dtype::get_builtin<float>();
	np::ndarray state_np = np::empty(shape,dtype);
	
	float* dest = reinterpret_cast<float*>(state_np.get_data());
	for(int i =0;i<state.rows();i++)
		dest[i] = state[i];
	
	p::object temp = get_action(state_np);
	np::ndarray action_np = np::from_object(temp);

	float* srcs = reinterpret_cast<float*>(action_np.get_data());

	Eigen::VectorXd action(mEnv->GetNumAction());
	for(int i=0;i<action.rows();i++)
		action[i] = srcs[i];

	return action;
}


void
AppWindow::
DrawEntity(const Entity* entity)
{
	if (!entity)
		return;
	const auto& bn = dynamic_cast<const BodyNode*>(entity);
	if(bn)
	{
		DrawBodyNode(bn);
		return;
	}

	const auto& sf = dynamic_cast<const ShapeFrame*>(entity);
	if(sf)
	{
		DrawShapeFrame(sf);
		return;
	}
}
void
AppWindow::
DrawBodyNode(const BodyNode* bn)
{	
	if(!bn)
		return;
	if(!mRI)
		return;

	mRI->pushMatrix();
	mRI->transform(bn->getRelativeTransform());

	auto sns = bn->getShapeNodesWith<VisualAspect>();
	for(const auto& sn : sns)
		DrawShapeFrame(sn);

	for(const auto& et : bn->getChildEntities())
		DrawEntity(et);

	mRI->popMatrix();

}
void
AppWindow::
DrawSkeleton(const SkeletonPtr& skel)
{
	DrawBodyNode(skel->getRootBodyNode());
}
void
AppWindow::
DrawShapeFrame(const ShapeFrame* sf)
{
	if(!sf)
		return;

	if(!mRI)
		return;

	const auto& va = sf->getVisualAspect();

	if(!va || va->isHidden())
		return;

	mRI->pushMatrix();
	mRI->transform(sf->getRelativeTransform());

	DrawShape(sf->getShape().get(),va->getRGBA());
	mRI->popMatrix();
}
void
AppWindow::
DrawShape(const Shape* shape,const Eigen::Vector4d& color)
{
	if(!shape)
		return;
	if(!mRI)
		return;

	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_COLOR_MATERIAL);
	glEnable(GL_DEPTH_TEST);
	mRI->setPenColor(color);
	if(mDrawOBJ == false)
	{
		if (shape->is<SphereShape>())
		{
			const auto* sphere = static_cast<const SphereShape*>(shape);
			mRI->drawSphere(sphere->getRadius());
		}
		else if (shape->is<BoxShape>())
		{
			const auto* box = static_cast<const BoxShape*>(shape);
			mRI->drawCube(box->getSize());
		}
		else if (shape->is<CapsuleShape>())
		{
			const auto* capsule = static_cast<const CapsuleShape*>(shape);
			mRI->drawCapsule(capsule->getRadius(), capsule->getHeight());
		}	
	}
	else
	{
		if (shape->is<MeshShape>())
		{
			const auto& mesh = static_cast<const MeshShape*>(shape);
			glDisable(GL_COLOR_MATERIAL);
			mRI->drawMesh(mesh->getScale(), mesh->getMesh());
			float y = mEnv->GetGround()->getBodyNode(0)->getTransform().translation()[1] + dynamic_cast<const BoxShape*>(mEnv->GetGround()->getBodyNode(0)->getShapeNodesWith<dart::dynamics::VisualAspect>()[0]->getShape().get())->getSize()[1]*0.5;
			this->DrawShadow(mesh->getScale(), mesh->getMesh(),y);
		}

	}
	
	glDisable(GL_COLOR_MATERIAL);
}

void
AppWindow::
DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y) 
{
	glDisable(GL_LIGHTING);
	glPushMatrix();
	glScalef(scale[0],scale[1],scale[2]);
	GLfloat matrix[16];
	glGetFloatv(GL_MODELVIEW_MATRIX, matrix);
	Eigen::Matrix3d A;
	Eigen::Vector3d b;
	A<<matrix[0],matrix[4],matrix[8],
	matrix[1],matrix[5],matrix[9],
	matrix[2],matrix[6],matrix[10];
	b<<matrix[12],matrix[13],matrix[14];

	Eigen::Affine3d M;
	M.linear() = A;
	M.translation() = b;
	M = (mViewMatrix.inverse()) * M;

	glPushMatrix();
	glLoadIdentity();
	glMultMatrixd(mViewMatrix.data());
	DrawAiMesh(mesh,mesh->mRootNode,M,y);
	glPopMatrix();
	glPopMatrix();
	glEnable(GL_LIGHTING);
}
void
AppWindow::
DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y)
{
	unsigned int i;
    unsigned int n = 0, t;
    Eigen::Vector3d v;
    Eigen::Vector3d dir(0.4,0,-0.4);
    glColor3f(0.3,0.3,0.3);
    
    // update transform

    // draw all meshes assigned to this node
    for (; n < nd->mNumMeshes; ++n) {
        const struct aiMesh* mesh = sc->mMeshes[nd->mMeshes[n]];

        for (t = 0; t < mesh->mNumFaces; ++t) {
            const struct aiFace* face = &mesh->mFaces[t];
            GLenum face_mode;

            switch(face->mNumIndices) {
                case 1: face_mode = GL_POINTS; break;
                case 2: face_mode = GL_LINES; break;
                case 3: face_mode = GL_TRIANGLES; break;
                default: face_mode = GL_POLYGON; break;
            }
            glBegin(face_mode);
        	for (i = 0; i < face->mNumIndices; i++)
        	{
        		int index = face->mIndices[i];

        		v[0] = (&mesh->mVertices[index].x)[0];
        		v[1] = (&mesh->mVertices[index].x)[1];
        		v[2] = (&mesh->mVertices[index].x)[2];
        		v = M*v;
        		double h = v[1]-y;
        		
        		v += h*dir;
        		
        		v[1] = y+0.001;
        		glVertex3f(v[0],v[1],v[2]);
        	}
            glEnd();
        }

    }

    // draw all children
    for (n = 0; n < nd->mNumChildren; ++n) {
        DrawAiMesh(sc, nd->mChildren[n],M,y);
    }

}
void
AppWindow::
DrawGround(double y)
{
	glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
	glDisable(GL_LIGHTING);
	double width = 0.005;
	int count = 0;
	glBegin(GL_QUADS);
	for(double x = -100.0;x<100.01;x+=1.0)
	{
		for(double z = -100.0;z<100.01;z+=1.0)
		{
			if(count%2==0)
				glColor3f(216.0/255.0,211.0/255.0,204.0/255.0);			
			else
				glColor3f(216.0/255.0-0.1,211.0/255.0-0.1,204.0/255.0-0.1);
			count++;
			glVertex3f(x,y,z);
			glVertex3f(x+1.0,y,z);
			glVertex3f(x+1.0,y,z+1.0);
			glVertex3f(x,y,z+1.0);
		}
	}
	glEnd();
	glEnable(GL_LIGHTING);
}


Eigen::VectorXd
AppWindow::
getPoseForBvh()
{
    auto skeleton = mEnv->GetCharacter()->GetSkeleton();
    auto &node_names = mEnv->GetCharacter()->GetBVH()->mNodeNames;
    auto &node_map = mEnv->GetCharacter()->GetBVH()->mBVHToSkelMap;
    int pose_idx = 0;
    Eigen::VectorXd pose(3*node_names.size()+3);
    pose.setZero();
//    std::cout << node_names.size() << " " << node_map.size() << std::endl;
//    for(int i=0; i<node_names.size(); i++)
//    {
//        std::cout << node_names[i] <<std::endl;
//    }
    for(int i=0; i<node_names.size(); i++)
    {
//        std::cout << i << " " << node_names[i] << std::endl;
        if (node_map.find(node_names[i]) == node_map.end())
        {
            pose_idx += 3;
        }
        else {
            auto joint = skeleton->getJoint(node_map[node_names[i]]);
//            std::cout << joint->getName() << std::endl;
            Eigen::VectorXd joint_position = joint->getPositions();
//            std::cout << joint_position << std::endl;
            if (i == 0) {
                pose.head(3) = Eigen::Vector3d(0., 98.09, -3.08) + joint_position.segment(3, 3) * 100.;
                pose.segment(3, 3) = joint_position.head(3);
                pose_idx += 6;
            } else {
                if (joint->getNumDofs() == 1) {
                    pose.segment(pose_idx, 3) = joint_position[0] * ((dart::dynamics::RevoluteJoint *) joint)->getAxis();
                } else if (joint->getNumDofs() == 3) {
                    pose.segment(pose_idx, 3) = joint_position;
                }
                pose_idx += 3;
            }
        }
    }
    return pose;
}

void
AppWindow::
SaveSkelMotion(const std::string& path) {
	std::ofstream fout;
	fout.open(path.c_str());
	for(int i=0; i<motion.size();i++){
		for(int j=0; j<motion[i].rows();j++){
			fout << motion[i][j] << " ";
		}
		fout << std::endl;
	}
	fout.close();
}