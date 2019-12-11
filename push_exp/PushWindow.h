#ifndef __MASS_PUSHWINDOW_H__
#define __MASS_PUSHWINDOW_H__
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include "PushSim.h"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace MASS
{
class Environment;
class Muscle;
class PushWindow : public dart::gui::Win3D, public PushSim
{
public:
    PushWindow(const std::string &meta_file, const std::string &nn_path);
    PushWindow(const std::string &meta_file, const std::string &nn_path, const std::string &muscle_nn_path);

	void draw() override;
	void keyboard(unsigned char _key, int _x, int _y) override;
	void displayTimer(int _val) override;
private:
	void SetFocusing();

	void DrawEntity(const dart::dynamics::Entity* entity);
	void DrawBodyNode(const dart::dynamics::BodyNode* bn);
	void DrawSkeleton(const dart::dynamics::SkeletonPtr& skel);
	void DrawShapeFrame(const dart::dynamics::ShapeFrame* shapeFrame);
	void DrawShape(const dart::dynamics::Shape* shape,const Eigen::Vector4d& color);
	void DrawTrajectory();
    void DrawWalkingDir();
    void DrawDetourPerp();
    void DrawPush();

	void DrawMuscles(const std::vector<Muscle*>& muscles);
	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	void DrawGround(double y);

	void StepMotion();

    void SamplePushForce();
    void PrintPushParamsSampled();

    bool mFocus;
	bool mSimulating;
	bool mDrawOBJ;
	bool mDrawShadow;
	Eigen::Affine3d mViewMatrix;

    int mBVHPlaying;
    std::vector<Eigen::Vector3d> mRootTrajectory;
};
};


#endif