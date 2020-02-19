#ifndef __MASS_WINDOW_H__
#define __MASS_WINDOW_H__
#include "dart/dart.hpp"
#include "dart/gui/gui.hpp"
#include <boost/python.hpp>
#include <boost/python/numpy.hpp>

namespace p = boost::python;
namespace np = boost::python::numpy;

namespace MASS
{
class SimpleEnvironment;
class AppWindow : public dart::gui::Win3D
{
public:
	AppWindow(SimpleEnvironment* env,const std::string& nn_path);

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

	void DrawShadow(const Eigen::Vector3d& scale, const aiScene* mesh,double y);
	void DrawAiMesh(const struct aiScene *sc, const struct aiNode* nd,const Eigen::Affine3d& M,double y);
	void DrawGround(double y);
	void Step();
	void Reset(bool RSI=true);

	void StepMotion();
    void AddBodyExtForce(const std::string &name, const Eigen::Vector3d &_force);
    double GetSimulationTime();

	Eigen::VectorXd GetActionFromNN();

	Eigen::VectorXd getPoseForBvh();
	void SaveSkelMotion(const std::string& path);
	Eigen::Vector3d GetBodyPosition(const std::string &name);

	std::vector<Eigen::VectorXd> motion;

	p::object mm,mns,sys_module,nn_module;

	SimpleEnvironment* mEnv;
	bool mFocus;
	bool mSimulating;
	bool mDrawOBJ;
	bool mDrawShadow;
	bool mNNLoaded;
	Eigen::Affine3d mViewMatrix;

	bool isCudaAvaliable;
    int mBVHPlaying;

    std::vector<int> push_start_frame;
    std::vector<int> push_end_frame;
    int push_frame_index;
    double push_start_time;
    double push_end_time;
    double push_force;
    Eigen::Vector3d push_force_vec;
    bool push_forced;

    std::vector<int> param_change_frame;
};
};


#endif
