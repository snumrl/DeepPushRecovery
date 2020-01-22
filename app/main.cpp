#include "AppWindow.h"
#include "SimpleEnvironment.h"
#include "DARTHelper.h"
#include "Character.h"
#include "BVH.h"
#include "Muscle.h"
namespace p = boost::python;
namespace np = boost::python::numpy;
int main(int argc,char** argv)
{
	MASS::SimpleEnvironment* env = new MASS::SimpleEnvironment();

	env->Initialize();

	Py_Initialize();
	np::initialize();
	glutInit(&argc, argv);

	MASS::AppWindow* window;
	window = new MASS::AppWindow(env, std::string(MASS_ROOT_DIR)+
		std::string("/nn/done/torque_push_both_sf_all_adaptive_k1_depth3/max.pt"));
	
	window->initWindow(1024,768,"gui");
	glutMainLoop();
}
