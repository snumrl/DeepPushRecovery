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
	auto* env = new MASS::SimpleEnvironment();

	env->Initialize();

	Py_Initialize();
	np::initialize();
	glutInit(&argc, argv);

    std::string metadata_name;
    int metadata_type;
    std::cout << "type (0:A*, 1:B*, 2:A, 3:B)? ";
	std::cin >> metadata_type;

    if (metadata_type == 0)
        metadata_name = std::string("torque_push_both_sf_all_adaptive_k1_depth3");
    if (metadata_type == 1)
        metadata_name = std::string("torque_push_both_sf_all_uniform_depth3");
    if (metadata_type == 2)
        metadata_name = std::string("torque_nopush_sf_all_adaptive_k1_depth3");
    if (metadata_type == 3)
        metadata_name = std::string("torque_nopush_sf_all_uniform_depth3");

	MASS::AppWindow* window;
	window = new MASS::AppWindow(env, std::string(MASS_ROOT_DIR)+
		std::string("/nn/done/") + metadata_name+ std::string("/max.pt"));
	
	window->initWindow(1024,768,"gui");
	glutMainLoop();
}
