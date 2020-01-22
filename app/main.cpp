#include "Window.h"
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

	MASS::Window* window;
	window = new MASS::Window(env);
	
	window->initWindow(1024,768,"gui");
	glutMainLoop();
}
