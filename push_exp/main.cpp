#include "PushWindow.h"

namespace p = boost::python;
namespace np = boost::python::numpy;

int main(int argc,char** argv)
{
	Py_Initialize();
	np::initialize();
	glutInit(&argc, argv);

	MASS::PushWindow* window = nullptr;
    if(argc == 4) {
        window = new MASS::PushWindow(argv[1],argv[2], argv[3]);
    }
    else if(argc == 3) {
		window = new MASS::PushWindow(argv[1], argv[2]);
	}

    if (window != nullptr) {
        window->initWindow(1024, 768, "gui");
        glutMainLoop();
    }

	return 0;
}
