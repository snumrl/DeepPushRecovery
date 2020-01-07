#include "PushWindow.h"
#include <glob.h>
#include <sstream>

namespace p = boost::python;
namespace np = boost::python::numpy;
using std::string;

int main(int argc,char** argv)
{
	Py_Initialize();
	np::initialize();
	glutInit(&argc, argv);

	MASS::PushWindow* window = nullptr;
	std::string metadata_name, pt_name, meta_file_path, nn_dir_path;
	std::cout << "Please provide metadata name: ";
	std::cin >> metadata_name;
    std::cout << "Please provide pt file prefix: ";
    std::cin >> pt_name;
	meta_file_path = string(MASS_ROOT_DIR)+string("/data/metadata/")+metadata_name+string(".txt");
	nn_dir_path = string(MASS_ROOT_DIR)+string("/nn/*/")+metadata_name+string("/")+pt_name+string("*.pt");

    // glob struct resides on the stack
    glob_t glob_result;
    memset(&glob_result, 0, sizeof(glob_result));

    // do the glob operation
    int return_value = glob(nn_dir_path.c_str(), GLOB_TILDE, NULL, &glob_result);
    if(return_value != 0) {
        globfree(&glob_result);
        std::stringstream ss;
        ss << "glob() failed with return_value " << return_value << std::endl;
        throw std::runtime_error(ss.str());
    }

    // collect all the filenames into a std::list<std::string>
    std::vector<std::string> filenames;
    for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
        filenames.emplace_back(glob_result.gl_pathv[i]);
    }

    // cleanup
    globfree(&glob_result);

    if(metadata_name.find(string("muscle")) != string::npos) {
        window = new MASS::PushWindow(meta_file_path,filenames[0], filenames[1]);
    }
    else if(metadata_name.find(string("torque")) != string::npos) {
        window = new MASS::PushWindow(meta_file_path, filenames[0]);
	}
    else if(metadata_name.find(string("Subject")) != string::npos) {
        window = new MASS::PushWindow(meta_file_path, filenames[0]);
    }

    if (window != nullptr) {
        window->initWindow(1024, 768, "gui");
        glutMainLoop();
    }

	return 0;
}
