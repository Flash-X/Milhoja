#include <iostream>
#include "Milhoja.h"
#include "Milhoja_real.h"
#include "Milhoja_IntVect.h"
#include "Milhoja_RealVect.h"
#include "Milhoja_FArray1D.h"
#include "Milhoja_FArray2D.h"
#include "Milhoja_FArray3D.h"
#include "Milhoja_FArray4D.h"
#include <nlohmann/json.hpp>
#include <fstream>

void createSizesJson() {
    using namespace milhoja;
    using json = nlohmann::json;

    // open file
    std::ofstream output;
    output.open("sizes.json");          

    // todo: memory alignment?

    // create new json object
    json sizes = {
        {"byte_align", 16},
        {"int", sizeof(int)},
        {"unsigned int", sizeof(unsigned int)},
        {"std::size_t", sizeof(std::size_t)},
        {"real", sizeof(Real)},
        {"IntVect", sizeof(IntVect)},
        {"RealVect", sizeof(RealVect)},
        {"FArray1D", sizeof(FArray1D)},
        {"FArray2D", sizeof(FArray2D)},
        {"FArray3D", sizeof(FArray3D)},
        {"FArray4D", sizeof(FArray4D)}
    };

    output << sizes.dump(4);

    output.close();
}

int main() {
    createSizesJson();
    return 0;
}
