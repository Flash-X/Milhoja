#include <iostream>
#include <fstream>

#include <nlohmann/json.hpp>

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray2D.h>
#include <Milhoja_FArray3D.h>
#include <Milhoja_FArray4D.h>

/**
 * TODO: Write explanation.  Include that this assumes that calling code knows
 * what it is doing and that pre-existing files will be overwritten.
 */
int main(int argc, char* argv[]) {
    //----- HARDCODED VALUES
    constexpr int            INDENT = 4;
    // TODO: This shouldn't be hardcoded but determined in accord with the
    // hardware on the target platform.  We choose a large, conservative value
    // for now in the hope that we can get reasonable data packet performance on
    // all platforms.
    constexpr unsigned int   BYTE_ALIGNMENT = 16;

    //----- COMMAND LINE ARGUMENTS
    if (argc != 2) {
        std::cerr << "Must provide the sizes filename and nothing else" << std::endl;
        return 1;
    }
    std::string   filename = argv[1];

    //----- CREATE NEW JSON OBJECT
    nlohmann::json sizes = {
        {"byte_align", BYTE_ALIGNMENT},
        {"int", sizeof(int)},
        {"unsigned int", sizeof(unsigned int)},
        {"std::size_t", sizeof(std::size_t)},
        {"real", sizeof(milhoja::Real)},
        {"IntVect", sizeof(milhoja::IntVect)},
        {"RealVect", sizeof(milhoja::RealVect)},
        {"FArray1D", sizeof(milhoja::FArray1D)},
        {"FArray2D", sizeof(milhoja::FArray2D)},
        {"FArray3D", sizeof(milhoja::FArray3D)},
        {"FArray4D", sizeof(milhoja::FArray4D)}
    };

    //----- SAVE TO FILE
    std::ofstream output;
    output.exceptions(std::ofstream::failbit | std::ofstream::badbit);
    try {
        output.open(filename, std::ofstream::out | std::ofstream::trunc);
        output << sizes.dump(INDENT);
        output.close();
    } catch (std::ifstream::failure e) {
        std::cerr << "Unable to open, write to, or close " << filename << std::endl;
        return 1;
    }

    return 0;
}

