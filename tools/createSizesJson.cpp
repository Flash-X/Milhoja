#include <iostream>
#include <fstream>
#include <tuple>
#include <nlohmann/json.hpp>

#include <Milhoja.h>
#include <Milhoja_real.h>
#include <Milhoja_IntVect.h>
#include <Milhoja_RealVect.h>
#include <Milhoja_FArray1D.h>
#include <Milhoja_FArray2D.h>
#include <Milhoja_FArray3D.h>
#include <Milhoja_FArray4D.h>

// The primary purpose of this file is to compile the byte size information
// of all data types used by the data packet, as well as determine the byte
// alignment value. This is necessary so that memory used by the
// data packet can be optimized, and any gpu code can avoid illegal access
// or alignment errors.
// 
// Because this script is called when the library is built, it assumes that
// the calling code knows what it is doing. Therefore, the file that is
// written by this script will be overwriten. The output of this script,
// sizes.json, should not be edited manually. 
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

// TODO: See todo further down.
//    using types = std::tuple<
//        int, unsigned int, real, IntVect,
//        RealVect, FArray1D, FArray2D,
//        FArray3D, FArray4D, std::size_t
//    >;
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
        //output << "{" << std::endl;

        // TODO: There should be a better way to print the name of a data
        //       type and its size by using a std::tuple object as well as
        //       typeid(T).name(). This way we can create a tuple object filled
        //       with types, making it easy to loop over as well as add new 
        //       types if necessary. However, for the sake of getting the 
        //       datapacket_tests into main, I'll ignore this for now.
        //output << "    \"byte_align\": " << BYTE_ALIGNMENT << "," << std::endl;
        //output << "    \"int\": " << sizeof(int) << "," << std::endl;
        //output << "    \"unsigned int\": " << sizeof(unsigned int) << "," << std::endl;
        //output << "    \"std::size_t\": " << sizeof(std::size_t) << "," << std::endl;
        //output << "    \"real\": " << sizeof(milhoja::Real) << "," << std::endl;
        //output << "    \"IntVect\": " << sizeof(milhoja::IntVect) << "," << std::endl;
        //output << "    \"RealVect\": " << sizeof(milhoja::RealVect) << "," << std::endl;
        //output << "    \"FArray1D\": " << sizeof(milhoja::FArray1D) << "," << std::endl;
        //output << "    \"FArray2D\": " << sizeof(milhoja::FArray2D) << "," << std::endl;
        //output << "    \"FArray3D\": " << sizeof(milhoja::FArray3D) << "," << std::endl;
        //output << "    \"FArray4D\": " << sizeof(milhoja::FArray4D) << std::endl;
        output << sizes.dump(INDENT);
        
        //output << "}" << std::endl;
        output.close();
    } catch (std::ifstream::failure e) {
        std::cerr << "Unable to open, write to, or close " << filename << std::endl;
        return 1;
    }

    return 0;
}

