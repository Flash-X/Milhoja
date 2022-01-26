#ifndef RUNTIME_PARAMETERS_H__
#define RUNTIME_PARAMETERS_H__

#include <string>

#include <Milhoja_real.h>

#include <nlohmann/json.hpp>

class RuntimeParameters {
public:
    ~RuntimeParameters(void);

    static void                initialize(const std::string& filename);
    static RuntimeParameters&  instance(void);
    void                       finalize(void);

    // I tried these as a single templated get routine, but calling code looked
    // ugly.  Since this is for tests, hopefully slightly decreased
    // maintainability here wins out over calling code that is easier to read.
    unsigned int       getUnsignedInt(const std::string& group,
                                      const std::string& parameter) const;
    std::size_t        getSizeT(const std::string& group,
                                const std::string& parameter) const;
    milhoja::Real      getReal(const std::string& group,
                               const std::string& parameter) const;

private:
    RuntimeParameters(void);

    static std::string     filename_;
    static bool            initialized_;
    static bool            finalized_;

    nlohmann::json         json_;
};

#endif

