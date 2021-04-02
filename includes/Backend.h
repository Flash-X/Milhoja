#ifndef BACKEND_H__
#define BACKEND_H__

#include <cstddef>

namespace orchestration {

class Backend {
public:
    virtual ~Backend(void)     { instantiated_ = false; };

    Backend(Backend&)                  = delete;
    Backend(const Backend&)            = delete;
    Backend(Backend&&)                 = delete;
    Backend& operator=(Backend&)       = delete;
    Backend& operator=(const Backend&) = delete;
    Backend& operator=(Backend&&)      = delete;

    static void     instantiate(const unsigned int nStreams,
                                const std::size_t  nBytesInMemoryPools);
    static Backend& instance(void);

protected:
    Backend(void)     {};

private:
    static bool           instantiated_;
    static unsigned int   nStreams_;
    static std::size_t    nBytesInMemoryPools_;
};

}

#endif

