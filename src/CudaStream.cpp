#include "CudaStream.h"

// TODO: I need to include this so that NULL_STREAM_ID can be used in files other than
// CudaStream.h.  Else, the linker complains that it can't find it.  Why is this
// necessary?  My current guess is that by defining it here, its address will
// appear in an object file, which can be searched by the linker.  When the
// linker finds the reference, does it just include the location of the value or
// does it copy the value over to the reference to NULL_STREAM_ID?
constexpr unsigned int CudaStream::NULL_STREAM_ID;

