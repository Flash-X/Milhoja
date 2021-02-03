/**
 * \file   MoverUnpacker.h
 * 
 * \brief  A class that implements a single thread team helper that is capable of
 * transferring data from device to host and, if necessary, splitting up composite
 * data items into small items and enqueueing these with a data receiver.
 *
 * The threads of the data publishers to objects of this class that call enqueue
 * are used to initiate asynchronous transport of the given data item back to
 * the host.  Once this starts, the calling threads are released and the objects
 * will make certain, using other host threads, that unpacking and subsequent
 * enqueueing of data items with data receivers occur.  All thread team
 * configurations that have this helper as the terminal element of a pipeline
 * must call the helper's wait member function.  If they do not, then the
 * runtime might conclude that its action bundle has been applied to all given
 * data items even though some data items might still be transferring back to
 * the host.
 *
 * The present implementation assumes that once closeQueue is called, no
 * attempts will be made to enqueue data until the next cycle starts.
 *
 * \todo Add in a wait routine so that TT configurations that don't have a data
 * receiver attached to the MoverUnpacker can know when the execution cycle has
 * really finished as opposed to when the transfer of the final data packet has
 * been initiated.
 */

#ifndef MOVER_UNPACKER_H__
#define MOVER_UNPACKER_H__

#include <memory>
#include <pthread.h>

#include "RuntimeElement.h"

namespace orchestration {

class MoverUnpacker : public RuntimeElement {
public:
    MoverUnpacker(void);
    ~MoverUnpacker(void);

    MoverUnpacker(MoverUnpacker&)                  = delete;
    MoverUnpacker(const MoverUnpacker&)            = delete;
    MoverUnpacker(MoverUnpacker&&)                 = delete;
    MoverUnpacker& operator=(MoverUnpacker&)       = delete;
    MoverUnpacker& operator=(const MoverUnpacker&) = delete;
    MoverUnpacker& operator=(MoverUnpacker&&)      = delete;

    void increaseThreadCount(const unsigned int nThreads) override;

    void enqueue(std::shared_ptr<DataItem>&& dataItem) override;
    void closeQueue(void) override;

    RuntimeElement*  dataReceiver(void) const  { return dataReceiver_; }
    void             notifyCallbackFinished(void);

private:
    struct CallbackData {
        std::shared_ptr<DataItem>*   dataItem = nullptr;
        MoverUnpacker*               unpacker = nullptr;
    };

    static void finalizeAfterTransfer(void* userData);

    unsigned int   nInCallback_;
    bool           wasCloseQueueCalled_;

    pthread_mutex_t   mutex_;  //!< Use to access all private data members
};

}

#endif

