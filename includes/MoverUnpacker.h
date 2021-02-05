/**
 * \file   MoverUnpacker.h
 * 
 * \brief  A class that implements a single thread team helper that is capable of
 * transferring data from device to host and, if necessary, splitting up composite
 * data items into small items and enqueueing these with a data receiver.
 *
 * This thread helper class is designed as an extended finite state
 * machine (eFSM).  While its design is similar to that of the ThreadTeam, it is
 * simpler as it has no internal threads to manage.
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
 * As objects instantiated from this class will be used to build thread team
 * configurations, such objects must manage correctly the shared_ptr to
 * DataPacket items whose ownership flows through pipelines along with the
 * share_ptrs themselves.  It is assumed that neither this helper nor the
 * downstream elements in its pipeline will use the given DataPacket once the
 * helper has finished with it.  Therefore, the enqueue member function will
 * assume ownership of the given shared_ptr to a DataPacket as expected.  The
 * ownership is transferred to the DataPacket to hold during the transfer.
 * Ownership is transferred back to the MoverUnpacker when the DataPacket calls
 * handleTransferFinished.  Finally, this static function will free the
 * shared_ptr once all necessary work has been applied to the contents of the
 * DataPacket.
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

    void startCycle(void);
    void increaseThreadCount(const unsigned int nThreads) override;
    void enqueue(std::shared_ptr<DataItem>&& dataItem) override;
    void closeQueue(void) override;

    void wait(void);

    RuntimeElement*  dataReceiver(void) const  { return dataReceiver_; }

private:
    enum class State {Idle, Open, Closed};

    struct CallbackData {
        std::shared_ptr<DataItem>*   dataItem = nullptr;
        MoverUnpacker*               unpacker = nullptr;
    };

    static void handleTransferFinished(void* userData);
    void        handleTransferFinished_Stateful(void);

    State          state_;          //!< Qualitative state of eFSM
    unsigned int   nInTransit_;     //!< N DataPackets in transit.  Quantitative state of eFSM.

    pthread_mutex_t   mutex_;               //!< Use to access all private data members
    pthread_cond_t    unblockWaitThreads_;  //!< To be emitted when last transfer handled
};

}

#endif

