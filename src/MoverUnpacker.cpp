#include <iostream>

#include "MoverUnpacker.h"

#include "DataPacket.h"

namespace orchestration {

MoverUnpacker::MoverUnpacker(void)
    : nInCallback_{0},
      wasCloseQueueCalled_{false} {
    pthread_mutex_init(&mutex_, NULL);
}

MoverUnpacker::~MoverUnpacker(void) {
    pthread_mutex_lock(&mutex_);

    if (nInCallback_ != 0) {
        std::cerr << "[MoverUnpacker::~MoverUnpacker] "
                  << "Data packets still in transit\n";
    }

    pthread_mutex_unlock(&mutex_);
    pthread_mutex_destroy(&mutex_);
}

void MoverUnpacker::increaseThreadCount(const unsigned int nThreads) {
    throw std::logic_error("[MoverUnpacker::increaseThreadCount] "
                           "MoverUnpackers do no have threads to awaken");
}

/**
 * Use the calling thread to initiate the asynchronous transfer of the given
 * data packet from device to host. 
 *
 * This member function will assume ownership of the given shared_ptr.  It is
 * assumed that neither this helper nor the downstream elements in its pipeline
 * will use the given DataPacket once the helper has finished with it.
 * Therefore, the shared_ptr will be automatically freed once the transfer has
 * finished, data is unpacked, and enqueueing done.
 *
 * \param  dataItem   The data packet that should be moved back to the host and
 *                    whose constituent data items should be enqueued with this
 *                    helper's data receiver if it has one.
 *
 * \todo This assumes that the calling thread need not be involved in the
 * unpacking process, which might not work for all runtime backends.  If it
 * does, can we get away with using preprocessors to accommodate the fact that
 * not all runtime backends will have identical interfaces to the
 * initiateDeviceToHostTransfer method?
 */
void MoverUnpacker::enqueue(std::shared_ptr<DataItem>&& dataItem) {
    pthread_mutex_lock(&mutex_);
    if (wasCloseQueueCalled_) {
        throw std::logic_error("[MoverUnpacker::enqueue] "
                               "The queue has been closed already");
    }
    ++nInCallback_;
    pthread_mutex_unlock(&mutex_);

    // The action taken here only makes sense if the given data item is a packet.
    // Therefore, this ugly upcasting is reasonable.
    DataPacket*    packet = dynamic_cast<DataPacket*>(dataItem.get());

    // Ownership of this memory resource will flow through the component
    // handling the transfer and back to the callback function, which must
    // release the resource.
    CallbackData*   userData = new CallbackData{};
    userData->unpacker = this;

    // Take ownership in such a way that we can transfer the ownership from
    // MoverUnpacker -> transfer component -> MoverUnpacker.
    // 
    // We need to continue wrapping the DataItem in a shared_ptr so that
    // multiple copies of this DataPacket can be flowing down different
    // pipelines simultaneously and still have correct release of resources when
    // the final shared_ptr exits its pipeline.
    // 
    // We need a dynamically-allocated shared_ptr, which is ugly in that it runs
    // counter to the intent of using smart pointers
    //    (i.e. don't use raw C++ pointers and
    //          limit use of new calls outside of constructors),
    // so that the shared_ptr persists as it moves to CUDA and back.
    userData->dataItem = new std::shared_ptr<DataItem>{ std::move(dataItem) };
    if ((dataItem.use_count() != 0) || (dataItem != nullptr)) {
        throw std::runtime_error("[MoverUnpacker::enqueue] "
                                 "shared_ptr not nullified as expected");
    }

    // This registers the callback
    packet->initiateDeviceToHostTransfer(finalizeAfterTransfer, userData);
    userData = nullptr;
}

/**
 * It is intended that this routine only be called by the callback function
 * finalizeAfterTransfer once it has finished its work.  This is necessary so
 * that the static callback routine can allow the object that registered the
 * callback with the packet can keep track of how many callback routines have
 * yet to be run and to close the queue of the object's data receiver once all
 * callbacks have been run.
 */
void  MoverUnpacker::notifyCallbackFinished(void) {
    pthread_mutex_lock(&mutex_);

    if (nInCallback_ <= 0) {
        throw std::logic_error("[MoverUnpacker::notifyCallbackFinished] "
                               "Callback count underflow");
    }
    --nInCallback_;

    if (wasCloseQueueCalled_ && (nInCallback_ == 0)) {
        // Reset for the next runtime execution cycle
        wasCloseQueueCalled_ = false;

        if (dataReceiver_) {
            dataReceiver_->closeQueue();
        }
    }

    pthread_mutex_unlock(&mutex_);
}

/**
 * It is intended that this static function only be used as a callback
 * registered with a data packet when its transfer to host is initiated.  This
 * implies that this routine will be called by a host thread outside of the
 * thread team configuration when the packet's transfer is complete.  This
 * callback is, therefore, responsible for unpacking the packet's data and
 * enqueueing constituent data items with the data receiver of the helper that
 * handled the packet.  In addition, this function must close the queue of the
 * data receiver if it is working on the final data packet to be transferred in
 * the current runtime execution cycle.
 *
 * This routine not only assumes ownership of the given userData structure, and
 * therefore has the responsibility to release its resources, but also the
 * ownership of the contained DataPacket.  This routine will destroy this latter
 * shared_ptr so that the resources of its contents can also be released if
 * the given shared_ptr is the last copy of its kind.
 *
 * \param   userData - a pointer to an object of type CallbackData that contains
 *                     all data necessary for this static callback to carry out
 *                     its work.  This includes a pointer to the object that
 *                     registered the callback as well as the data item that was
 *                     transferred.
 */
void MoverUnpacker::finalizeAfterTransfer(void* userData) {
    CallbackData*   data = static_cast<CallbackData*>(userData);
    if (!data) {
        throw std::logic_error("[MoverUnpacker::finalizeAfterTransfer] Given null pointer");
    }
    MoverUnpacker*   unpacker = data->unpacker;
    if (!unpacker) {
        throw std::logic_error("[MoverUnpacker::finalizeAfterTransfer] Given null unpacker");
    }
    RuntimeElement*  dataReceiver = unpacker->dataReceiver();
 
    // Unpacking only makes sense if the given data item is a packet.
    // Therefore, this ugly upcasting is reasonable.
    DataPacket*      packet = dynamic_cast<DataPacket*>(data->dataItem->get());
    if (!packet) {
        throw std::logic_error("[MoverUnpacker::finalizeAfterTransfer] Given null packet");
    }
    packet->unpack();

    // Transfer the ownership of the data items in the packet to the next team
    if (dataReceiver) {
        while (packet->nTiles() > 0) {
            dataReceiver->enqueue( std::move(packet->popTile()) );
        }
        dataReceiver = nullptr;
    }
    packet = nullptr;

    // Allow for possibility that resources can be freed as soon as possible.
    // This shared_ptr to the DataPacket is no longer needed here or downstream
    // in the pipeline.
    delete data->dataItem;   data->dataItem = nullptr;
    data->unpacker = nullptr;
    delete data;   data = nullptr;

    unpacker->notifyCallbackFinished();
    unpacker = nullptr;
}

/**
 * While this helper is presently implemented without a queue, we can
 * conceptually understand that the helper's "queue" is closed when this member
 * function is called in the sense that no data publishers should subsequently
 * call the helper's enqueue member function.  However, the queue of any data
 * receiver will not be closed until the last data packet to be transferred in
 * the present runtime execution cycle by this helper has been handled by a
 * callback function.
 *
 * \todo If calling code calls closeQueue when there are no callbacks out, then
 *       the calling code could subsequently call enqueue() without triggering an
 *       error.  I believe that this is an edge case associated with a
 *       programmer's logical error, but it should be studied.
 */
void MoverUnpacker::closeQueue(void) {
    pthread_mutex_lock(&mutex_);

    if (wasCloseQueueCalled_) {
        throw std::logic_error("[MoverUnpacker::closeQueue] "
                               "closeQueue already called");
    }

    if (nInCallback_ > 0) {
        // Setting this true means that this object's "queue" is effectively
        // closed.  However, its data receiver's queue is not closed until the
        // last callback has finished running.
        wasCloseQueueCalled_ = true;
    } else if (dataReceiver_) {
        // Both this helper and its receiver have closed queues.
        dataReceiver_->closeQueue();
    }

    pthread_mutex_unlock(&mutex_);
}

}

