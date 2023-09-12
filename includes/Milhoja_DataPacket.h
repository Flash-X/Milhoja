/**
 * \file Milhoja_DataPacket.h
 *
 * It is important that the runtime be decoupled fully from the content of the
 * computations that it helps execute.  In particular, this means that it should
 * function properly without having any knowledge of what computations it is
 * helping execute nor what data/information such computations require to
 * execute correctly.  This implies that all actions that the runtime calls must
 * have the same simple function interface.  As each action routine could
 * potentially require a different set of input arguments, this common interface
 * must hide away this variability and does so by insisting that all inputs to
 * the action routine being called must be encapsulated as a single object that,
 * for the purpose of polymorphism, is inherited from DataItem.  Please refer to
 * the documentation for ACTION_ROUTINE for more information regarding this
 * interface.
 *
 * As part of this, a DataItem can be classified as either a Tile or a
 * DataPacket.  The Grid unit defines the Tile interface and concrete Grid
 * implementations include concrete implementations of the Tile interface.  The
 * runtime's distributors use the Grid's iterator to access Tile objects and can
 * subsequently push each of these to the appropriate pipelines in its associated
 * configuration.  The threads in ThreadTeams couple these objects with actions
 * to form tasks.  In this sense, Tile is the fundamental DataItem for the
 * runtime.  Note that when a thread forms and executes a task, it passes the
 * task's DataItem to the task's function and this function must know that the
 * object is a Tile so that it can correctly pull the associated action's input
 * from the object using the runtime's Tile interface.
 *
 * The DataPacket class defined here exists to package up one or more tiles so
 * that these can be transferred together with the intent of decreasing the
 * impact on performance associated with such data movements.  As the
 * distributor accesses Tile objects, it adds these to a DataPacket object.
 * Once a DataPacket is full, the distributor asks the DataPacket to prepare
 * itself for transfer (i.e. pack itself), initiates data movements, and pushes
 * it to the appropriate pipelines so that threads will decompose it into Tiles
 * that are then used to construct tasks.  This implies that the distributor
 * must have the ability to create DataPacket objects as needed.
 *
 * To satisfy the aforementioned function interface requirement and to minimize
 * data movements, non-tile-specific data needed by an action routine (e.g. dt)
 * can also be included in each DataPacket object.  This, however, implies that
 * each action that works with DataPackets could require different content and that
 * only the application that defines a particular action knows what this content
 * should be.  Hence, concrete DataPackets, unlike concrete Tiles, must be
 * specified outside the runtime library.  Sensibly, these are derived from
 * DataPacket and must implement the packing of each DataPacket object.  The
 * associated function that receives DataPacket objects must be written in
 * tandem to unpack the contents and therefore gain access to the input
 * arguments it needs for executing its action.
 *
 * This class was designed using the Prototype design pattern (Citation XXX) so
 * that given a prototype DataPacket object of an unknown concrete type, the
 * runtime can clone this and therefore create new DataPacket objects of the
 * appropriate concrete type.  Note that the runtime is therefore decoupled from
 * the fine details of each DataPacket --- it is effectively a conduit.  Given a
 * prototype, the runtime blindly creates DataPacket objects that know how to
 * pack themselves and passes the objects to a task function that knows how
 * to unpack and use them.
 *
 * Designers that would like to derive from this class should refer to the
 * documentation of the protected data members to understand which of these must
 * be set by a derived class's pack() member function.
 *
 * The interface allows for some optimization of data transfers by allowing
 * DataPackets to be structured such that there is a contiguous copy-in block, a
 * continguous copy-in/out block, and a contiguous copy-out block.  Ideally, the
 * copy-in and copy-in/out blocks will be adjacent so that only these two blocks
 * can be sent to a device and done so in a single transfer.  Similarly, the
 * copy in/out and copy-out blocks should be adjacent so that only these
 * sections are transferred back to host and in a single transfer.
 *
 * As part of packing, a DataPacket object should be assigned a single Stream
 * object with the expectation that all host-to-GPU transfers of the packet
 * occur on the Stream and that GPU computations also be launched on the same
 * Stream to ensure that computation only begins once the data is in GPU memory.
 * Additionally, the packet can acquire at packing extra Streams.  These should
 * not be used for data transfers, but only for allowing concurrent execution of
 * kernels on the GPU.  While the main Stream will be released automatically by
 * the runtime once the packet has arrived back at the host, it is the task
 * function's responsiblity to release the extra Streams once they are no longer
 * needed for concurrent computations.
 *
 * A DataPacket is effectively consumed once it arrives back at the host memory
 * and is automatically unpacked (Refer to the documentation for unpack() to
 * understand how host-side unpacking is different from the device-side
 * unpacking carried out by the packet's task function).  For instance, all
 * resources assigned to the packet are released and no code should try to
 * access data previously managed by the packet.  It is not intended that a
 * DataPacket object be reused once it is consumed.  Rather this resource should
 * be released as well.
 *
 * While the DataPacket objects are created dynamically by distributors, the
 * ownership of these objects flows with them as they move through a thread team
 * configuration.  As part of the DataItem resource management scheme designed
 * in accord with this notion of flowing ownership, it is assumed that at any
 * point in time at most one thread can have access to a given DataPacket
 * object, which implies exclusive ownership.  In particular, this class is not
 * thread safe.  Refer to the system-level documentation for a discussion of how
 * the overall design should ensure that this assumption is not violated.
 *
 * DataPackets have two CC buffers so that computations can use data stored in
 * one and write results to the other.  This is important for computations whose
 * results cannot be written in place.  Ideally, only the location that contains
 * the original data will be transferred to the device and only the location
 * storing the result will be transferred back to host.  The DataPacket allows
 * for setting and accessing the current location and this must always be
 * maintained up-to-date.  For instance, the unpack() member function needs knows
 * where to find the data.
 *
 * Each packet is configured to only manage a single contiguous group of
 * variables.  This includes the possiblity of mananging all variables.  In
 * particular, the unpack() function will only copy to host memory results in
 * the packet associated with this group of variables.  If this weren't the case
 * and two or more data-indepedent actions were executed concurrently, the copy
 * back could overwrite results computed by another action.
 *
 * @todo This DataPacket is designed assuming that the packet starts in the
 * host, is transferred to the GPU, and then transferred back.  What about the
 * case of a packet that starts at a the host, is transferred to the GPU, then
 * to the FPGA, and then back to the host.  What about packets that needn't
 * start or end on the host?  Note that such a data packet would not be
 * associated with a single thread team or a single action.
 * @todo Make sure that ACTION_ROUTINE above links to the actual documentation
 *       in doxygen.
 * @todo Add in citation to Gang of Four book.
 */

#ifndef MILHOJA_DATA_PACKET_H__
#define MILHOJA_DATA_PACKET_H__

#include <stdexcept>

#include "Milhoja.h"
#include "Milhoja_IntVect.h"
#include "Milhoja_RealVect.h"
#include "Milhoja_FArray4D.h"
#include "Milhoja_Tile.h"
#include "Milhoja_DataItem.h"
#include "Milhoja_Stream.h"

#if defined(MILHOJA_CUDA_OFFLOADING) || defined(MILHOJA_CUDA_RUNTIME_BACKEND)
#include <cuda_runtime.h>
#endif

namespace milhoja {

/**
 * Store and communicate the location of the current values of the cell-centered
 * data managed by a DataPacket.
 */
enum class PacketDataLocation {NOT_ASSIGNED, CC1, CC2};

/**
 * @class DataPacket Milhoja_DataPacket.h
 * @brief Define and effectively export to applications the minimal DataPacket
 * interface that their concrete DataPackets must implement.
 */
class DataPacket : public DataItem {
public:
    /**
     * Obtain a pointer to a new, empty DataPacket of the same concrete type as
     * the calling object.  The main workhorse of the Prototype design pattern.
     *
     * @return The pointer.  For the main use cases in the runtime, this should
     * be cast to a shared_ptr.  We return a unique_ptr based on the discussion
     * in Item 19 (Pp 113) of Effective Modern C++.
     *
     * @todo Add in citation.
     */
    virtual std::unique_ptr<DataPacket>  clone(void) const = 0;

    virtual ~DataPacket(void);

    DataPacket(DataPacket&)                  = delete;
    DataPacket(const DataPacket&)            = delete;
    DataPacket(DataPacket&& packet)          = delete;
    DataPacket& operator=(DataPacket&)       = delete;
    DataPacket& operator=(const DataPacket&) = delete;
    DataPacket& operator=(DataPacket&&)      = delete;

    /**
     * Obtain the number of Tiles presently included in the packet.
     */
    std::size_t            nTiles(void) const        { return tiles_.size(); }
    void                   addTile(std::shared_ptr<Tile>&& tileDesc);
    std::shared_ptr<Tile>  popTile(void);

    /**
     * The runtime calls this member function automatically once all tiles to be
     * included in the DataPacket have been added using the addTile() function.
     * Derived classes implement this functionality and in doing so specify the
     * structure of the data packet and pack the DataPacket's content in host
     * memory in preparation for transfer to device memory.  Concrete
     * implementations determine in what type of memory packing occurs and to
     * what type of memory the data is sent.
     */
    virtual void           pack(void) = 0;
    virtual void           unpack(void) = 0;

    /**
     * Obtain a pointer to the start of the contiguous block of memory on the
     * host side that will be transferred to GPU memory.
     */
    void*                  copyToGpuStart_host(void)           { return (void*)copyInStart_p_; };

    /**
     * Obtain a pointer to the start of the contiguous block of memory in GPU
     * device memory that will receive the packet's data from the host.
     */
    void*                  copyToGpuStart_gpu(void)            { return (void*)copyInStart_d_; };

    /**
     * Obtain the number of bytes to be sent from the host to the GPU.
     */
    std::size_t            copyToGpuSizeInBytes(void) const    { return nCopyToGpuBytes_; };

    /**
     * Obtain a pointer to the start of the contiguous block of memory in host
     * memory that will receive the packet's data upon transfer back to the
     * host.
     */
    void*                  returnToHostStart_host(void)        { return (void*)copyInOutStart_p_; };

    /**
     * Obtain a pointer to the start of the contiguous block of memory on the
     * GPU side that will be transferred back to host memory.
     */
    void*                  returnToHostStart_gpu(void)         { return (void*)copyInOutStart_d_; };

    /**
     * Obtain the number of bytes to be sent from the GPU back to the host.
     */
    std::size_t            returnToHostSizeInBytes(void) const { return nReturnToHostBytes_; };

#ifdef MILHOJA_OPENACC_OFFLOADING
    /**
     * Obtain the main OpenACC asynchronous queue assigned to the packet, on
     * which communications are scheduled and computation can also be scheduled.
     * This can be called after pack() is called and before unpack() is called.
     */
    int                    asynchronousQueue(void) { return stream_.accAsyncQueue; }

    /**
     * Obtain the indicated extra OpenACC asynchronous queue assigned to the
     * packet for concurrent kernel execution.  This can be called after pack()
     * is called and before unpack() is called.  These queues must be explicitly
     * released by the task function using releaseExtraQueue() and before the
     * task function terminates.
     *
     * While calling code can call this as many times as desired for any given
     * id, it is a logical error to call this function to obtain a function that has
     * already been released.
     *
     * It is the task function's responsibility to use this queue correctly.
     * This includes synchronizing computation on this queue with the arrival of
     * data on the packet's main queue.
     *
     * @param id - the index of the queue to obtain.  If the packet has acquired
     * N total queues, valid values are 2 to N inclusive.
     */
    virtual int            extraAsynchronousQueue(const unsigned int id)
        { throw std::logic_error("[DataPacket::extraAsynchronousQueue] no extra queues"); }

    /**
     * Release the indicated extra OpenACC asynchronous queue.  This must be 
     * called after calling pack() and before calling unpack().  It is a logical
     * error to call this more than once for any given id or on the main queue.
     * 
     * @param id - the index of the queue to release.
     */
    virtual void           releaseExtraQueue(const unsigned int id)
        { throw std::logic_error("[DataPacket::releaseExtraQueue] no extra queue"); }
#endif
#if defined(MILHOJA_CUDA_OFFLOADING) || defined(MILHOJA_CUDA_RUNTIME_BACKEND)
    /**
     * Obtain the main CUDA stream assigned to the DataPacket for transferring the
     * packet to and from the host.  This can be called after pack() is called
     * and before unpack() is called.  Calling code can call this as many times
     * as desired.
     */
    cudaStream_t           stream(void)            { return stream_.cudaStream; };
#endif

protected:
    DataPacket(void);

    void         nullify(void);
    std::string  isNull(void) const;
 
    void*                                  packet_p_;           /*!< The starting location in pinned memory
                                                                 *   of the memory allocated to the packet */ 
    void*                                  packet_d_;           /*!< The starting location in GPU memory
                                                                 *   of the memory allocated to the packet */ 
    char*                                  copyInStart_p_;      /*!< The starting location in pinned memory
                                                                 *   of the copy-in block*/ 
    char*                                  copyInStart_d_;      /*!< The starting location in GPU memory 
                                                                 *   of the copy-in block*/ 
    char*                                  copyInOutStart_p_;   /*!< The starting location in pinned memory 
                                                                 *   of the copy-in/out block*/ 
    char*                                  copyInOutStart_d_;   /*!< The starting location in GPU memory
                                                                 *   of the copy-in/out block*/ 
    std::deque<std::shared_ptr<Tile>>      tiles_;              /*!< Tiles included in packet.  Derived
                                                                 * classes need only read from this. */
    Stream                                 stream_;             //!< Main stream for communication
    std::size_t                            nCopyToGpuBytes_;    //!< Number of bytes in copy in and copy in/out
    std::size_t                            nReturnToHostBytes_; //!< Number of bytes in copy in/out and copy out
};

}

#endif

