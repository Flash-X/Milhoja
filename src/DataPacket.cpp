#include "DataPacket.h"
#include "Backend.h"

namespace orchestration {

/**
 * Constructs a DataPacket object containing no data items and with no
 * resources assigned to it.
 *
 * @todo The error checks below should be asserts.
 */
DataPacket::DataPacket(void)
      : mainStream_{},
        extraStreams_{},
        items_{},
        itemShapes_{},
        memoryPtr_src_{nullptr},
        memoryPtr_trg_{nullptr}
{
    clearMemorySizes();
    checkErrorNull(__func__);
}

/**
 * Destroys object.  Under normal operations, this object should have been
 * consumed prior to the execution of the destructor and therefore it should
 * not own resources.  However, resources are released upon abnormal
 * termination.
 */
DataPacket::~DataPacket(void) {
    nullify();
    checkErrorNull(__func__);
}

/**
 * Set the DataPacket into a null/empty state.  This includes releasing any
 * resources owned by the object and setting data members to ridiculous values.
 */
void  DataPacket::nullify(void) {
    // release streams
    mainStream_release();
    extraStreams_release();

    // release memory
    Backend::instance().releaseGpuMemory(&memoryPtr_src_, &memoryPtr_trg_);
    memoryPtr_src_ = nullptr;
    memoryPtr_trg_ = nullptr;

    // clear memory sizes
    clearMemorySizes();

    // clear item shapes
    clearItemShapes();

    // check items
    assert(0 == items_.size());

    // check self
    checkErrorNull(__func__);
}

/**
 * Determine if the packet is in the null/empty state.
 *
 * @return An empty string if yes; otherwise, an explanation of why it is not
 * null.
 */
std::string  DataPacket::isNull(void) const {
    // return error string if not null
    if (mainStream_.isValid()) {
        return "Main stream already acquired";
    } else if (0 < extraStreams_.size()) {
        return "Extra streams already acquired";
    } else if (memoryPtr_src_ != nullptr) {
        return "Source memory (pinned memory) already allocated";
    } else if (memoryPtr_trg_ != nullptr) {
        return "Target memory (device memory) already allocated";
    } else if (0 < items_.size()) {
        return "Items already added";
    } else if (0 < itemShapes_.size()) {
        return "Item shapes already added";
    } else if (hasValidMemorySizes()) {
        return "Memory sizes already set up";
    }

    // return empty string if null
    return "";
}

void  DataPacket::checkErrorNull(const std::string functionName) const {
    std::string   errMsg = isNull();

    if (!errMsg.empty()) {
        if (functionName.empty()) {
            throw std::logic_error("[DataPacket] " + errMsg);
        } else {
            throw std::logic_error("[DataPacket::" + functionName + "] " + errMsg);
        }
    }
}

void  DataPacket::pack_initialize(void) {
    // setup shapes of items
    setupItemShapes();

    // calculate memory sizes
    setupMemorySizes();
    assert(0 < getSize_src());
    assert(0 < getSize_trg());

    // allocate memory (pinned memory on host and memory on GPU)
    Backend::instance().requestGpuMemory(getSize_src(), &memoryPtr_src_,
                                         getSize_trg(), &memoryPtr_trg_);
    assert(nullptr != memoryPtr_src_);
    assert(nullptr != memoryPtr_trg_);
}

void  DataPacket::pack_finalize(void) {
    // request stream
    mainStream_request();
}

void  DataPacket::pack_finalize(unsigned int nExtraStreams) {
    // request streams
    mainStream_request();
    extraStreams_request(nExtraStreams);
}

void  DataPacket::unpack_initialize(void) {
    // release streams
    extraStreams_release();
    mainStream_release();
}

void  DataPacket::unpack_finalize(void) {
    // clear packet, since it is consumed upon unpacking
    nullify();
}

/****** STREAMS/QUEUES ******/

void  DataPacket::mainStream_request(void) {
    mainStream_ = Backend::instance().requestStream(true);
    if (!mainStream_.isValid()) {
        throw std::runtime_error("[DataPacket::mainStream_request] Unable to acquire main stream");
    }
}

void  DataPacket::mainStream_release(void) {
    Backend::instance().releaseStream(mainStream_);
    assert(!mainStream_.isValid());
}

void  DataPacket::extraStreams_request(unsigned int nExtraStreams) {
    for (unsigned int i=0; i<nExtraStreams; ++i) {
        extraStreams_.push_back( Backend::instance().requestStream(true) );
        if (!extraStreams_.back().isValid()) {
            throw std::runtime_error("[DataPacket::extraStreams_request] Unable to acquire extra stream");
        }
    }
}

void  DataPacket::extraStreams_releaseId(unsigned int id) {
    Stream& s = extraStreams_.at(id);

    if (s.isValid()) {
        Backend::instance().releaseStream(s);
        assert(!s.isValid());
    }
}

void  DataPacket::extraStreams_release(void) {
    for (unsigned int i=0; i<extraStreams_.size(); ++i) {
        extraStreams_releaseId(i);
    }
    extraStreams_.clear();
}

/****** ITEMS ******/

/**
 * Add the given Tile to the DataPacket.  As part of this, the packet assumes
 * ownership of the Tile and the calling code's pointer is nullified.
 *
 * @todo The checks should be asserts.  Figure out how to get that working.
 */
void   DataPacket::addTile(std::shared_ptr<Tile>&& item) {
    items_.push_front( std::move(item) );
    if ((item != nullptr) || (item.use_count() != 0)) {
        throw std::runtime_error("[DataPacket::addTile] Ownership of item not transferred");
    }
}

/**
 * Obtain one of the Tiles included in the packet.  As part of this, the Tile is
 * removed from the packet and ownership of the Tile passes to the calling code.
 *
 * @todo The checks should be asserts.  Figure out how to get that working.
 */
std::shared_ptr<Tile>  DataPacket::popTile(void) {
    if (items_.size() == 0) {
        throw std::invalid_argument("[DataPacket::popTile] No tiles to pop");
    }

    std::shared_ptr<Tile> item{ std::move(items_.front()) };
    if ((items_.front() != nullptr) || (items_.front().use_count() != 0)) {
        throw std::runtime_error("[DataPacket::popTile] Ownership of item not transferred");
    }

    items_.pop_front();
    if ((item == nullptr) || (item.use_count() == 0)) {
        throw std::runtime_error("[DataPacket::popTile] Bad item");
    }

    return item;
}

/****** MEMORY ******/

std::size_t DataPacket::getOffsetToPartItemVar(MemoryPartition part,
                                               unsigned int itemId,
                                               unsigned int varId) const {
    // check self
    if (items_.size() <= 0) {
        throw std::logic_error("[DataPacket::getOffsetToPartItemVar] No items in data packet");
    } else if (1 != itemShapes_.size() || items_.size() != itemShapes_.size()) {
        throw std::logic_error("[DataPacket::getOffsetToPartItemVar] Size mismatch of items and shapes");
    } else if (!hasValidMemorySizes()) {
        throw std::logic_error("[DataPacket::getOffsetToPartItemVar] Memory sizes not set up");
    }

    // check input
    if (MemoryPartition::_N <= part) {
        throw std::out_of_range("[DataPacket::getOffsetToPartItemVar] Memory partition is out of bounds");
    } else if (0 == memorySize_[part]) {
        throw std::invalid_argument("[DataPacket::getOffsetToPartItemVar] Memory partition is empty");
    } else if (items_.size() <= itemId) {
        throw std::out_of_range("[DataPacket::getOffsetToPartItemVar] Item is out of bounds");
    }

    // init offset
    std::size_t offset = 0;
    std::size_t siz;

    // find offset to partition
    for (unsigned int i=0; i<part; ++i) {
        offset += memorySize_[i];
    }

    // find offset to item (inside partition)
    if (part != MemoryPartition::SHARED) {
        if (isUniformItemShape()) {
            siz = itemShapes_.at(0).sizePartition(part);
            assert(siz != 0);
            offset += itemId * siz;
        } else {
            for (unsigned int n=0; n<itemId; ++n) {
                siz = itemShapes_.at(n).sizePartition(part);
                assert(siz != 0);
                offset += siz;
            }
        }
    }

    // find offset to variable (inside partition, inside item)
    const DataShape&  shape = getItemShape(itemId);
    if (shape.getN() <= varId) {
        throw std::invalid_argument("[DataPacket::getOffsetToPartItemVar] Variable ID is out of bounds");
    } else if (shape.atPart(varId) != part) {
        throw std::invalid_argument("[DataPacket::getOffsetToPartItemVar] Variable is not in memory partition");
    }
    for (unsigned int i=0; i<varId; ++i) {
        if (shape.atPart(i) != part) {
            // skip variables that are not in partition
            continue;
        } else {
            // add variable size to offset
            siz = shape.sizeVariable(i);
            assert(siz != 0);
            offset += siz;
        }
    }

    // return offset: partition > item > variable
    return offset;
}

void* DataPacket::getInPointerToPartItemVar_src(MemoryPartition part,
                                                unsigned int itemId,
                                                unsigned int varId) const {
    // check input
    if (MemoryPartition::INOUT < part) {
        throw std::invalid_argument("[DataPacket::getInPointerToPartItemVar_src] Memory partition is out of bounds");
    }

    // calculate offset
    std::size_t offset = getOffsetToPartItemVar(part, itemId, varId);

    // return pointer to memory
    return (void*)( ((char*)memoryPtr_trg_) + offset );
}

void* DataPacket::getOutPointerToPartItemVar_src(MemoryPartition part,
                                                 unsigned int itemId,
                                                 unsigned int varId) const {
    // check input
    if (part < MemoryPartition::INOUT) {
        throw std::invalid_argument("[DataPacket::getOutPointerToPartItemVar_src] Memory partition is out of bounds");
    }

    // calculate offset
    std::size_t offset = getOffsetToPartItemVar(part, itemId, varId);
    offset -= (memorySize_[MemoryPartition::SHARED] + memorySize_[MemoryPartition::IN]);

    // return pointer to memory
    return (void*)( ((char*)memoryPtr_trg_) + offset );
}

void* DataPacket::getPointerToPartItemVar_trg(MemoryPartition part,
                                              unsigned int itemId,
                                              unsigned int varId) const {
    // calculate offset
    std::size_t offset = getOffsetToPartItemVar(part, itemId, varId);

    // return pointer to memory
    return (void*)( ((char*)memoryPtr_trg_) + offset );
}

void  DataPacket::setupMemorySizes(void) {
    // check self
    if (items_.size() <= 0) {
        throw std::logic_error("[DataPacket::setupMemorySizes] No items in data packet");
    } else if (!isUniformItemShape() || items_.size() != itemShapes_.size()) {
        throw std::logic_error("[DataPacket::setupMemorySizes] Size mismatch of items and shapes");
    }

    // calculate memory sizes
    if (isUniformItemShape()) { /* if shapes are uniform across items */
        unsigned int      nItems = items_.size();
        const DataShape&  shape  = itemShapes_.at(0);

        memorySize_[MemoryPartition::SHARED]  =          shape.sizePartition(MemoryPartition::SHARED);
        memorySize_[MemoryPartition::IN]      = nItems * shape.sizePartition(MemoryPartition::IN);
        memorySize_[MemoryPartition::INOUT]   = nItems * shape.sizePartition(MemoryPartition::INOUT);
        memorySize_[MemoryPartition::OUT]     = nItems * shape.sizePartition(MemoryPartition::OUT);
        memorySize_[MemoryPartition::SCRATCH] = nItems * shape.sizePartition(MemoryPartition::SCRATCH);
    } else { /* otherwise each item has its own shape */
        memorySize_[MemoryPartition::SHARED]  = itemShapes_.at(0).sizePartition(MemoryPartition::SHARED);
        memorySize_[MemoryPartition::IN]      = 0;
        memorySize_[MemoryPartition::INOUT]   = 0;
        memorySize_[MemoryPartition::OUT]     = 0;
        memorySize_[MemoryPartition::SCRATCH] = 0;

        for (unsigned int n=0; n<itemShapes_.size(); ++n) {
            const DataShape&  shape = itemShapes_[n];

            if (memorySize_[MemoryPartition::SHARED] != shape.sizePartition(MemoryPartition::SHARED)) {
                throw std::logic_error("[DataPacket::setupMemorySizes] Memory size of share partition is not uniform");
            }
            memorySize_[MemoryPartition::IN]      += shape.sizePartition(MemoryPartition::IN);
            memorySize_[MemoryPartition::INOUT]   += shape.sizePartition(MemoryPartition::INOUT);
            memorySize_[MemoryPartition::OUT]     += shape.sizePartition(MemoryPartition::OUT);
            memorySize_[MemoryPartition::SCRATCH] += shape.sizePartition(MemoryPartition::SCRATCH);
        }
    }
}

void  DataPacket::clearMemorySizes(void) {
    for (unsigned int i=0; i<MemoryPartition::_N; ++i) {
        memorySize_[i] = 0;
    }
}

bool  DataPacket::hasValidMemorySizes(void) const {
    for (unsigned int i=0; i<MemoryPartition::_N; ++i) {
        if (0 < memorySize_[i]) {
            return true;
        }
    }
    return false;
}

std::size_t   DataPacket::getSize_src(void) const {
    if (getOutSize() < getInSize()) {
        return getInSize();
    } else {
        return getOutSize();
    }
}

std::size_t   DataPacket::getSize_trg(void) const {
    std::size_t s=0;

    for (unsigned int i=0; i<MemoryPartition::_N; ++i) {
        s += memorySize_[i];
    }
    return s;
}

}

