#ifndef DATA_SHAPE_H__
#define DATA_SHAPE_H__

#include <cstddef>
#include <stdexcept>

namespace orchestration {

class DataShape {
protected:
    unsigned int    n_;    //!< Number of variables
    unsigned int    d_;    //!< Max dimensions per variable shape
    std::size_t**   ls_;   //!< List of shapes, size=(n x dim)
    std::size_t*    pad_;  //!< List of paddings, size=(n)
    unsigned int*   part_; //!< List of partitions, size=(n)

public:
    /**
     * Constructs a DataShape object.
     */
    DataShape(unsigned int, unsigned int);

    /**
     * Destroys object.
     */
    ~DataShape(void);

    // override defaults
    DataShape(DataShape&)                  = delete;
    DataShape(const DataShape&)            = delete;
    DataShape(DataShape&&)                 = delete;
    DataShape& operator=(DataShape&)       = delete;
    DataShape& operator=(const DataShape&) = delete;
    DataShape& operator=(DataShape&&)      = delete;

    void          set(const std::size_t*);
    void          setPadding(const std::size_t*);
    void          setPartition(const unsigned int*);

    unsigned int  getN(void) const { return n_; }
    unsigned int  getD(void) const { return d_; }

    std::size_t   at(unsigned int varIdx, unsigned int dIdx) const {
        if (n_ <= varIdx) throw std::out_of_range("[DataShape::at] Variable index out of range");
        if (d_ <= dIdx)   throw std::out_of_range("[DataShape::at] Dimension index out of range");
        return ls_[varIdx][dIdx];
    }

    unsigned int  atPart(unsigned int varIdx) const {
        if (n_ <= varIdx) throw std::out_of_range("[DataShape::atPart] Variable index out of range");
        return part_[varIdx];
    }

    std::size_t   size(void) const;
    std::size_t   sizePartition(unsigned int) const;
    std::size_t   sizeVariable(unsigned int) const;

private:
    unsigned int nEntries(void) const { return n_*d_; }
};

}

#endif
