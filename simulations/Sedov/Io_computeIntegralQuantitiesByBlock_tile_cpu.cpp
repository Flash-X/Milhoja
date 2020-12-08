#include "Io.h"
#include "Driver.h"

#include "Tile.h"
#include "Grid.h"

/**
 * The action routine wrapper of computeIntegralQuantitiesByBlock created so that
 * the function can be executed by the orchestration runtime using CPU resources
 * applied to Tiles.
 *
 * \param tID - the ID of the runtime thread that called this function
 * \param dataItem - the data item to which the action routine should be
 *                   applied.  It is an error if this pointer does not point to
 *                   a Tile object.
 */
void ActionRoutines::Io_computeIntegralQuantitiesByBlock_tile_cpu(const int tId,
                                                                  orchestration::DataItem* dataItem) {
    using namespace orchestration;

    Io&   io   = Io::instance();
    Grid& grid = Grid::instance();

    Tile*  tileDesc = dynamic_cast<Tile*>(dataItem);

    unsigned int        level = tileDesc->level();
    const IntVect       lo    = tileDesc->lo();
    const IntVect       hi    = tileDesc->hi();
    const FArray4D      U     = tileDesc->data();

    // Integrate only over the whole interior
    Real   volumes_buffer[  (hi.I() - lo.I() + 1)
                          * (hi.J() - lo.J() + 1)
                          * (hi.K() - lo.K() + 1)];
    grid.fillCellVolumes(level, lo, hi, volumes_buffer); 
    const FArray3D   volumes{volumes_buffer, lo, hi};

    io.computeIntegralQuantitiesByBlock(Driver::simTime, tId, lo, hi, volumes, U);
}

