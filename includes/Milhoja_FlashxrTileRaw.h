#ifndef MILHOJA_FLASHXR_TILERAW_H__
#define MILHOJA_FLASHXR_TILERAW_H__

#include "Milhoja.h"
#include "Milhoja_real.h"

#ifdef RUNTIME_USES_TILEITER
#error "This file should only be compiled if the Runtime class does not invoke a tile iterator"
#endif

#define LIST_MDIM(x,y,z) x,y,z

//namespace milhoja {


struct FlashxrTileRawPtrs {
  milhoja::Real*           unkBlkPtr;
  milhoja::Real*           fluxxBlkPtr, fluxyBlkPtr, fluxzBlkPtr;
};

struct FlashxTileRawInts {
  int nCcComp;
  int nFluxComp;
  int LIST_MDIM(loGCX,loGCY,loGCZ);
  int LIST_MDIM(hiGCX,hiGCY,hiGCZ);
  int LIST_MDIM(loX,loY,loZ);
  int LIST_MDIM(hiX,hiY,hiZ);
  int ndim;
  int level;
  int gridIdxOrBlkId;
  int tileIdx;
};

struct FlashxTileRawReals {
  milhoja::Real LIST_MDIM(deltaX,deltaY,deltaZ);
};

struct FlashxTileRaw {
  FlashxrTileRawPtrs sP;
  FlashxTileRawInts  sI;
  FlashxTileRawReals sR;
};
//}
#endif

