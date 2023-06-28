/* _connector:size_constructor */
SIZE_DT + SIZE_NTILES

/* _connector:size_tilemetadata */
SIZE_TILE_DELTAS + SIZE_TILE_LO + SIZE_TILE_HI

/* _connector:size_tilein */
0

/* _connector:size_tileinout */
SIZE_U

/* _connector:size_tileout */
0

/* _connector:size_tilescratch */
SIZE_AUXC + SIZE_FLX + SIZE_FLY + SIZE_FLZ

/* _connector:constructor_args */
Real dt

/* _connector:set_members */
_dt_h{dt},
_dt_d{nullptr},
_nTiles_h{tiles_.size()},
_nTiles_d{nullptr},
_deltas_d{nullptr},
_lo_d{nullptr},
_hi_d{nullptr},
_U_d{nullptr},
_auxC_d{nullptr},
_flX_d{nullptr},
_flY_d{nullptr},
_flZ_d{nullptr}

/* _connector:host_members */
_dt_h

/* _connector:size_determination */
constexpr std::size_t SIZE_DT = pad( sizeof(Real) );
constexpr std::size_t SIZE_NTILES = pad( sizeof(int) );
constexpr std::size_t SIZE_TILE_DELTAS = 3 * sizeof(Real);
constexpr std::size_t SIZE_TILE_LO = 3 * sizeof(int);
constexpr std::size_t SIZE_TILE_HI = 3 * sizeof(int);
constexpr std::size_t SIZE_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * (8 - 0 + 1) * sizeof(Real);
constexpr std::size_t SIZE_AUXC = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * sizeof(Real);
constexpr std::size_t SIZE_FLX = ((8 + 2 * 1) + 1) * (8 + 2 * 1) * (1 + 2 * 0) * (5) * sizeof(Real);
constexpr std::size_t SIZE_FLY = (8 + 2 * 1) * ((8 + 2 * 1) + 1) * (1 + 2 * 0) * (5) * sizeof(Real);
constexpr std::size_t SIZE_FLZ = (1) * (1) * (1) * (1) * sizeof(Real);

/* _connector:pointers_constructor */
Real* dt_p = static_cast<Real*>( static_cast<void*>(ptr_p) );
_dt_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
ptr_p+=SIZE_DT;
ptr_d+=SIZE_DT;

int* nTiles_p = static_cast<int*>( static_cast<void*>(ptr_p) );
_nTiles_d = static_cast<int*>( static_cast<void*>(ptr_d) );
ptr_p+=SIZE_NTILES;
ptr_d+=SIZE_NTILES;


/* _connector:memcpy_constructor */
std::memcpy(dt_p, static_cast<void*>(&_dt_h), SIZE_DT);
std::memcpy(nTiles_p, static_cast<void*>(&_nTiles_h), SIZE_NTILES);

/* _connector:public_members */
Real _dt_h;
Real* _dt_d;
int _nTiles_h;
int* _nTiles_d;
Real* _deltas_d;
int* _lo_d;
int* _hi_d;
Real* _U_d;
Real* _auxC_d;
Real* _flX_d;
Real* _flY_d;
Real* _flZ_d;

/* _connector:pointers_tilemetadata */
Real* deltas_p = static_cast<Real*>( static_cast<void*>(ptr_p) );
_deltas_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_TILE_DELTAS;
ptr_d+=_nTiles_h * SIZE_TILE_DELTAS;

int* lo_p = static_cast<int*>( static_cast<void*>(ptr_p) );
_lo_d = static_cast<int*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_TILE_LO;
ptr_d+=_nTiles_h * SIZE_TILE_LO;

int* hi_p = static_cast<int*>( static_cast<void*>(ptr_p) );
_hi_d = static_cast<int*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_TILE_HI;
ptr_d+=_nTiles_h * SIZE_TILE_HI;


/* _connector:memcpy_tilemetadata */
Real deltas_h[3] = { deltas.I(), deltas.J(), deltas.K() };
char_ptr = static_cast<char*>( static_cast<void*>(deltas_p) ) + n * SIZE_TILE_DELTAS;
std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(deltas_h), SIZE_TILE_DELTAS);

int lo_h[3] = { lo.I()+1, lo.J()+1, lo.K()+1 };
char_ptr = static_cast<char*>( static_cast<void*>(lo_p) ) + n * SIZE_TILE_LO;
std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(lo_h), SIZE_TILE_LO);

int hi_h[3] = { hi.I()+1, hi.J()+1, hi.K()+1 };
char_ptr = static_cast<char*>( static_cast<void*>(hi_p) ) + n * SIZE_TILE_HI;
std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(hi_h), SIZE_TILE_HI);


/* _connector:tile_descriptor */
const RealVect deltas = tileDesc_h->deltas();
const IntVect lo = tileDesc_h->lo();
const IntVect hi = tileDesc_h->hi();

/* _connector:pointers_tilein */

/* _connector:memcpy_tilein */

/* _connector:memcpy_tileinout */
std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 8 - 0 + 1 ) * sizeof(Real);
char_ptr = static_cast<char*>( static_cast<void*>(U_p) ) + n * SIZE_U;
std::memcpy(static_cast<void*>(char_ptr), static_cast<void*>(data_h + offset_U), nBytes_U);
pinnedPtrs_[n].CC1_data = static_cast<Real*>( static_cast<void*>(char_ptr) );

/* _connector:pointers_tileinout */
Real* U_p = static_cast<Real*>( static_cast<void*>(ptr_p) );
_U_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
ptr_p+=_nTiles_h * SIZE_U;
ptr_d+=_nTiles_h * SIZE_U;


/* _connector:unpack_tileinout */
std::size_t offset_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * static_cast<std::size_t>(0);
Real*        start_h_U = data_h + offset_U;
const Real*  start_p_U = data_p + offset_U;
std::size_t nBytes_U = (8 + 2 * 1) * (8 + 2 * 1) * (1 + 2 * 0) * ( 7 - 0 + 1 ) * sizeof(Real);
std::memcpy(static_cast<void*>(start_h_U), static_cast<const void*>(start_p_U), nBytes_U);

/* _connector:pointers_tileout */

/* _connector:memcpy_tileout */

/* _connector:unpack_tileout */

/* _connector:pointers_tilescratch */
_auxC_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
ptr_d+= _nTiles_h * SIZE_AUXC;

_flX_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
ptr_d+= _nTiles_h * SIZE_FLX;

_flY_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
ptr_d+= _nTiles_h * SIZE_FLY;

_flZ_d = static_cast<Real*>( static_cast<void*>(ptr_d) );
ptr_d+= _nTiles_h * SIZE_FLZ;


/* _connector:memcpy_tilescratch */

