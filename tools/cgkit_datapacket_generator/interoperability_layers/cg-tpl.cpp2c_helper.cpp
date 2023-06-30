/* _connector:host_arguments */
const int queue1_h = packet_h->asynchronousQueue();
const int nTiles_h = packet_h->_nTiles_h;

/* _connector:c2f_argument_list */
void* packet_h,
const int queue1_h,
const void* _dt_d,
const void* _lo_d,
const void* _hi_d,
const void* _deltas_d,
const void* _loU_d,
const void* _U_d,
const void* _loAuxC_d,
const void* _auxC_d,
const void* _loFl_d,
const void* _flX_d,
const void* _flY_d,
const void* _flZ_d

/* _connector:c2f_arguments */
packet_h,
queue1_h,
_dt_d,
_lo_d,
_hi_d,
_deltas_d,
_loU_d,
_U_d,
_loAuxC_d,
_auxC_d,
_loFl_d,
_flX_d,
_flY_d,
_flZ_d

/* _connector:device_arguments */
void* _dt_d = static_cast<void*>( packet_h->_dt_d );
void* _lo_d = static_cast<void*>( packet_h->_lo_d );
void* _hi_d = static_cast<void*>( packet_h->_hi_d );
void* _deltas_d = static_cast<void*>( packet_h->_deltas_d );
void* _loU_d = static_cast<void*>( packet_h->_loU_d );
void* _U_d = static_cast<void*>( packet_h->_U_d );
void* _loAuxC_d = static_cast<void*>( packet_h->_loAuxC_d );
void* _auxC_d = static_cast<void*>( packet_h->_auxC_d );
void* _loFl_d = static_cast<void*>( packet_h->_loFl_d );
void* _flX_d = static_cast<void*>( packet_h->_flX_d );
void* _flY_d = static_cast<void*>( packet_h->_flY_d );
void* _flZ_d = static_cast<void*>( packet_h->_flZ_d );

/* _connector:instance_args */
const Real dt

/* _connector:host_members */
dt