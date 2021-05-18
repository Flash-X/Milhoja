/* _connector:setup */
/* _param:idx = i */
__global__
void _param:axyFunction(_param:axyType _param:a, _param:axyType *_param:x, _param:axyType *_param:y)
{
  int _param:idx = blockIdx.x*blockDim.x + threadIdx.x;
  /* _link:kernel */
}
/* _connector:execute */
/* _param:threadBlockSize = 320 */
_param:axyFunction<<<_param:size/_param:threadBlockSize, _param:threadBlockSize>>>(_param:a, _param:x, _param:y);
