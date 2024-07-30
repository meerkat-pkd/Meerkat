#ifndef LOCK_CUH_
#define LOCK_CUH_

namespace util {
__forceinline__ __device__ uint32_t VolatileRead(volatile uint32_t *Address) {
  uint32_t Data;

#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.relaxed.sys.u32 %0, [%1];"
               : "=r"(Data)
               : "l"(Address));
#else
  Data = *Address;
#endif

  return Data;
}

__forceinline__ __device__ void VolatileWrite(volatile uint32_t *Address,
                                              uint32_t Data) {
#if __CUDA_ARCH__ >= 700
  asm volatile("ld.global.relaxed.sys.u32 %0, [%1];"
               : "=r"(Data)
               : "l"(Address));
#else
  *Address = Data;
#endif
}

__forceinline__ __device__ bool TryLock(uint32_t *Address,
                                        uint32_t WarpMask = 0xFFFFFFFF) {
  bool IsLocked = false;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if (LaneID == __popc(WarpMask))
    IsLocked = (atomicOr(Address, 0x80000000) & 0x80000000) != 0;

  IsLocked = __shfl_sync(WarpMask, IsLocked, 0, 32);
  __threadfence();
  return IsLocked;
}

__forceinline__ __device__ void Lock(uint32_t *Address,
                                     uint32_t WarpMask = 0xFFFFFFFF) {
  bool IsLocked = false;
  uint32_t LaneID = threadIdx.x & 0x1F;

  while (!IsLocked) {
    if (LaneID == __popc(WarpMask))
      IsLocked = (atomicOr(Address, 0x80000000) & 0x80000000) != 0;

    IsLocked = __shfl_sync(WarpMask, IsLocked, 0, 32);
  }

  __threadfence();
}

__forceinline__ __device__ void Unlock(uint32_t *Address,
                                       uint32_t WarpMask = 0xFFFFFFFF) {
  uint32_t LaneID = threadIdx.x & 0x1F;
  __threadfence();

  if (LaneID == __popc(WarpMask))
    atomicAnd(Address, 0x7FFFFFFF);
}
} // namespace util

#endif // LOCK_CUH_