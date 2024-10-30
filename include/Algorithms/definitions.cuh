#pragma once

#define SRC 0
#define INF 0xFFFFFFFF
#define WARP_SIZE 32
#define INFINF 0xFFFFFFFFFFFFFFFFULL
#define INVALID_VERTEX 0xFFFFFFFF
#define DISTANCE_VALID 0
#define DISTANCE_INVALIDATED 1
#define DISTANCE_UPDATED 2

using distance_t = unsigned long long int;

#define PACK(DISTANCE, PARENT)                                                 \
  (static_cast<distance_t>(DISTANCE) << 32) | (static_cast<distance_t>(PARENT))
#define DISTANCE(X) static_cast<uint32_t>((X) >> 32)
#define PARENT(X) static_cast<uint32_t>((X) & INF)

#include <cooperative_groups.h>

inline static int ConvertSMVersionToCores(int major, int minor) {
  typedef struct {
    int SMVersion;
    int Cores;
  } SMVersionToCores;

  SMVersionToCores GpuArchCoresPerSM[] = {
      {0x30, 192}, {0x32, 192}, {0x35, 192}, {0x37, 192}, {0x50, 128},
      {0x52, 128}, {0x53, 128}, {0x60, 64},  {0x61, 128}, {0x62, 128},
      {0x70, 64},  {0x72, 64},  {0x75, 64},  {0x80, 64},  {0x86, 128},
      {0x87, 128}, {0x89, 128}, {0x90, 128}, {-1, -1}};

  int I = 0;
  int GpuSMVersion = (major << 4) + minor;
  while (GpuArchCoresPerSM[I].SMVersion != -1) {
    if (GpuArchCoresPerSM[I].SMVersion == GpuSMVersion)
      return GpuArchCoresPerSM[I].Cores;
    ++I;
  }
  return -1;
}

namespace cg = cooperative_groups;