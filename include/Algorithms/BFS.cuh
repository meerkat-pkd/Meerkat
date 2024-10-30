#pragma once

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/optional.h>

#include "Algorithms/definitions.cuh"
#include "Graph/SlabGraph.cuh"

template <typename AllocPolicy> struct BFS {
private:
  using EdgePolicy = UnweightedEdgePolicyT<AllocPolicy>;

public:
  using GraphT = DynGraph<EdgePolicy>;
  using DynGraphContext = typename GraphT::GraphContextT;

private:
  GraphT &G;
  uint32_t SourceVertex;
  uint32_t VertexN;
  distance_t *DistanceFromSrc;

  /* Device Properties */
  int SMCount;
  int WarpSize;
  int SharedMemorySize;
  int SMCoreCount;

public:
  BFS(GraphT &G, uint32_t SourceVertex, uint32_t VertexN)
      : G{G}, SourceVertex{SourceVertex}, VertexN{VertexN} {
    cudaDeviceProp Properties;
    cudaError_t ErrorStatus = cudaGetDeviceProperties(&Properties, 0);

    if (ErrorStatus != cudaSuccess)
      throw std::runtime_error("cudaGetDeviceProperties() Failed !!");

    SMCount = Properties.multiProcessorCount;
    WarpSize = Properties.warpSize;
    SharedMemorySize = Properties.sharedMemPerBlock;
    SMCoreCount = ConvertSMVersionToCores(Properties.major, Properties.minor);

    CHECK_ERROR(cudaMalloc(&DistanceFromSrc, sizeof(distance_t) * VertexN));
    CHECK_ERROR(
        cudaMemset(DistanceFromSrc, 0xFF, sizeof(distance_t) * VertexN));
    distance_t SrcDist = PACK(0, SourceVertex);
    CHECK_ERROR(cudaMemcpy(&DistanceFromSrc[SourceVertex], &SrcDist,
                           sizeof(distance_t), cudaMemcpyHostToDevice));
  }

  std::vector<uint32_t> Distances() {
    std::vector<uint32_t> TheDistances(VertexN, UINT32_MAX);
    CHECK_ERROR(cudaMemcpy(TheDistances.data(), DistanceFromSrc,
                           sizeof(float) * VertexN, cudaMemcpyDeviceToHost));
    return TheDistances;
  }

  ~BFS() { CHECK_ERROR(cudaFree(DistanceFromSrc)); }

  float Static(uint32_t EdgesN);
  float Incremental(uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev,
                    uint32_t EdgesN);
  float Decremental(uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev,
                    uint32_t EdgesN);
};

#include "Algorithms/impl/BFSImpl.cuh"