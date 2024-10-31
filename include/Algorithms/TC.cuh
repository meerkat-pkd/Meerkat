#pragma once

#include "Algorithms/definitions.cuh"
#include "Graph/SlabGraph.cuh"
#include <cstdint>
#include <thrust/device_vector.h>
#include <utility>

template <typename AllocPolicy> struct TC {
private:
  using EdgePolicy = UnweightedEdgePolicyT<AllocPolicy>;

public:
  using GraphT = DynGraph<EdgePolicy>;
  using DynGraphContext = typename GraphT::GraphContextT;

private:
  GraphT &G;
  uint32_t VertexN;
  unsigned long long int TriangleCount;

  void ReverseMapping(uint32_t *Src, uint32_t *Dst, uint32_t Length,
                      thrust::device_vector<uint32_t> &Mapping,
                      uint32_t &MappingLength, uint32_t *MappedSrc,
                      uint32_t *MappedDst);

public:
  TC(GraphT &G, uint32_t VertexN) : G{G}, VertexN{VertexN}, TriangleCount{0} {}

  unsigned long long int GetCount() const { return TriangleCount; }

  float Static(uint32_t EdgesN);

  float
  Incremental(uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev, uint32_t EdgeN,
              std::unique_ptr<typename AllocPolicy::DynamicAllocatorT> &Alloc);

  float
  Decremental(uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev, uint32_t EdgeN,
              std::unique_ptr<typename AllocPolicy::DynamicAllocatorT> &Alloc);
};

#include "Algorithms/impl/TCImpl.cuh"