#pragma once

#include <cstdint>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>

#include "Graph/SlabGraph.cuh"

template <typename AllocPolicy> struct PageRank {
private:
  using Allocator = typename AllocPolicy::DynamicAllocatorT;
  using EdgePolicy = RevUnweightedEdgePolicyT<AllocPolicy>;

public:
  using GraphT = DynGraph<EdgePolicy>;

private:
  GraphT &G;
  float DampingFactor;
  float ErrorMargin;
  uint32_t VertexN;
  uint32_t MaxIter;

  float *PageRankValues;
  float *ContributionPerVertex;
  uint32_t *VertexOutDegreeDev;

  float PageRankComputation();
  float DynamicPageRankComputation(uint32_t *EdgesSrc, uint32_t *EdgesDst,
                                   uint32_t EdgesN, bool IsIncremental);

public:
  PageRank(GraphT &G, float DampingFactor, float ErrorMargin, uint32_t VertexN,
           uint32_t *VertexOutDegree, uint32_t MaxIter = 100)
      : G{G}, DampingFactor{DampingFactor}, ErrorMargin{ErrorMargin},
        VertexN{VertexN}, MaxIter{MaxIter} {
    assert(VertexN == VertexOutDegree.length());

    CHECK_ERROR(cudaMalloc(&PageRankValues, sizeof(float) * VertexN));
    CHECK_ERROR(cudaMalloc(&ContributionPerVertex, sizeof(float) * VertexN));
    CHECK_ERROR(cudaMalloc(&VertexOutDegreeDev, sizeof(uint32_t) * VertexN));

    thrust::fill(thrust::device, PageRankValues, PageRankValues + VertexN,
                 1 / static_cast<float>(VertexN));
    thrust::fill(thrust::device, ContributionPerVertex,
                 ContributionPerVertex + VertexN, 0);
    CHECK_ERROR(cudaMemcpy(VertexOutDegreeDev, VertexOutDegree,
                           sizeof(uint32_t) * VertexN, cudaMemcpyHostToDevice));
  }

  std::vector<float> Values() {
    std::vector<float> TheValues(VertexN, 0.0f);
    CHECK_ERROR(cudaMemcpy(TheValues.data(), PageRankValues,
                           sizeof(float) * VertexN, cudaMemcpyDeviceToHost));
    return TheValues;
  }

  ~PageRank() {
    CHECK_ERROR(cudaFree(VertexOutDegreeDev));
    CHECK_ERROR(cudaFree(ContributionPerVertex));
    CHECK_ERROR(cudaFree(PageRankValues));
  }

  float Static();
  float Incremental(uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev,
                    uint32_t EdgesN);
  float Decremental(uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev,
                    uint32_t EdgesN);
};

#include "Algorithms/impl/PageRankImpl.cuh"