#ifndef SLABGRAPH_CUH_
#define SLABGRAPH_CUH_

#include "slab_hash.cuh"
#include <cmath>
#include <numeric>
#include <type_traits>
#include <vector>

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

#define __BLOCK_SIZE__ 128

template <bool IsWeighted, typename VertexTy, typename ValueTy,
          typename ContainerPolicyTy, typename EdgeInsertionPolicyTy,
          typename EdgeUpdatePolicyTy, typename EdgeDeletionPolicyTy,
          typename EdgeSearchPolicyTy, typename CountTy = uint32_t>
struct EdgePolicyT {
  using VertexT = VertexTy;
  using EdgeValueT = ValueTy;
  using ContainerPolicyT = ContainerPolicyTy;
  using EdgeInsertionPolicyT = EdgeInsertionPolicyTy;
  using EdgeUpdatePolicyT = EdgeUpdatePolicyTy;
  using EdgeDeletionPolicyT = EdgeDeletionPolicyTy;
  using EdgeSearchPolicyT = EdgeSearchPolicyTy;
  using CountT = CountTy;

  static constexpr bool IsWeightedEdge = IsWeighted;
};

/* Forward declarations for edge insertion / update / deletion policies */

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
struct WeightedEdgeInsertionPolicy;

template <typename VertexTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
struct UnweightedEdgeInsertionPolicy;

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
struct WeightedRevEdgeInsertionPolicy;

template <typename VertexTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
struct UnweightedRevEdgeInsertionPolicy;

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
struct EdgeUpdatePolicy;

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
struct RevEdgeUpdatePolicy;

struct UnweightedEdgeUpdatePolicy;

template <typename VertexTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
struct EdgeDeletionPolicy;

template <typename VertexTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
struct RevEdgeDeletionPolicy;

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
struct WeightedEdgeSearchPolicy;

template <typename VertexTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
struct UnweightedEdgeSearchPolicy;

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
struct WeightedRevEdgeSearchPolicy;

template <typename VertexTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
struct UnweightedRevEdgeSearchPolicy;

/* End forward declarations */

template <typename InputIterator, typename OutputIterator, typename T>
OutputIterator ExclusiveScan(InputIterator First, InputIterator Last,
                             OutputIterator Out, T Init) {
  return thrust::exclusive_scan(thrust::host, First, Last, Out, Init);
}

template <typename SlabAllocPolicyTy, typename VertexTy = uint32_t,
          typename CountTy = uint32_t>
using UnweightedEdgePolicyT = EdgePolicyT<
    false, VertexTy, VertexTy, ConcurrentSetPolicy<VertexTy, SlabAllocPolicyTy>,
    UnweightedEdgeInsertionPolicy<
        VertexTy,
        typename ConcurrentSetPolicy<VertexTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    UnweightedEdgeUpdatePolicy,
    EdgeDeletionPolicy<VertexTy,
                       typename ConcurrentSetPolicy<
                           VertexTy, SlabAllocPolicyTy>::SlabHashContextT,
                       CountTy>,
    UnweightedEdgeSearchPolicy<
        VertexTy,
        typename ConcurrentSetPolicy<VertexTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    CountTy>;

template <typename SlabAllocPolicyTy, typename VertexTy = uint32_t,
          typename CountTy = uint32_t>
using RevUnweightedEdgePolicyT = EdgePolicyT<
    false, VertexTy, VertexTy, ConcurrentSetPolicy<VertexTy, SlabAllocPolicyTy>,
    UnweightedRevEdgeInsertionPolicy<
        VertexTy,
        typename ConcurrentSetPolicy<VertexTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    UnweightedEdgeUpdatePolicy,
    EdgeDeletionPolicy<VertexTy,
                       typename ConcurrentSetPolicy<
                           VertexTy, SlabAllocPolicyTy>::SlabHashContextT,
                       CountTy>,
    UnweightedRevEdgeSearchPolicy<
        VertexTy,
        typename ConcurrentSetPolicy<VertexTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    CountTy>;

template <typename SlabAllocPolicyTy, typename VertexTy = uint32_t,
          typename ValueTy = uint32_t, typename CountTy = uint32_t>
using WeightedEdgePolicyMapT = EdgePolicyT<
    true, VertexTy, ValueTy,
    ConcurrentMapPolicy<VertexTy, ValueTy, SlabAllocPolicyTy>,
    WeightedEdgeInsertionPolicy<
        VertexTy, ValueTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    EdgeUpdatePolicy<
        VertexTy, ValueTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    EdgeDeletionPolicy<
        VertexTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    WeightedEdgeSearchPolicy<
        VertexTy, ValueTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    CountTy>;

template <typename SlabAllocPolicyTy, typename VertexTy = uint32_t,
          typename ValueTy = uint32_t, typename CountTy = uint32_t>
using RevWeightedEdgePolicyMapT = EdgePolicyT<
    true, VertexTy, ValueTy,
    ConcurrentMapPolicy<VertexTy, ValueTy, SlabAllocPolicyTy>,
    WeightedRevEdgeInsertionPolicy<
        VertexTy, ValueTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    RevEdgeUpdatePolicy<
        VertexTy, ValueTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    RevEdgeDeletionPolicy<
        VertexTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    WeightedRevEdgeSearchPolicy<
        VertexTy, ValueTy,
        typename ConcurrentMapPolicy<VertexTy, ValueTy,
                                     SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    CountTy>;

template <typename SlabAllocPolicyTy, typename VertexTy = uint32_t,
          typename ValueTy = uint32_t, typename CountTy = uint32_t>
using WeightedEdgePolicyPCMapT = EdgePolicyT<
    true, VertexTy, ValueTy,
    PhaseConcurrentMapPolicy<VertexTy, ValueTy, SlabAllocPolicyTy>,
    WeightedEdgeInsertionPolicy<
        VertexTy, ValueTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    EdgeUpdatePolicy<
        VertexTy, ValueTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    EdgeDeletionPolicy<
        VertexTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    WeightedEdgeSearchPolicy<
        VertexTy, ValueTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    CountTy>;

template <typename SlabAllocPolicyTy, typename VertexTy = uint32_t,
          typename ValueTy = uint32_t, typename CountTy = uint32_t>
using RevWeightedEdgePolicyPCMapT = EdgePolicyT<
    true, VertexTy, ValueTy,
    PhaseConcurrentMapPolicy<VertexTy, ValueTy, SlabAllocPolicyTy>,
    WeightedRevEdgeInsertionPolicy<
        VertexTy, ValueTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    RevEdgeUpdatePolicy<
        VertexTy, ValueTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        typename SlabAllocPolicyTy::AllocatorContextT, CountTy>,
    RevEdgeDeletionPolicy<
        VertexTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    WeightedRevEdgeSearchPolicy<
        VertexTy, ValueTy,
        typename PhaseConcurrentMapPolicy<VertexTy, ValueTy,
                                          SlabAllocPolicyTy>::SlabHashContextT,
        CountTy>,
    CountTy>;

template <typename EdgePolicyT, bool IsWeighted> class DynamicGraph;
template <typename EdgePolicyT, bool IsWeighted> class DynamicGraphContext;

template <typename EdgePolicyT> class DynamicGraphContext<EdgePolicyT, true> {
public:
  using EdgePolicy = EdgePolicyT;
  using EdgeContainerPolicy = typename EdgePolicy::ContainerPolicyT;

  using EdgeAllocPolicy = typename EdgeContainerPolicy::AllocPolicyT;
  using EdgeDynAllocator = typename EdgeAllocPolicy::DynamicAllocatorT;
  using EdgeDynAllocCtxt = typename EdgeAllocPolicy::AllocatorContextT;

  using VertexT = typename EdgePolicy::VertexT;
  using EdgeValueT = typename EdgePolicy::EdgeValueT;

  using EdgeHashT = typename EdgeContainerPolicy::SlabHashT;
  using EdgeHashContext = typename EdgeContainerPolicy::SlabHashContextT;
  using EdgeInsertionPolicy = typename EdgePolicy::EdgeInsertionPolicyT;
  using EdgeUpdatePolicy = typename EdgePolicy::EdgeUpdatePolicyT;
  using EdgeDeletionPolicy = typename EdgePolicy::EdgeDeletionPolicyT;
  using EdgeSearchPolicy = typename EdgePolicy::EdgeSearchPolicyT;
  using SlabInfoT = typename EdgeContainerPolicy::SlabInfoT;
  
  using CountT = typename EdgePolicy::CountT;

private:
  EdgeDynAllocCtxt AllocCtxt;
  EdgeHashContext *EdgeHashCtxts;

  CountT *VertexDegrees;
  CountT *VertexBucketsOffsets;
  CountT *EdgesPerBucket;

public:
  friend class DynamicGraph<EdgePolicyT, true>;

  DynamicGraphContext()
      : AllocCtxt{}, EdgeHashCtxts{nullptr}, VertexDegrees{nullptr},
        VertexBucketsOffsets{nullptr}, EdgesPerBucket{nullptr} {}

  __host__ __device__
  DynamicGraphContext(const DynamicGraphContext<EdgePolicyT, true> &TheContext)
      : AllocCtxt{TheContext.AllocCtxt},
        EdgeHashCtxts{TheContext.EdgeHashCtxts},
        VertexDegrees{TheContext.VertexDegrees},
        VertexBucketsOffsets{TheContext.VertexBucketsOffsets},
        EdgesPerBucket{TheContext.EdgesPerBucket} {}

  __host__ DynamicGraphContext<EdgePolicy, true> &
  operator=(const DynamicGraphContext<EdgePolicy, true> &Other) {
    AllocCtxt = Other.AllocCtxt;
    EdgeHashCtxts = Other.EdgeHashCtxts;
    VertexDegrees = Other.VertexDegrees;
    VertexBucketsOffsets = Other.VertexBucketsOffsets;
    EdgesPerBucket = Other.EdgesPerBucket;
    return *(this);
  }

  __host__ __device__ ~DynamicGraphContext() {}

  __device__ __forceinline__ EdgeDynAllocCtxt &GetEdgeDynAllocCtxt() {
    return AllocCtxt;
  }

  __device__ __forceinline__ EdgeHashContext *GetEdgeHashCtxts() {
    return EdgeHashCtxts;
  }

  __device__ __forceinline__ uint32_t *GetVertexDegrees() {
    return VertexDegrees;
  }

  __device__ __forceinline__ uint32_t *GetVertexBucketsOffsets() {
    return VertexBucketsOffsets;
  }

  __device__ __forceinline__ uint32_t *GetEdgesPerBucket() {
    return EdgesPerBucket;
  }

  __device__ void InsertEdge(bool &ToInsert, uint32_t &LaneID,
                             VertexT &SourceVertex, VertexT &DestinationVertex,
                             EdgeValueT &EdgeValue,
                             EdgeDynAllocCtxt &AllocCtxt);

  template <typename FilterMapTy,
            typename = typename std::enable_if<
                std::is_default_constructible<FilterMapTy>::value>>
  __device__ void UpdateEdge(bool &ToUpdate, uint32_t &LaneID,
                             VertexT &SourceVertex, VertexT &DestinationVertex,
                             EdgeValueT &EdgeValue,
                             FilterMapTy *IncomingFilterMap = nullptr);

  __device__ bool SearchEdge(bool &ToSearch, uint32_t &LaneID,
                             VertexT &SourceVertex, VertexT &DestinationVertex,
                             EdgeValueT &EdgeValue);

  __device__ void DeleteEdge(bool &ToDelete, uint32_t &LaneID,
                             VertexT &SourceVertex, VertexT &DestinationVertex);
};

template <typename EdgePolicyT> class DynamicGraph<EdgePolicyT, true> {
public:
  using GraphContextT = DynamicGraphContext<EdgePolicyT, true>;

  using EdgePolicy = EdgePolicyT;
  using EdgeContainerPolicy = typename EdgePolicy::ContainerPolicyT;

  using EdgeAllocPolicy = typename EdgeContainerPolicy::AllocPolicyT;
  using EdgeDynAllocator = typename EdgeAllocPolicy::DynamicAllocatorT;
  using EdgeDynAllocCtxt = typename EdgeAllocPolicy::AllocatorContextT;

  using VertexT = typename EdgePolicy::VertexT;
  using EdgeValueT = typename EdgePolicy::EdgeValueT;

  using EdgeHashT = typename EdgeContainerPolicy::SlabHashT;
  using EdgeHashContext = typename EdgeContainerPolicy::SlabHashContextT;
  using EdgeInsertionPolicy = typename EdgePolicy::EdgeInsertionPolicyT;
  using EdgeUpdatePolicy = typename EdgePolicy::EdgeUpdatePolicyT;
  using EdgeDeletionPolicy = typename EdgePolicy::EdgeDeletionPolicyT;
  using EdgeSearchPolicy = typename EdgePolicy::EdgeSearchPolicyT;
    using SlabInfoT = typename EdgeContainerPolicy::SlabInfoT;

  using CountT = typename EdgePolicy::CountT;

private:
  uint32_t NodesMax;
  EdgeDynAllocator &EdgeAllocator;
  std::vector<EdgeHashT> EdgeContainer;
  DynamicGraphContext<EdgePolicy, true> TheGraphContext;

  uint32_t *HeadSlabPtr;
  uint32_t *VertexBucketsOffsetsHost;
  uint32_t *BucketsPerVertexHost;

  uint32_t *FirstUpdatedSlab;
  uint8_t *FirstUpdatedLaneId;
  bool *IsSlablistUpdated;

public:
  DynamicGraph(uint32_t N, EdgeDynAllocator &TheEdgeAllocator,
               double LoadFactor, uint32_t *VertexDegreeHints,
               uint32_t DeviceIdx = 0)
      : NodesMax{N}, EdgeAllocator{TheEdgeAllocator}, EdgeContainer{},
        TheGraphContext{}, HeadSlabPtr{nullptr},
        VertexBucketsOffsetsHost{nullptr}, BucketsPerVertexHost{nullptr},
        FirstUpdatedSlab{nullptr}, FirstUpdatedLaneId{nullptr},
        IsSlablistUpdated{nullptr} {
    cudaSetDevice(DeviceIdx);

    CHECK_CUDA_ERROR(
        cudaMalloc(&TheGraphContext.VertexDegrees, sizeof(uint32_t) * N));
    CHECK_CUDA_ERROR(
        cudaMemset(TheGraphContext.VertexDegrees, 0x00, sizeof(uint32_t) * N));

    CHECK_CUDA_ERROR(cudaMalloc(&TheGraphContext.VertexBucketsOffsets,
                                sizeof(uint32_t) * N));

    uint32_t TotalNumberOfBuckets = 0;
    BucketsPerVertexHost = new uint32_t[N];
    VertexBucketsOffsetsHost = new uint32_t[N];
    for (int i = 0; i < N; ++i) {
      uint32_t x = static_cast<uint32_t>(std::ceil(
          static_cast<double>(VertexDegreeHints[i]) / (32 * LoadFactor)));
      BucketsPerVertexHost[i] = (x == 0) ? 1 : x;
      TotalNumberOfBuckets += BucketsPerVertexHost[i];
    }

    ExclusiveScan(BucketsPerVertexHost, BucketsPerVertexHost + N,
                  VertexBucketsOffsetsHost, 0);

    CHECK_CUDA_ERROR(cudaMemcpy(TheGraphContext.VertexBucketsOffsets,
                                VertexBucketsOffsetsHost, sizeof(uint32_t) * N,
                                cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&TheGraphContext.EdgesPerBucket,
                                sizeof(uint32_t) * TotalNumberOfBuckets));
    CHECK_CUDA_ERROR(cudaMemset(TheGraphContext.EdgesPerBucket, 0x00,
                                sizeof(uint32_t) * TotalNumberOfBuckets));

    CHECK_CUDA_ERROR(cudaMalloc(&HeadSlabPtr, 128 * TotalNumberOfBuckets));
    CHECK_CUDA_ERROR(cudaMemset(HeadSlabPtr, 0xFF, 128 * TotalNumberOfBuckets));

    CHECK_CUDA_ERROR(
        cudaMalloc(&FirstUpdatedSlab, sizeof(uint32_t) * TotalNumberOfBuckets));
    thrust::fill(thrust::device, FirstUpdatedSlab,
                 FirstUpdatedSlab + TotalNumberOfBuckets,
                 static_cast<uint32_t>(SlabInfoT::A_INDEX_POINTER));

    CHECK_CUDA_ERROR(cudaMalloc(&FirstUpdatedLaneId,
                                sizeof(uint8_t) * TotalNumberOfBuckets));
    thrust::fill(thrust::device, FirstUpdatedLaneId,
                 FirstUpdatedLaneId + TotalNumberOfBuckets, 0);

    CHECK_CUDA_ERROR(
        cudaMalloc(&IsSlablistUpdated, sizeof(bool) * TotalNumberOfBuckets));
    thrust::fill(thrust::device, IsSlablistUpdated,
                 IsSlablistUpdated + TotalNumberOfBuckets, false);

    for (int i = 0; i < N; ++i)
      EdgeContainer.emplace_back(
          reinterpret_cast<int8_t *>(HeadSlabPtr) +
              (VertexBucketsOffsetsHost[i] * 128),
          FirstUpdatedSlab + VertexBucketsOffsetsHost[i],
          FirstUpdatedLaneId + VertexBucketsOffsetsHost[i],
          IsSlablistUpdated + VertexBucketsOffsetsHost[i],
          BucketsPerVertexHost[i], &EdgeAllocator, DeviceIdx);

    std::vector<EdgeHashContext> EdgeHashCtxts;

    auto GetContainerHashCtxt = [](auto &Container) {
      return Container.getSlabHashContext();
    };

    std::transform(std::begin(EdgeContainer), std::end(EdgeContainer),
                   std::back_inserter(EdgeHashCtxts), GetContainerHashCtxt);

    CHECK_CUDA_ERROR(cudaMalloc(&TheGraphContext.EdgeHashCtxts,
                                sizeof(EdgeHashContext) * N));
    CHECK_CUDA_ERROR(
        cudaMemcpy(TheGraphContext.EdgeHashCtxts, EdgeHashCtxts.data(),
                   sizeof(EdgeHashContext) * N, cudaMemcpyHostToDevice));

    TheGraphContext.AllocCtxt = *EdgeAllocator.getContextPtr();
  }

  GraphContextT &GetDynamicGraphContext() { return TheGraphContext; }

  ~DynamicGraph() {
    delete VertexBucketsOffsetsHost;
    delete BucketsPerVertexHost;
    CHECK_CUDA_ERROR(cudaFree(HeadSlabPtr));
    CHECK_CUDA_ERROR(cudaFree(FirstUpdatedSlab));
    CHECK_CUDA_ERROR(cudaFree(FirstUpdatedLaneId));
    CHECK_CUDA_ERROR(cudaFree(IsSlablistUpdated));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.EdgeHashCtxts));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.EdgesPerBucket));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.VertexBucketsOffsets));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.VertexDegrees));
  }

  uint32_t *GetVertexBucketsOffsets() { return VertexBucketsOffsetsHost; }
  uint32_t *GetBucketsPerVertex() { return BucketsPerVertexHost; }

  void InsertEdges(VertexT *SourceVertices, VertexT *DestinationVertices,
                   CountT NumberOfEdges, EdgeValueT *EdgeValues);

  void DeleteEdges(VertexT *SourceVertices, VertexT *DestinationVertices,
                   CountT NumberOfEdges);

  void UpdateSlabPointers(); // Function to shift the pointers to the first
                             // locations to update

  template <typename FilterMapTy>
  typename std::enable_if<
      std::is_default_constructible<FilterMapTy>::value>::type
  UpdateEdges(VertexT *SourceVertices, VertexT *DestinationVertices,
              CountT NumberOfEdges, EdgeValueT *NewEdgeValues,
              FilterMapTy *EdgeFilterMaps = nullptr);

#if 0
  void SearchEdges(VertexT *SourceVertices, VertexT *DestinationVertices,
                   uint32_t NumberOfEdges, EdgeValueT *EdgeValues);
#endif
};

template <typename EdgePolicyT> class DynamicGraphContext<EdgePolicyT, false> {
public:
  using EdgePolicy = EdgePolicyT;
  using EdgeContainerPolicy = typename EdgePolicy::ContainerPolicyT;

  using EdgeAllocPolicy = typename EdgeContainerPolicy::AllocPolicyT;
  using EdgeDynAllocator = typename EdgeAllocPolicy::DynamicAllocatorT;
  using EdgeDynAllocCtxt = typename EdgeAllocPolicy::AllocatorContextT;

  using VertexT = typename EdgePolicy::VertexT;
  using EdgeValueT = typename EdgePolicy::EdgeValueT;

  using EdgeHashT = typename EdgeContainerPolicy::SlabHashT;
  using EdgeHashContext = typename EdgeContainerPolicy::SlabHashContextT;
  using EdgeInsertionPolicy = typename EdgePolicy::EdgeInsertionPolicyT;
  using EdgeUpdatePolicy = typename EdgePolicy::EdgeUpdatePolicyT;
  using EdgeDeletionPolicy = typename EdgePolicy::EdgeDeletionPolicyT;
  using EdgeSearchPolicy = typename EdgePolicy::EdgeSearchPolicyT;

  using CountT = typename EdgePolicy::CountT;

private:
  EdgeDynAllocCtxt AllocCtxt;
  EdgeHashContext *EdgeHashCtxts;

  CountT *VertexDegrees;
  CountT *VertexBucketsOffsets;
  CountT *EdgesPerBucket;

public:
  friend class DynamicGraph<EdgePolicyT, false>;

  DynamicGraphContext()
      : AllocCtxt{}, EdgeHashCtxts{nullptr}, VertexDegrees{nullptr},
        VertexBucketsOffsets{nullptr}, EdgesPerBucket{nullptr} {}

  __host__ __device__
  DynamicGraphContext(const DynamicGraphContext<EdgePolicyT, false> &TheContext)
      : AllocCtxt{TheContext.AllocCtxt},
        EdgeHashCtxts{TheContext.EdgeHashCtxts},
        VertexDegrees{TheContext.VertexDegrees},
        VertexBucketsOffsets{TheContext.VertexBucketsOffsets},
        EdgesPerBucket{TheContext.EdgesPerBucket} {}

  __host__ __device__ DynamicGraphContext<EdgePolicy, false> &
  operator=(const DynamicGraphContext<EdgePolicy, false> &Other) {
    AllocCtxt = Other.AllocCtxt;
    EdgeHashCtxts = Other.EdgeHashCtxts;
    VertexDegrees = Other.VertexDegrees;
    VertexBucketsOffsets = Other.VertexBucketsOffsets;
    EdgesPerBucket = Other.EdgesPerBucket;
  }

  __host__ __device__ ~DynamicGraphContext() {}

  __device__ __forceinline__ EdgeDynAllocCtxt &GetEdgeDynAllocCtxt() {
    return AllocCtxt;
  }

  __device__ __forceinline__ uint32_t *GetVertexDegrees() {
    return VertexDegrees;
  }

  __device__ __forceinline__ uint32_t *GetVertexBucketsOffsets() {
    return VertexBucketsOffsets;
  }

  __device__ __forceinline__ uint32_t *GetEdgesPerBucket() {
    return EdgesPerBucket;
  }

  __device__ __forceinline__ EdgeHashContext *GetEdgeHashCtxts() {
    return EdgeHashCtxts;
  }

  __device__ void InsertEdge(bool &ToInsert, uint32_t &LaneID,
                             VertexT &SourceVertex, VertexT &DestinationVertex,
                             EdgeDynAllocCtxt &AllocCtxt);

  __device__ bool SearchEdge(bool &ToSearch, uint32_t &LaneID,
                             VertexT &SourceVertex, VertexT &DestinationVertex);

  __device__ void DeleteEdge(bool &ToDelete, uint32_t &LaneID,
                             VertexT &SourceVertex, VertexT &DestinationVertex);
};

template <typename EdgePolicyT> class DynamicGraph<EdgePolicyT, false> {
public:
  using GraphContextT = DynamicGraphContext<EdgePolicyT, false>;

  using EdgePolicy = EdgePolicyT;
  using EdgeContainerPolicy = typename EdgePolicy::ContainerPolicyT;

  using EdgeAllocPolicy = typename EdgeContainerPolicy::AllocPolicyT;
  using EdgeDynAllocator = typename EdgeAllocPolicy::DynamicAllocatorT;
  using EdgeDynAllocCtxt = typename EdgeAllocPolicy::AllocatorContextT;

  using VertexT = typename EdgePolicy::VertexT;
  using EdgeValueT = typename EdgePolicy::EdgeValueT;

  using EdgeHashT = typename EdgeContainerPolicy::SlabHashT;
  using EdgeHashContext = typename EdgeContainerPolicy::SlabHashContextT;
  using EdgeInsertionPolicy = typename EdgePolicy::EdgeInsertionPolicyT;
  using EdgeUpdatePolicy = typename EdgePolicy::EdgeUpdatePolicyT;
  using EdgeDeletionPolicy = typename EdgePolicy::EdgeDeletionPolicyT;
  using EdgeSearchPolicy = typename EdgePolicy::EdgeSearchPolicyT;
  using SlabInfoT = typename EdgeContainerPolicy::SlabInfoT;

  using CountT = typename EdgePolicy::CountT;

private:
  uint32_t NodesMax;
  EdgeDynAllocator &EdgeAllocator;
  std::vector<EdgeHashT> EdgeContainer;
  DynamicGraphContext<EdgePolicy, false> TheGraphContext;

  uint32_t *HeadSlabPtr;
  uint32_t *VertexBucketsOffsetsHost;
  uint32_t *BucketsPerVertexHost;

  uint32_t *FirstUpdatedSlab;
  uint8_t *FirstUpdatedLaneId;
  bool *IsSlablistUpdated;

public:
  DynamicGraph(uint32_t N, EdgeDynAllocator &TheEdgeAllocator,
               double LoadFactor, uint32_t *VertexDegreeHints,
               uint32_t DeviceIdx = 0)
      : NodesMax{N}, EdgeAllocator{TheEdgeAllocator}, EdgeContainer{},
        TheGraphContext{}, HeadSlabPtr{nullptr},
        VertexBucketsOffsetsHost{nullptr}, BucketsPerVertexHost{nullptr},
        FirstUpdatedSlab{nullptr}, FirstUpdatedLaneId{nullptr},
        IsSlablistUpdated{nullptr} {
    cudaSetDevice(DeviceIdx);

    CHECK_CUDA_ERROR(
        cudaMalloc(&TheGraphContext.VertexDegrees, sizeof(uint32_t) * N));
    CHECK_CUDA_ERROR(
        cudaMemset(TheGraphContext.VertexDegrees, 0x00, sizeof(uint32_t) * N));

    CHECK_CUDA_ERROR(cudaMalloc(&TheGraphContext.VertexBucketsOffsets,
                                sizeof(uint32_t) * N));

    uint32_t TotalNumberOfBuckets = 0;
    BucketsPerVertexHost = new uint32_t[N];
    VertexBucketsOffsetsHost = new uint32_t[N];
    for (int i = 0; i < N; ++i) {
      uint32_t x = static_cast<uint32_t>(std::ceil(
          static_cast<double>(VertexDegreeHints[i]) / (32 * LoadFactor)));
      BucketsPerVertexHost[i] = (x == 0) ? 1 : x;
      TotalNumberOfBuckets += BucketsPerVertexHost[i];
    }

    ExclusiveScan(BucketsPerVertexHost, BucketsPerVertexHost + N,
                  VertexBucketsOffsetsHost, 0);

    CHECK_CUDA_ERROR(cudaMemcpy(TheGraphContext.VertexBucketsOffsets,
                                VertexBucketsOffsetsHost, sizeof(uint32_t) * N,
                                cudaMemcpyHostToDevice));

    CHECK_CUDA_ERROR(cudaMalloc(&TheGraphContext.EdgesPerBucket,
                                sizeof(uint32_t) * TotalNumberOfBuckets));
    CHECK_CUDA_ERROR(cudaMemset(TheGraphContext.EdgesPerBucket, 0x00,
                                sizeof(uint32_t) * TotalNumberOfBuckets));
    CHECK_CUDA_ERROR(cudaMalloc(&HeadSlabPtr, 128 * TotalNumberOfBuckets));
    CHECK_CUDA_ERROR(cudaMemset(HeadSlabPtr, 0xFF, 128 * TotalNumberOfBuckets));

    CHECK_CUDA_ERROR(
        cudaMalloc(&FirstUpdatedSlab, sizeof(uint32_t) * TotalNumberOfBuckets));
    thrust::fill(thrust::device, FirstUpdatedSlab,
                 FirstUpdatedSlab + TotalNumberOfBuckets,
                 static_cast<uint32_t>(SlabInfoT::A_INDEX_POINTER));

    CHECK_CUDA_ERROR(cudaMalloc(&FirstUpdatedLaneId,
                                sizeof(uint8_t) * TotalNumberOfBuckets));
    thrust::fill(thrust::device, FirstUpdatedLaneId,
                 FirstUpdatedLaneId + TotalNumberOfBuckets, 0);

    CHECK_CUDA_ERROR(
        cudaMalloc(&IsSlablistUpdated, sizeof(bool) * TotalNumberOfBuckets));
    thrust::fill(thrust::device, IsSlablistUpdated,
                 IsSlablistUpdated + TotalNumberOfBuckets, false);

    for (int i = 0; i < N; ++i) {
      EdgeContainer.emplace_back(
          reinterpret_cast<int8_t *>(HeadSlabPtr) +
              (VertexBucketsOffsetsHost[i] * 128),
          FirstUpdatedSlab + VertexBucketsOffsetsHost[i],
          FirstUpdatedLaneId + VertexBucketsOffsetsHost[i],
          IsSlablistUpdated + VertexBucketsOffsetsHost[i],
          BucketsPerVertexHost[i], &EdgeAllocator, DeviceIdx);
    }

    std::vector<EdgeHashContext> EdgeHashCtxts;

    auto GetContainerHashCtxt = [](auto &Container) {
      return Container.getSlabHashContext();
    };

    std::transform(std::begin(EdgeContainer), std::end(EdgeContainer),
                   std::back_inserter(EdgeHashCtxts), GetContainerHashCtxt);

    CHECK_CUDA_ERROR(cudaMalloc(&TheGraphContext.EdgeHashCtxts,
                                sizeof(EdgeHashContext) * N));
    CHECK_CUDA_ERROR(
        cudaMemcpy(TheGraphContext.EdgeHashCtxts, EdgeHashCtxts.data(),
                   sizeof(EdgeHashContext) * N, cudaMemcpyHostToDevice));

    TheGraphContext.AllocCtxt = *EdgeAllocator.getContextPtr();
  }

  ~DynamicGraph() {
    delete VertexBucketsOffsetsHost;
    delete BucketsPerVertexHost;
    CHECK_CUDA_ERROR(cudaFree(HeadSlabPtr));
    CHECK_CUDA_ERROR(cudaFree(FirstUpdatedSlab));
    CHECK_CUDA_ERROR(cudaFree(FirstUpdatedLaneId));
    CHECK_CUDA_ERROR(cudaFree(IsSlablistUpdated));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.EdgeHashCtxts));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.EdgesPerBucket));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.VertexBucketsOffsets));
    CHECK_CUDA_ERROR(cudaFree(TheGraphContext.VertexDegrees));
  }

  GraphContextT &GetDynamicGraphContext() { return TheGraphContext; }

  uint32_t *GetVertexBucketsOffsets() { return VertexBucketsOffsetsHost; }
  uint32_t *GetBucketsPerVertex() { return BucketsPerVertexHost; }

  void InsertEdges(VertexT *SourceVertices, VertexT *DestinationVertices,
                   CountT NumberOfEdges);

  void DeleteEdges(VertexT *SourceVertices, VertexT *DestinationVertices,
                   CountT NumberOfEdges);

  void UpdateSlabPointers(); // Function to shift the pointers to the first
                             // locations to update

#if 0
  void SearchEdges(VertexT *SourceVertices,
                   VertexT *DestinationVertices,
                   CountT NumberOfEdges,
                   EdgeValueT *EdgeValues);
#endif
};

#include "warp/DeleteEdge.cuh"
#include "warp/InsertEdge.cuh"
#include "warp/SearchEdge.cuh"
#include "warp/UpdateEdge.cuh"

#include "device/DeleteEdgeKernel.cuh"
#include "device/InsertEdgeKernel.cuh"
#include "device/SearchEdgeKernel.cuh"
#include "device/UpdateEdgeKernel.cuh"
#include "device/UpdateSlablistPointers.cuh"

#include "SlabGraphImpl.cuh"


template <typename EdgePolicy>
using DynGraph = DynamicGraph<EdgePolicy, EdgePolicy::IsWeightedEdge>;

#endif // SLABGRAPH_CUH_
