#ifndef INSERT_EDGE_CUH_
#define INSERT_EDGE_CUH_

#include <limits>

namespace Ins {
template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
__device__ __forceinline__ void
Insert(bool ToInsert, VertexTy &SourceVertex, VertexTy &DestinationVertex,
       ValueTy &EdgeValue, uint32_t LaneID, CountTy *VertexDegrees,
       CountTy *VertexBucketsOffsets, CountTy *EdgesPerBucket,
       EdgeHashContextTy *EdgeHashCtxts,
       EdgeAllocatorContextTy &EdgeAllocCtxt) {
  uint32_t WorkQueue = 0;
  uint32_t DestinationVertexBucket =
      ToInsert ? EdgeHashCtxts[SourceVertex].computeBucket(DestinationVertex)
               : 0xFFFFFFFFu;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToInsert)) != 0) {
    uint32_t CurrentLane = __ffs(WorkQueue) - 1;
    uint32_t CurrentSourceVertex =
        __shfl_sync(0xFFFFFFFF, SourceVertex, CurrentLane, 32);
    bool SameSourceVertex = (SourceVertex == CurrentSourceVertex);
    bool ToBeInserted = ToInsert && SameSourceVertex;

#if 0
  // TODO: Fix this
    UpsertStatusKind InsertStatus =
        EdgeHashCtxts[CurrentSourceVertex]
            .template upsertPair<AlwaysTrueFilter<ValueTy>>(
                ToBeInserted, LaneID, DestinationVertex, EdgeValue,
                DestinationVertexBucket, EdgeAllocCtxt);

    uint32_t InsertionCount =
        __popc(__ballot_sync(0xFFFFFFFF, InsertStatus == USK_INSERT));
    if (LaneID == 0)
      atomicAdd(VertexDegrees + CurrentSourceVertex, InsertionCount);

    if (InsertStatus == USK_INSERT)
      atomicAdd(&EdgesPerBucket[VertexBucketsOffsets[SourceVertex] +
                                DestinationVertexBucket],
                1);
#endif

    EdgeHashCtxts[CurrentSourceVertex].insertPair(
        ToBeInserted, LaneID, DestinationVertex, EdgeValue,
        DestinationVertexBucket, EdgeAllocCtxt);
    

    uint32_t InsertionCount =
        __popc(__ballot_sync(0xFFFFFFFF, ToBeInserted));
    if (LaneID == 0)
      atomicAdd(VertexDegrees + CurrentSourceVertex, InsertionCount);

    if (ToBeInserted)
      atomicAdd(&EdgesPerBucket[VertexBucketsOffsets[SourceVertex] +
                                DestinationVertexBucket],
                1);

    if (ToInsert && SameSourceVertex)
      ToInsert = false;
  }
}

template <typename VertexTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy = uint32_t>
__device__ __forceinline__ void
Insert(bool ToInsert, VertexTy &SourceVertex, VertexTy &DestinationVertex,
       uint32_t LaneID, CountTy *VertexDegrees, CountTy *VertexBucketsOffsets,
       CountTy *EdgesPerBucket, EdgeHashContextTy *EdgeHashCtxts,
       EdgeAllocatorContextTy &EdgeAllocCtxt) {
  uint32_t WorkQueue = 0;

  uint32_t DestinationVertexBucket =
      ToInsert ? EdgeHashCtxts[SourceVertex].computeBucket(DestinationVertex)
               : 0xFFFFFFFFu;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToInsert)) != 0) {
    uint32_t CurrentLane = __ffs(WorkQueue) - 1;
    uint32_t CurrentSourceVertex =
        __shfl_sync(0xFFFFFFFF, SourceVertex, CurrentLane, 32);
    bool SameSourceVertex = (SourceVertex == CurrentSourceVertex);
    bool ToBeInserted = ToInsert && SameSourceVertex;

    bool InsertStatus = EdgeHashCtxts[CurrentSourceVertex].insertKey(
        ToBeInserted, LaneID, DestinationVertex, DestinationVertexBucket,
        EdgeAllocCtxt);

    uint32_t InsertionCount =
        __popc(__ballot_sync(0xFFFFFFFF, InsertStatus == USK_INSERT));
    if (LaneID == 0)
      atomicAdd(VertexDegrees + CurrentSourceVertex, InsertionCount);

    if (InsertStatus)
      atomicAdd(&EdgesPerBucket[VertexBucketsOffsets[SourceVertex] +
                                DestinationVertexBucket],
                1);

    if (ToInsert && SameSourceVertex)
      ToInsert = false;
  }
}

} // end namespace Ins

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy>
struct WeightedEdgeInsertionPolicy {
  __device__ __forceinline__ void
  Insert(bool ToInsert, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         ValueTy &EdgeValue, uint32_t LaneID, CountTy *VertexDegrees,
         CountTy *VertexBucketsOffsets, CountTy *EdgesPerBucket,
         EdgeHashContextTy *EdgeHashCtxts,
         EdgeAllocatorContextTy &EdgeAllocCtxt) {
    Ins::Insert<VertexTy, ValueTy, EdgeHashContextTy, EdgeAllocatorContextTy,
                CountTy>(ToInsert, SourceVertex, DestinationVertex, EdgeValue,
                         LaneID, VertexDegrees, VertexBucketsOffsets,
                         EdgesPerBucket, EdgeHashCtxts, EdgeAllocCtxt);
  }
};

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy>
struct WeightedRevEdgeInsertionPolicy {
  __device__ __forceinline__ void
  Insert(bool ToInsert, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         ValueTy &EdgeValue, uint32_t LaneID, CountTy *VertexDegrees,
         CountTy *VertexBucketsOffsets, CountTy *EdgesPerBucket,
         EdgeHashContextTy *EdgeHashCtxts,
         EdgeAllocatorContextTy &EdgeAllocCtxt) {
    Ins::Insert<VertexTy, ValueTy, EdgeHashContextTy, EdgeAllocatorContextTy,
                CountTy>(ToInsert, DestinationVertex, SourceVertex, EdgeValue,
                         LaneID, VertexDegrees, VertexBucketsOffsets,
                         EdgesPerBucket, EdgeHashCtxts, EdgeAllocCtxt);
  }
};

template <typename VertexTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy>
struct UnweightedEdgeInsertionPolicy {
  __device__ __forceinline__ void
  Insert(bool ToInsert, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         uint32_t LaneID, CountTy *VertexDegrees, CountTy *VertexBucketsOffsets,
         CountTy *EdgesPerBucket, EdgeHashContextTy *EdgeHashCtxts,
         EdgeAllocatorContextTy &EdgeAllocCtxt) {
    Ins::Insert<VertexTy, EdgeHashContextTy, EdgeAllocatorContextTy, CountTy>(
        ToInsert, SourceVertex, DestinationVertex, LaneID, VertexDegrees,
        VertexBucketsOffsets, EdgesPerBucket, EdgeHashCtxts, EdgeAllocCtxt);
  }
};

template <typename VertexTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy>
struct UnweightedRevEdgeInsertionPolicy {
  __device__ __forceinline__ void
  Insert(bool ToInsert, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         uint32_t LaneID, CountTy *VertexDegrees, CountTy *VertexBucketsOffsets,
         CountTy *EdgesPerBucket, EdgeHashContextTy *EdgeHashCtxts,
         EdgeAllocatorContextTy &EdgeAllocCtxt) {
    Ins::Insert<VertexTy, EdgeHashContextTy, EdgeAllocatorContextTy, CountTy>(
        ToInsert, DestinationVertex, SourceVertex, LaneID, VertexDegrees,
        VertexBucketsOffsets, EdgesPerBucket, EdgeHashCtxts, EdgeAllocCtxt);
  }
};

template <typename EdgePolicyT>
__device__ void DynamicGraphContext<EdgePolicyT, true>::InsertEdge(
    bool &ToInsert, uint32_t &LaneID, VertexT &SourceVertex,
    VertexT &DestinationVertex, EdgeValueT &EdgeValue,
    EdgeDynAllocCtxt &AllocCtxt) {

  typename DynamicGraphContext<EdgePolicyT, true>::EdgeInsertionPolicy I{};

  I.Insert(ToInsert, SourceVertex, DestinationVertex, EdgeValue, LaneID,
           VertexDegrees, VertexBucketsOffsets, EdgesPerBucket, EdgeHashCtxts,
           AllocCtxt);
}

template <typename EdgePolicyT>
__device__ void DynamicGraphContext<EdgePolicyT, false>::InsertEdge(
    bool &ToInsert, uint32_t &LaneID, VertexT &SourceVertex,
    VertexT &DestinationVertex, EdgeDynAllocCtxt &AllocCtxt) {

  typename DynamicGraphContext<EdgePolicyT, false>::EdgeInsertionPolicy I{};

  I.Insert(ToInsert, SourceVertex, DestinationVertex, LaneID, VertexDegrees,
           VertexBucketsOffsets, EdgesPerBucket, EdgeHashCtxts, AllocCtxt);
}

#endif