#ifndef DELETE_EDGE_CUH_
#define DELETE_EDGE_CUH_

namespace Del {
template <typename VertexTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
__device__ __forceinline__ void
Delete(bool ToDelete, VertexTy &SourceVertex, VertexTy &DestinationVertex,
       uint32_t LaneID, CountTy *VertexDegrees, CountTy *VertexBucketOffsets,
       CountTy *EdgesPerBucket, EdgeHashContextTy *EdgeHashCtxts) {
  uint32_t WorkQueue = 0;
  uint32_t DestinationVertexBucket =
      ToDelete ? EdgeHashCtxts[SourceVertex].computeBucket(DestinationVertex)
               : 0xFFFFFFFFu;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToDelete)) != 0) {
    uint32_t CurrentLane = __ffs(WorkQueue) - 1;
    uint32_t CurrentSourceVertex =
        __shfl_sync(0xFFFFFFFF, SourceVertex, CurrentLane, 32);
    bool SameSourceVertex = (SourceVertex == CurrentSourceVertex);
    bool ToBeDeleted = ToDelete && SameSourceVertex;

    bool DeletionStatus = EdgeHashCtxts[CurrentSourceVertex].deleteKey(
        ToBeDeleted, LaneID, DestinationVertex,
        DestinationVertexBucket);

    uint32_t DeletionCount = __popc(__ballot_sync(0xFFFFFFFF, DeletionStatus));
    if (LaneID == 0)
      atomicSub(VertexDegrees + CurrentSourceVertex, DeletionCount);

    if (DeletionStatus)
      atomicSub(&EdgesPerBucket[VertexBucketOffsets[SourceVertex] +
                                DestinationVertexBucket],
                1);

    if (ToDelete && SameSourceVertex)
      ToDelete = false;
  }
}
} // namespace Del

template <typename VertexTy, typename EdgeHashContextTy, typename CountTy>
struct EdgeDeletionPolicy {
  __device__ __forceinline__ void
  Delete(bool ToDelete, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         uint32_t LaneID, CountTy *VertexDegrees, CountTy *VertexBucketOffsets,
         CountTy *EdgesPerBucket, EdgeHashContextTy *EdgeHashCtxts) {
    Del::Delete<VertexTy, EdgeHashContextTy, CountTy>(
        ToDelete, SourceVertex, DestinationVertex, LaneID, VertexDegrees,
        VertexBucketOffsets, EdgesPerBucket, EdgeHashCtxts);
  }
};

template <typename VertexTy, typename EdgeHashContextTy, typename CountTy>
struct RevEdgeDeletionPolicy {
  __device__ __forceinline__ void
  Delete(bool ToDelete, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         uint32_t LaneID, CountTy *VertexDegrees, CountTy *VertexBucketOffsets,
         CountTy *EdgesPerBucket, EdgeHashContextTy *EdgeHashCtxts) {
    Del::Delete<VertexTy, EdgeHashContextTy, CountTy>(
        ToDelete, DestinationVertex, SourceVertex, LaneID, VertexDegrees,
        VertexBucketOffsets, EdgesPerBucket, EdgeHashCtxts);
  }
};

template <typename EdgePolicyT>
__device__ void DynamicGraphContext<EdgePolicyT, true>::DeleteEdge(
    bool &ToDelete, uint32_t &LaneID, VertexT &SourceVertex,
    VertexT &DestinationVertex) {

  typename DynamicGraphContext<EdgePolicyT, true>::EdgeDeletionPolicy D{};

  D.Delete(ToDelete, SourceVertex, DestinationVertex, LaneID, VertexDegrees,
           VertexBucketsOffsets, EdgesPerBucket, EdgeHashCtxts);
}

template <typename EdgePolicyT>
__device__ void DynamicGraphContext<EdgePolicyT, false>::DeleteEdge(
    bool &ToDelete, uint32_t &LaneID, VertexT &SourceVertex,
    VertexT &DestinationVertex) {

  typename DynamicGraphContext<EdgePolicyT, false>::EdgeDeletionPolicy D{};

  D.Delete(ToDelete, SourceVertex, DestinationVertex, LaneID, VertexDegrees,
           VertexBucketsOffsets, EdgesPerBucket, EdgeHashCtxts);
}

#endif // DELETE_EDGE_CUH_