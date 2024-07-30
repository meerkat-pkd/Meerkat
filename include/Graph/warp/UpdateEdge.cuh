#ifndef UPDATE_EDGE_CUH_
#define UPDATE_EDGE_CUH_

#include <limits>

namespace Up {
template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename FilterMapTy,
          typename CountTy = uint32_t>
__device__ __forceinline__ void
Update(bool ToUpdate, VertexTy &SourceVertex, VertexTy &DestinationVertex,
       ValueTy &EdgeValue, uint32_t LaneID, EdgeHashContextTy *EdgeHashCtxts,
       EdgeAllocatorContextTy &EdgeAllocCtxt, FilterMapTy *TheFilterMap) {
  uint32_t WorkQueue = 0;
  uint32_t DestinationVertexBucket =
      ToUpdate ? EdgeHashCtxts[SourceVertex].computeBucket(DestinationVertex)
               : std::numeric_limits<uint32_t>::max();

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToUpdate)) != 0) {
    uint32_t CurrentLane = __ffs(WorkQueue) - 1;
    uint32_t CurrentSourceVertex =
        __shfl_sync(0xFFFFFFFF, SourceVertex, CurrentLane, 32);
    bool SameSourceVertex = (SourceVertex == CurrentSourceVertex);

    EdgeHashCtxts[CurrentSourceVertex].template updatePair<FilterMapTy>(
        ToUpdate && SameSourceVertex, LaneID, DestinationVertex, EdgeValue,
        DestinationVertexBucket, EdgeAllocCtxt, TheFilterMap);

    if (ToUpdate && SameSourceVertex)
      ToUpdate = false;
  }
}

} // end namespace Up

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy>
struct EdgeUpdatePolicy {
  template <typename FilterMapTy>
  __device__ __forceinline__ void
  Update(bool ToUpdate, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         ValueTy &EdgeValue, uint32_t LaneID, EdgeHashContextTy *EdgeHashCtxts,
         EdgeAllocatorContextTy &EdgeAllocCtxt, FilterMapTy *TheFilterMap) {

    Up::Update<VertexTy, ValueTy, EdgeHashContextTy, EdgeAllocatorContextTy,
               FilterMapTy, CountTy>(ToUpdate, SourceVertex, DestinationVertex,
                                     EdgeValue, LaneID, EdgeHashCtxts,
                                     EdgeAllocCtxt, TheFilterMap);
  }
};

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename EdgeAllocatorContextTy, typename CountTy>
struct RevEdgeUpdatePolicy {
  template <typename FilterMapTy>
  __device__ __forceinline__ void
  Update(bool ToUpdate, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         ValueTy &EdgeValue, uint32_t LaneID, EdgeHashContextTy *EdgeHashCtxts,
         EdgeAllocatorContextTy &EdgeAllocCtxt, FilterMapTy *TheFilterMap) {

    Up::Update<VertexTy, ValueTy, EdgeHashContextTy, EdgeAllocatorContextTy,
               FilterMapTy, CountTy>(ToUpdate, DestinationVertex, SourceVertex,
                                     EdgeValue, LaneID, EdgeHashCtxts,
                                     EdgeAllocCtxt, TheFilterMap);
  }
};

struct UnweightedEdgeUpdatePolicy {};

template <typename EdgePolicyT>
template <typename FilterMapTy, typename _>
__device__ void DynamicGraphContext<EdgePolicyT, true>::UpdateEdge(
    bool &ToUpdate, uint32_t &LaneID, VertexT &SourceVertex,
    VertexT &DestinationVertex, EdgeValueT &EdgeValue,
    FilterMapTy *EdgeFilterMap) {

  typename DynamicGraphContext::EdgeUpdatePolicy U{};

  U.Update(ToUpdate, SourceVertex, DestinationVertex, EdgeValue, LaneID,
           EdgeHashCtxts, EdgeFilterMap);
}

#endif // UPDATE_EDGE_CUH_