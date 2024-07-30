#ifndef SEARCH_EDGE_CUH_
#define SEARCH_EDGE_CUH_

namespace Search {
template <typename VertexTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
__device__ __forceinline__ void
Search(bool ToSearch, VertexTy &SourceVertex, VertexTy &DestinationVertex,
       uint32_t LaneID, EdgeHashContextTy *EdgeHashCtxts, bool &SearchStatus) {
  uint32_t WorkQueue = 0;
  uint32_t DestinationVertexBucket =
      ToSearch ? EdgeHashCtxts[SourceVertex].computeBucket(DestinationVertex)
               : 0xFFFFFFFFu;
  bool Status{};

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    uint32_t CurrentLane = __ffs(WorkQueue) - 1;
    uint32_t CurrentSourceVertex =
        __shfl_sync(0xFFFFFFFF, SourceVertex, CurrentLane, 32);
    bool SameSourceVertex = (SourceVertex == CurrentSourceVertex);
    bool ToBeSearched = ToSearch && SameSourceVertex;

    Status = EdgeHashCtxts[CurrentSourceVertex].searchKey(
        ToBeSearched, LaneID, DestinationVertex, DestinationVertexBucket);

    if (ToSearch && SameSourceVertex) {
      ToSearch = false;
      SearchStatus = Status;
    }
  }
}

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename CountTy = uint32_t>
__device__ __forceinline__ void
Search(bool ToSearch, VertexTy &SourceVertex, VertexTy &DestinationVertex,
       uint32_t LaneID, EdgeHashContextTy *EdgeHashCtxts, bool &SearchStatus,
       ValueTy &TheValue) {
  uint32_t WorkQueue = 0;
  uint32_t DestinationVertexBucket =
      ToSearch ? EdgeHashCtxts[SourceVertex].computeBucket(DestinationVertex)
               : 0xFFFFFFFFu;
  bool Status{};
  ValueTy Value{};

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    uint32_t CurrentLane = __ffs(WorkQueue) - 1;
    uint32_t CurrentSourceVertex =
        __shfl_sync(0xFFFFFFFF, SourceVertex, CurrentLane, 32);
    bool SameSourceVertex = (SourceVertex == CurrentSourceVertex);
    bool ToBeSearched = ToSearch && SameSourceVertex;

    Status = EdgeHashCtxts[CurrentSourceVertex].searchKey(
        ToBeSearched, LaneID, DestinationVertex, Value,
        DestinationVertexBucket);

    if (ToSearch && SameSourceVertex) {
      ToSearch = false;
      SearchStatus = Status;
      TheValue = Value;
    }
  }
}
} // namespace Search

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename CountTy>
struct WeightedEdgeSearchPolicy {
  __device__ __forceinline__ void
  Search(bool ToSearch, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         uint32_t LaneID, EdgeHashContextTy *EdgeHashCtxts, bool &SearchStatus,
         ValueTy &TheValue) {
    Search::Search<VertexTy, ValueTy, EdgeHashContextTy, CountTy>(
        ToSearch, SourceVertex, DestinationVertex, LaneID, EdgeHashCtxts,
        SearchStatus, TheValue);
  }
};

template <typename VertexTy, typename ValueTy, typename EdgeHashContextTy,
          typename CountTy>
struct WeightedRevEdgeSearchPolicy {
  __device__ __forceinline__ void
  Search(bool ToSearch, VertexTy &SourceVertex, VertexTy &DestinationVertex,
         uint32_t LaneID, EdgeHashContextTy *EdgeHashCtxts, bool &SearchStatus,
         ValueTy &TheValue) {
    Search::Search<VertexTy, ValueTy, EdgeHashContextTy, CountTy>(
        ToSearch, DestinationVertex, SourceVertex, LaneID, EdgeHashCtxts,
        SearchStatus, TheValue);
  }
};

template <typename VertexTy, typename EdgeHashContextTy, typename CountTy>
struct UnweightedEdgeSearchPolicy {
  __device__ __forceinline__ void Search(bool ToSearch, VertexTy &SourceVertex,
                                         VertexTy &DestinationVertex,
                                         uint32_t LaneID,
                                         EdgeHashContextTy *EdgeHashCtxts,
                                         bool &SearchStatus) {
    Search::Search<VertexTy, EdgeHashContextTy, CountTy>(
        ToSearch, SourceVertex, DestinationVertex, LaneID, EdgeHashCtxts,
        SearchStatus);
  }
};

template <typename VertexTy, typename EdgeHashContextTy, typename CountTy>
struct UnweightedRevEdgeSearchPolicy {
  __device__ __forceinline__ void Search(bool ToSearch, VertexTy &SourceVertex,
                                         VertexTy &DestinationVertex,
                                         uint32_t LaneID,
                                         EdgeHashContextTy *EdgeHashCtxts,
                                         bool &SearchStatus) {
    Search::Search<VertexTy, EdgeHashContextTy, CountTy>(
        ToSearch, DestinationVertex, SourceVertex, LaneID, EdgeHashCtxts,
        SearchStatus);
  }
};

template <typename EdgePolicyT>
__device__ bool DynamicGraphContext<EdgePolicyT, true>::SearchEdge(
    bool &ToSearch, uint32_t &LaneID, VertexT &SourceVertex,
    VertexT &DestinationVertex, EdgeValueT &TheValue) {

  typename DynamicGraphContext<EdgePolicyT, true>::EdgeSearchPolicy S{};
  bool SearchStatus{};

  S.Search(ToSearch, SourceVertex, DestinationVertex, LaneID, EdgeHashCtxts,
           SearchStatus, TheValue);
  return SearchStatus;
}

template <typename EdgePolicyT>
__device__ bool DynamicGraphContext<EdgePolicyT, false>::SearchEdge(
    bool &ToSearch, uint32_t &LaneID, VertexT &SourceVertex,
    VertexT &DestinationVertex) {

  typename DynamicGraphContext<EdgePolicyT, false>::EdgeSearchPolicy S{};
  bool SearchStatus{};

  S.Search(ToSearch, SourceVertex, DestinationVertex, LaneID, EdgeHashCtxts,
           SearchStatus);
  return SearchStatus;
}

#endif // SEARCH_EDGE_CUH_