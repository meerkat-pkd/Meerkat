#ifndef UPDATED_SLABLIST_POINTERS_CUH_
#define UPDATED_SLABLIST_POINTERS_CUH_

template <typename GraphContextT>
__global__ void UpdateSlabListPointers(GraphContextT TheGraphContext,
                                       uint32_t VertexN) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = (gridDim.x * blockDim.x);
  uint32_t LaneId = ThreadId & 0x1F;

  if ((ThreadId - LaneId) >= VertexN)
    return;

  int Vertex = 0xFFFFFFFF;
  bool ToUpdate = false;

  if (ThreadId < VertexN) {
    Vertex = ThreadId;
    ToUpdate = true;
  }

  using AdjacencyCtxt = typename GraphContextT::EdgeHashContext;
  using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;
  constexpr uint32_t INVALID_LANE = SlabInfoT::NEXT_PTR_LANE;

  AdjacencyCtxt *VertexAdjacencies = TheGraphContext.GetEdgeHashCtxts();
  uint32_t WorkQueue = 0;
  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToUpdate)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = ThreadId - LaneId + Lane;

    AdjacencyCtxt &VertexAdjacency = VertexAdjacencies[CurrentVertex];
    uint32_t NumBuckets = VertexAdjacency.getNumBuckets();

    uint32_t *FirstUpdatedSlabPtr =
        VertexAdjacency.getFirstUpdatedSlabPointer();
    uint8_t *FirstUpdatedLaneIdPtr =
        VertexAdjacency.getFirstUpdatedLaneIdPointer();
    bool *IsSlabListUpdatedPtr = VertexAdjacency.getIsSlablistUpdatedPointer();

    for (uint32_t BucketId = 0; BucketId < NumBuckets; ++BucketId) {
      uint32_t IsUpdated = false;
      if (LaneId == 0)
        IsUpdated = IsSlabListUpdatedPtr[BucketId];
      IsUpdated = __shfl_sync(0xFFFFFFFF, IsUpdated, 0, 32);

      if (!IsUpdated)
        continue;

      uint32_t FirstUpdatedSlab = 0xFFFFFFFF;
      uint8_t FirstUpdatedLaneId = INVALID_LANE;

      if (LaneId == SlabInfoT::NEXT_PTR_LANE) {
        FirstUpdatedSlab = FirstUpdatedSlabPtr[BucketId];
        FirstUpdatedLaneId = FirstUpdatedLaneIdPtr[BucketId];

        if (FirstUpdatedLaneId == INVALID_LANE) {
          FirstUpdatedSlab =
              (FirstUpdatedSlab == SlabInfoT::A_INDEX_POINTER)
                  ? *(VertexAdjacency.getPointerFromBucket(BucketId, LaneId))
                  : *(VertexAdjacency.getPointerFromSlab(FirstUpdatedSlab,
                                                         LaneId));
          FirstUpdatedLaneId = 0;
        }
      }

      FirstUpdatedSlab = __shfl_sync(0xFFFFFFFF, FirstUpdatedSlab,
                                     SlabInfoT::NEXT_PTR_LANE, 32);
      FirstUpdatedLaneId = __shfl_sync(0xFFFFFFFF, FirstUpdatedLaneId,
                                       SlabInfoT::NEXT_PTR_LANE, 32);

      uint32_t CurrentSlab = FirstUpdatedSlab;
      while (CurrentSlab != SlabInfoT::EMPTY_INDEX_POINTER) {
        uint32_t Data =
            (CurrentSlab == SlabInfoT::A_INDEX_POINTER)
                ? *(VertexAdjacency.getPointerFromBucket(BucketId, LaneId))
                : *(VertexAdjacency.getPointerFromSlab(CurrentSlab, LaneId));
        uint32_t NextLaneData =
            __shfl_sync(0xFFFFFFFF, Data, SlabInfoT::NEXT_PTR_LANE, 32);

        if (NextLaneData == SlabInfoT::EMPTY_INDEX_POINTER) {
          uint32_t NextUpdateLaneBitSet =
              (__ballot_sync(0xFFFFFFFF, Data == EMPTY_KEY) &
               SlabInfoT::REGULAR_NODE_KEY_MASK);

          if (LaneId == 0) {
            uint32_t UpdateLane = (NextUpdateLaneBitSet == 0)
                                      ? INVALID_LANE
                                      : (__ffs(NextUpdateLaneBitSet) - 1);
            FirstUpdatedSlabPtr[BucketId] = CurrentSlab;
            FirstUpdatedLaneIdPtr[BucketId] = UpdateLane;
            IsSlabListUpdatedPtr[BucketId] = false;
          }
        }

        CurrentSlab = NextLaneData;
      }
    }

    if (CurrentVertex == ThreadId)
      ToUpdate = false;
  }
}

#endif // UPDATED_SLABLIST_POINTERS_CUH_