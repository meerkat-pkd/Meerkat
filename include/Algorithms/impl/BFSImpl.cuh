#pragma once

__device__ __forceinline__ static void
Invalidate(cg::grid_group &G, uint32_t *BatchSrc, uint32_t *BatchDst,
           uint32_t BatchEdgesN, distance_t *DistanceFromSource,
           uint32_t *Invalidated) {
  for (uint32_t I = G.thread_rank(); I < BatchEdgesN; I += G.size()) {
    uint32_t Dst = BatchDst[I];
    if (PARENT(DistanceFromSource[Dst]) == BatchSrc[I]) {
      if (atomicCAS(Invalidated + Dst, DISTANCE_VALID, DISTANCE_INVALIDATED) ==
          DISTANCE_VALID)
        DistanceFromSource[Dst] = INFINF;
    }
  }
}

__device__ __forceinline__ static void
PropogateInvalidation(cg::grid_group &G, distance_t *DistanceFromSource,
                      uint32_t Src, uint32_t VertexN, uint32_t *Invalidated) {
  uint32_t Stride = G.size();
  for (int I = G.thread_rank(); I < VertexN; I += Stride) {
    distance_t DX = DistanceFromSource[I];

    if (DX != INFINF) {
      bool ToUpdate = false;
      uint32_t Ancestor = PARENT(DX);

      while (!ToUpdate && Ancestor != Src) {
        distance_t D = DistanceFromSource[Ancestor];
        if (D == INFINF)
          ToUpdate = true;
        else
          Ancestor = PARENT(D);
      }

      if (ToUpdate) {
        Invalidated[I] = DISTANCE_INVALIDATED;
        DistanceFromSource[I] = INFINF;
      }
    }
  }
}

__device__ __forceinline__ static void
FlushSharedMemPartition(cg::thread_block_tile<32> &Warp, uint32_t *FrontierSrc,
                        uint32_t *FrontierDst, uint32_t *FrontierSize,
                        uint32_t *SMemQueueSrc, uint32_t *SMemQueueDst,
                        uint32_t &SMemQueueLen) {
  uint32_t Offset = 0;
  if (Warp.thread_rank() == 0)
    Offset = atomicAdd(FrontierSize, SMemQueueLen);
  Offset = Warp.shfl(Offset, 0);

  uint32_t Index = Warp.thread_rank();
  uint32_t Limit = (SMemQueueLen >> 5) << 5;
  uint32_t RemainderLen = SMemQueueLen - Limit;

  for (; Index < Limit; Index += WARP_SIZE) {
    FrontierSrc[Offset + Index] = SMemQueueSrc[Index];
    FrontierDst[Offset + Index] = SMemQueueDst[Index];
  }

  if (Warp.thread_rank() < RemainderLen) {
    FrontierSrc[Offset + Index] = SMemQueueSrc[Index];
    FrontierDst[Offset + Index] = SMemQueueDst[Index];
  }

  SMemQueueLen = 0;
}

template <typename DynGraphContext>
__device__ __forceinline__ static void
PushIntoSharedMemPartition(cg::thread_block_tile<32> &Warp, bool HasEdge,
                           uint32_t Src, uint32_t Dst, uint32_t *FrontierSrc,
                           uint32_t *FrontierDst, uint32_t *FrontierSize,
                           uint32_t *SMemQueueSrc, uint32_t *SMemQueueDst,
                           uint32_t &SMemQueueLen, uint32_t SMemEdgesPerWarp) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  uint32_t BitSet = Warp.ballot(HasEdge) & SlabInfoT::REGULAR_NODE_KEY_MASK;

  uint32_t EdgesToQueue = __popc(BitSet);
  if ((SMemQueueLen + EdgesToQueue) > SMemEdgesPerWarp)
    FlushSharedMemPartition(Warp, FrontierSrc, FrontierDst, FrontierSize,
                            SMemQueueSrc, SMemQueueDst, SMemQueueLen);

  if (HasEdge) {
    uint32_t Offset =
        __popc(__brev(BitSet) & (0xFFFFFFFF << (32 - Warp.thread_rank())));
    SMemQueueSrc[SMemQueueLen + Offset] = Src;
    SMemQueueDst[SMemQueueLen + Offset] = Dst;
  }

  SMemQueueLen += EdgesToQueue;
}

template <typename DynGraphContext, typename VertexAdjacencies>
__device__ __forceinline__ void
BFSIterationCore(cg::thread_block_tile<32> &Warp, VertexAdjacencies *VA,
                 bool &ToConsider, uint32_t CurrentDst,
                 uint32_t *NextFrontierSrc, uint32_t *NextFrontierDst,
                 uint32_t *NextFrontierSize, uint32_t *WarpSMQueueSrc,
                 uint32_t *WarpSMQueueDst, uint32_t &WarpSMemQueueLen,
                 uint32_t MaxEdgesInSMem, uint32_t *Queued) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  uint32_t WorkQueue = 0;
  while ((WorkQueue = Warp.ballot(ToConsider)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = Warp.shfl(CurrentDst, Lane);
    uint32_t SQ = 0;

    if (Warp.thread_rank() == 0 && !Queued[CurrentVertex] &&
        atomicCAS(&Queued[CurrentVertex], 0, 1) == 0)
      SQ = 1;
    SQ = Warp.shfl(SQ, 0);

    if (SQ) {
      typename AdjacencyContext::Iterator First = (VA[CurrentVertex]).Begin();
      typename AdjacencyContext::Iterator Last = (VA[CurrentVertex]).End();

      while (First != Last) {
        uint32_t AdjacentVertex = *First.GetPointer(Warp.thread_rank());
        bool HasAdjacentVertex =
            ((1 << Warp.thread_rank()) & SlabInfoT::REGULAR_NODE_KEY_MASK) &&
            (AdjacentVertex != EMPTY_KEY) &&
            ((AdjacentVertex != TOMBSTONE_KEY)) &&
            (CurrentVertex != AdjacentVertex);

        PushIntoSharedMemPartition<DynGraphContext>(
            Warp, HasAdjacentVertex, CurrentVertex, AdjacentVertex,
            NextFrontierSrc, NextFrontierDst, NextFrontierSize, WarpSMQueueSrc,
            WarpSMQueueDst, WarpSMemQueueLen, MaxEdgesInSMem);
        ++First;
      }
    }

    if (Lane == Warp.thread_rank())
      ToConsider = false;
  }
}

template <typename DynGraphContext, typename VertexAdjacencies>
__device__ __forceinline__ void
BFSIterationCoreNQ(cg::thread_block_tile<32> &Warp, VertexAdjacencies *VA,
                   bool &ToConsider, uint32_t CurrentDst,
                   uint32_t *NextFrontierSrc, uint32_t *NextFrontierDst,
                   uint32_t *NextFrontierSize, uint32_t *WarpSMQueueSrc,
                   uint32_t *WarpSMQueueDst, uint32_t &WarpSMemQueueLen,
                   uint32_t MaxEdgesInSMem) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  uint32_t WorkQueue = 0;
  while ((WorkQueue = Warp.ballot(ToConsider)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = Warp.shfl(CurrentDst, Lane);

    typename AdjacencyContext::Iterator First = (VA[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last = (VA[CurrentVertex]).End();

    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(Warp.thread_rank());
      bool HasAdjacentVertex =
          ((1 << Warp.thread_rank()) & SlabInfoT::REGULAR_NODE_KEY_MASK) &&
          (AdjacentVertex != EMPTY_KEY) &&
          ((AdjacentVertex != TOMBSTONE_KEY)) &&
          (CurrentVertex != AdjacentVertex);

      PushIntoSharedMemPartition<DynGraphContext>(
          Warp, HasAdjacentVertex, CurrentVertex, AdjacentVertex,
          NextFrontierSrc, NextFrontierDst, NextFrontierSize, WarpSMQueueSrc,
          WarpSMQueueDst, WarpSMemQueueLen, MaxEdgesInSMem);
      ++First;
    }

    if (Lane == Warp.thread_rank())
      ToConsider = false;
  }
}

template <typename DynGraphContext, typename VertexAdjacencies>
__device__ __forceinline__ void
StaticBFSIterationCoreNQ(cg::thread_block_tile<32> &Warp, VertexAdjacencies *VA,
                         bool &ToConsider, uint32_t CurrentDst,
                         distance_t *DistanceFromSource,
                         uint32_t *NextFrontierSrc, uint32_t *NextFrontierDst,
                         uint32_t *NextFrontierSize, uint32_t *WarpSMQueueSrc,
                         uint32_t *WarpSMQueueDst, uint32_t &WarpSMemQueueLen,
                         uint32_t MaxEdgesInSMem) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  uint32_t WorkQueue = 0;
  while ((WorkQueue = Warp.ballot(ToConsider)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = Warp.shfl(CurrentDst, Lane);

    typename AdjacencyContext::Iterator First = (VA[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last = (VA[CurrentVertex]).End();

    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(Warp.thread_rank());
      bool HasAdjacentVertex =
          ((1 << Warp.thread_rank()) & SlabInfoT::REGULAR_NODE_KEY_MASK) &&
          (AdjacentVertex != EMPTY_KEY) &&
          ((AdjacentVertex != TOMBSTONE_KEY)) &&
          (CurrentVertex != AdjacentVertex) &&
          (DISTANCE(DistanceFromSource[AdjacentVertex]) == INF);

      PushIntoSharedMemPartition<DynGraphContext>(
          Warp, HasAdjacentVertex, CurrentVertex, AdjacentVertex,
          NextFrontierSrc, NextFrontierDst, NextFrontierSize, WarpSMQueueSrc,
          WarpSMQueueDst, WarpSMemQueueLen, MaxEdgesInSMem);
      ++First;
    }

    if (Lane == Warp.thread_rank())
      ToConsider = false;
  }
}

template <typename DynGraphContext, typename VertexAdjacencies>
__device__ __forceinline__ void
BFSIterationCoreNQInv(cg::thread_block_tile<32> &Warp, VertexAdjacencies *VA,
                      bool &ToConsider, uint32_t CurrentDst,
                      uint32_t *Invalidated, uint32_t *NextFrontierSrc,
                      uint32_t *NextFrontierDst, uint32_t *NextFrontierSize,
                      uint32_t *WarpSMQueueSrc, uint32_t *WarpSMQueueDst,
                      uint32_t &WarpSMemQueueLen, uint32_t MaxEdgesInSMem) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  uint32_t WorkQueue = 0;
  while ((WorkQueue = Warp.ballot(ToConsider)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = Warp.shfl(CurrentDst, Lane);

    typename AdjacencyContext::Iterator First = (VA[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last = (VA[CurrentVertex]).End();

    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(Warp.thread_rank());
      bool HasAdjacentVertex =
          ((1 << Warp.thread_rank()) & SlabInfoT::REGULAR_NODE_KEY_MASK) &&
          (AdjacentVertex != EMPTY_KEY) &&
          ((AdjacentVertex != TOMBSTONE_KEY)) &&
          (CurrentVertex != AdjacentVertex) &&
          (Invalidated[AdjacentVertex] == DISTANCE_INVALIDATED); // EXTRA COND

      PushIntoSharedMemPartition<DynGraphContext>(
          Warp, HasAdjacentVertex, CurrentVertex, AdjacentVertex,
          NextFrontierSrc, NextFrontierDst, NextFrontierSize, WarpSMQueueSrc,
          WarpSMQueueDst, WarpSMemQueueLen, MaxEdgesInSMem);
      ++First;
    }

    if (Lane == Warp.thread_rank())
      ToConsider = false;
  }
}

template <typename DynGraphContext>
__global__ void SetLevelDynamicIncremental(
    DynGraphContext TheGraphContext, uint32_t *BatchSrc, uint32_t *BatchDst,
    uint32_t BatchSize, uint32_t *FrontierSrc, uint32_t *FrontierDst,
    uint32_t *NextFrontierSrc, uint32_t *NextFrontierDst,
    uint32_t *NextFrontierSize, distance_t *DistanceFromSource,
    uint32_t *Queued, uint32_t SMemElementsPerWarp) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  extern __shared__ uint32_t SMemQueue[];
  uint32_t ThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t Stride = gridDim.x * blockDim.x;
  cg::thread_block ThreadBlock = cg::this_thread_block();
  cg::grid_group Grid = cg::this_grid();
  cg::thread_block_tile<32> Warp = cg::tiled_partition<32>(ThreadBlock);

  typename DynGraphContext::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  uint32_t MaxEdgesInSMem = SMemElementsPerWarp >> 1;
  uint32_t *WarpSMQueueSrc =
      SMemQueue + (SMemElementsPerWarp * (threadIdx.x >> 5));
  uint32_t *WarpSMQueueDst = WarpSMQueueSrc + MaxEdgesInSMem;

  if (BatchSize != 0) {
    uint32_t WarpSMemQueueLen = 0;
    uint32_t LoopLimit = ((BatchSize + WARP_SIZE - 1) >> 5) << 5;
    for (uint32_t I = ThreadId; I < LoopLimit; I += Stride) {
      bool ToConsider = false;
      uint32_t CurrentSrc = 0xFFFFFFFF;
      uint32_t CurrentDst = 0xFFFFFFFF;

      if (I < BatchSize) {
        CurrentSrc = BatchSrc[I];
        CurrentDst = BatchDst[I];
        if (CurrentSrc != CurrentDst) {
          distance_t SrcDist = DistanceFromSource[CurrentSrc];
          uint32_t D = DISTANCE(SrcDist) + 1;
          distance_t DistFromCurrentSrc = PACK(D, CurrentSrc);
          ToConsider = (SrcDist != INFINF) &&
                       (D < DISTANCE(DistanceFromSource[CurrentDst])) &&
                       (atomicMin(&DistanceFromSource[CurrentDst],
                                  DistFromCurrentSrc) > DistFromCurrentSrc);
        }
      }

      BFSIterationCore<DynGraphContext>(
          Warp, VertexAdjacencies, ToConsider, CurrentDst, FrontierSrc,
          FrontierDst, NextFrontierSize, WarpSMQueueSrc, WarpSMQueueDst,
          WarpSMemQueueLen, MaxEdgesInSMem, Queued);
    }
    FlushSharedMemPartition(Warp, FrontierSrc, FrontierDst, NextFrontierSize,
                            WarpSMQueueSrc, WarpSMQueueDst, WarpSMemQueueLen);
    Grid.sync();
  }

  uint32_t FrontierSize = *NextFrontierSize;
  Grid.sync();

  if (Grid.thread_rank() == 0)
    *NextFrontierSize = 0;
  for (uint32_t I = ThreadId; I < FrontierSize; I += Stride)
    Queued[FrontierSrc[I]] = 0;
  Grid.sync();

  while (FrontierSize != 0) {
    uint32_t WarpSMemQueueLen = 0;
    uint32_t LoopLimit = ((FrontierSize + WARP_SIZE - 1) >> 5) << 5;

    for (uint32_t I = ThreadId; I < LoopLimit; I += Stride) {
      bool ToConsider = false;
      uint32_t CurrentSrc = 0xFFFFFFFF;
      uint32_t CurrentDst = 0xFFFFFFFF;

      if (I < FrontierSize) {
        CurrentSrc = FrontierSrc[I];
        CurrentDst = FrontierDst[I];

        if (CurrentSrc != CurrentDst) {
          uint32_t D = DISTANCE(DistanceFromSource[CurrentSrc]) + 1;
          distance_t DistFromCurrentSrc = PACK(D, CurrentSrc);
          ToConsider = (D < DISTANCE(DistanceFromSource[CurrentDst])) &&
                       (atomicMin(DistanceFromSource + CurrentDst,
                                  DistFromCurrentSrc) > DistFromCurrentSrc);
        }
      }

      BFSIterationCore<DynGraphContext>(
          Warp, VertexAdjacencies, ToConsider, CurrentDst, NextFrontierSrc,
          NextFrontierDst, NextFrontierSize, WarpSMQueueSrc, WarpSMQueueDst,
          WarpSMemQueueLen, MaxEdgesInSMem, Queued);
    }

    FlushSharedMemPartition(Warp, NextFrontierSrc, NextFrontierDst,
                            NextFrontierSize, WarpSMQueueSrc, WarpSMQueueDst,
                            WarpSMemQueueLen);

    thrust::swap(FrontierSrc, NextFrontierSrc);
    thrust::swap(FrontierDst, NextFrontierDst);
    Grid.sync();

    FrontierSize = *NextFrontierSize;
    Grid.sync();

    if (Grid.thread_rank() == 0)
      *NextFrontierSize = 0;
    for (uint32_t I = ThreadId; I < FrontierSize; I += Stride)
      Queued[FrontierSrc[I]] = 0;
    Grid.sync();
  }
}

template <typename DynGraphContext>
__global__ void SetLevelDynamicDecremental(
    DynGraphContext TheGraphContext, uint32_t TheSrc, uint32_t VertexN,
    uint32_t *BatchSrc, uint32_t *BatchDst, uint32_t BatchSize,
    uint32_t *FrontierSrc, uint32_t *FrontierDst, uint32_t *NextFrontierSrc,
    uint32_t *NextFrontierDst, uint32_t *NextFrontierSize,
    distance_t *DistanceFromSource, uint32_t *Queued,
    uint32_t SMemElementsPerWarp) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  extern __shared__ uint32_t SMemQueue[];
  uint32_t ThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t Stride = gridDim.x * blockDim.x;
  cg::thread_block ThreadBlock = cg::this_thread_block();
  cg::grid_group Grid = cg::this_grid();
  cg::thread_block_tile<32> Warp = cg::tiled_partition<32>(ThreadBlock);

  typename DynGraphContext::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  uint32_t MaxEdgesInSMem = SMemElementsPerWarp >> 1;
  uint32_t *WarpSMQueueSrc =
      SMemQueue + (SMemElementsPerWarp * (threadIdx.x >> 5));
  uint32_t *WarpSMQueueDst = WarpSMQueueSrc + MaxEdgesInSMem;

  uint32_t *Invalidated = Queued;
  Invalidate(Grid, BatchSrc, BatchDst, BatchSize, DistanceFromSource,
             Invalidated);
  Grid.sync();
  PropogateInvalidation(Grid, DistanceFromSource, TheSrc, VertexN, Invalidated);
  if (Grid.thread_rank() == 0)
    *NextFrontierSize = 0;
  Grid.sync();

  /* create frontier decremental. */
  {
    uint32_t WarpSMemQueueLen = 0;
    uint32_t LoopLimit = ((VertexN + WARP_SIZE - 1)) << 5;
    for (uint32_t I = ThreadId; I < LoopLimit; I += Stride) {
      bool ToConsider = false;
      uint32_t CurrentV = INVALID_VERTEX;
      if (I < VertexN) {
        CurrentV = I;
        ToConsider = (DistanceFromSource[CurrentV] != INFINF);
      }

      BFSIterationCoreNQInv<DynGraphContext>(
          Warp, VertexAdjacencies, ToConsider, CurrentV, Invalidated,
          FrontierSrc, FrontierDst, NextFrontierSize, WarpSMQueueSrc,
          WarpSMQueueDst, WarpSMemQueueLen, MaxEdgesInSMem);
    }
    FlushSharedMemPartition(Warp, FrontierSrc, FrontierDst, NextFrontierSize,
                            WarpSMQueueSrc, WarpSMQueueDst, WarpSMemQueueLen);
    Grid.sync();
  }

  uint32_t FrontierSize = *NextFrontierSize;
  Grid.sync();

  if (Grid.thread_rank() == 0)
    *NextFrontierSize = 0;
  for (uint32_t I = ThreadId; I < VertexN; I += Stride)
    Invalidated[I] = 0;
  Grid.sync();

  while (FrontierSize != 0) {
    uint32_t WarpSMemQueueLen = 0;
    uint32_t LoopLimit = ((FrontierSize + WARP_SIZE - 1) >> 5) << 5;

    for (uint32_t I = ThreadId; I < LoopLimit; I += Stride) {
      bool ToConsider = false;
      uint32_t CurrentSrc = 0xFFFFFFFF;
      uint32_t CurrentDst = 0xFFFFFFFF;

      if (I < FrontierSize) {
        CurrentSrc = FrontierSrc[I];
        CurrentDst = FrontierDst[I];

        if (CurrentSrc != CurrentDst) {
          uint32_t D = DISTANCE(DistanceFromSource[CurrentSrc]) + 1;
          distance_t DistFromCurrentSrc = PACK(D, CurrentSrc);
          ToConsider = (D < DISTANCE(DistanceFromSource[CurrentDst])) &&
                       (atomicMin(DistanceFromSource + CurrentDst,
                                  DistFromCurrentSrc) > DistFromCurrentSrc);
        }
      }

      BFSIterationCore<DynGraphContext>(
          Warp, VertexAdjacencies, ToConsider, CurrentDst, NextFrontierSrc,
          NextFrontierDst, NextFrontierSize, WarpSMQueueSrc, WarpSMQueueDst,
          WarpSMemQueueLen, MaxEdgesInSMem, Queued);
    }

    FlushSharedMemPartition(Warp, NextFrontierSrc, NextFrontierDst,
                            NextFrontierSize, WarpSMQueueSrc, WarpSMQueueDst,
                            WarpSMemQueueLen);

    thrust::swap(FrontierSrc, NextFrontierSrc);
    thrust::swap(FrontierDst, NextFrontierDst);
    Grid.sync();

    FrontierSize = *NextFrontierSize;
    Grid.sync();

    if (Grid.thread_rank() == 0)
      *NextFrontierSize = 0;
    for (uint32_t I = ThreadId; I < FrontierSize; I += Stride)
      Queued[FrontierSrc[I]] = 0;
    Grid.sync();
  }
}

template <typename DynGraphContext>
__global__ void
SetLevelStatic(DynGraphContext TheGraphContext, uint32_t TheSrc,
               uint32_t *FrontierSrc, uint32_t *FrontierDst,
               uint32_t *NextFrontierSrc, uint32_t *NextFrontierDst,
               uint32_t *NextFrontierSize, distance_t *DistanceFromSource,
               uint32_t SMemElementsPerWarp) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  extern __shared__ uint32_t SMemQueue[];
  uint32_t ThreadId = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t Stride = gridDim.x * blockDim.x;
  cg::thread_block ThreadBlock = cg::this_thread_block();
  cg::grid_group Grid = cg::this_grid();
  cg::thread_block_tile<32> Warp = cg::tiled_partition<32>(ThreadBlock);

  typename DynGraphContext::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  uint32_t MaxEdgesInSMem = SMemElementsPerWarp >> 1;
  uint32_t *WarpSMQueueSrc =
      SMemQueue + (SMemElementsPerWarp * (threadIdx.x >> 5));
  uint32_t *WarpSMQueueDst = WarpSMQueueSrc + MaxEdgesInSMem;

  if (Grid.thread_rank() < WARP_SIZE) {
    uint32_t WarpSMQueueLen = 0;
    bool ToConsider = (Grid.thread_rank() == 0);

    BFSIterationCoreNQ<DynGraphContext>(
        Warp, VertexAdjacencies, ToConsider, TheSrc, FrontierSrc, FrontierDst,
        NextFrontierSize, WarpSMQueueSrc, WarpSMQueueDst, WarpSMQueueLen,
        MaxEdgesInSMem);
    Warp.sync();
    FlushSharedMemPartition(Warp, FrontierSrc, FrontierDst, NextFrontierSize,
                            WarpSMQueueSrc, WarpSMQueueDst, WarpSMQueueLen);
  }

  Grid.sync();

  uint32_t FrontierSize = *NextFrontierSize;
  Grid.sync();

  if (Grid.thread_rank() == 0)
    *NextFrontierSize = 0;
  Grid.sync();

  uint32_t CurrentLevel = 1;
  while (FrontierSize != 0) {
    uint32_t WarpSMemQueueLen = 0;
    uint32_t LoopLimit = ((FrontierSize + WARP_SIZE - 1) >> 5) << 5;

    for (uint32_t I = ThreadId; I < LoopLimit; I += Stride) {
      bool ToConsider = false;
      uint32_t CurrentSrc = 0xFFFFFFFF;
      uint32_t CurrentDst = 0xFFFFFFFF;

      if (I < FrontierSize) {
        CurrentSrc = FrontierSrc[I];
        CurrentDst = FrontierDst[I];
        if (CurrentSrc != CurrentDst) {
          distance_t DistanceParentPair = PACK(CurrentLevel, CurrentSrc);
          ToConsider = (atomicCAS(&DistanceFromSource[CurrentDst], INFINF,
                                  DistanceParentPair) == INFINF);
        }
      }

      StaticBFSIterationCoreNQ<DynGraphContext>(
          Warp, VertexAdjacencies, ToConsider, CurrentDst, DistanceFromSource,
          NextFrontierSrc, NextFrontierDst, NextFrontierSize, WarpSMQueueSrc,
          WarpSMQueueDst, WarpSMemQueueLen, MaxEdgesInSMem);
    }

    FlushSharedMemPartition(Warp, NextFrontierSrc, NextFrontierDst,
                            NextFrontierSize, WarpSMQueueSrc, WarpSMQueueDst,
                            WarpSMemQueueLen);

    thrust::swap(FrontierSrc, NextFrontierSrc);
    thrust::swap(FrontierDst, NextFrontierDst);
    CurrentLevel += 1;
    Grid.sync();

    FrontierSize = *NextFrontierSize;
    Grid.sync();

    if (Grid.thread_rank() == 0)
      *NextFrontierSize = 0;
    Grid.sync();
  }
}

template <typename AllocPolicy>
float BFS<AllocPolicy>::Static(uint32_t EdgesN) {
  uint32_t *Queued;
  uint32_t *FrontierSrcDev;
  uint32_t *FrontierDstDev;
  uint32_t *FrontierSizeDev;

  uint32_t *NextFrontierSrcDev;
  uint32_t *NextFrontierDstDev;
  uint32_t *NextFrontierSizeDev;

  CHECK_ERROR(cudaMalloc(&Queued, sizeof(uint32_t) * VertexN));

  CHECK_ERROR(cudaMalloc(&FrontierSrcDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&FrontierDstDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&FrontierSizeDev, sizeof(uint32_t)));

  CHECK_ERROR(
      cudaMalloc(&NextFrontierSrcDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(
      cudaMalloc(&NextFrontierDstDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&NextFrontierSizeDev, sizeof(uint32_t)));

  CHECK_ERROR(cudaMemset(FrontierSizeDev, 0x00, sizeof(uint32_t)));
  CHECK_ERROR(cudaMemset(NextFrontierSizeDev, 0x00, sizeof(uint32_t)));

  CHECK_ERROR(cudaMemset(Queued, 0x00, sizeof(uint32_t) * VertexN));

  DynGraphContext Ctxt = G.GetDynamicGraphContext();

  uint32_t BlockSize = 512;
  uint32_t GridSize = SMCount;
  uint32_t WarpsN = BlockSize / WarpSize;
  dim3 BlockDim(BlockSize, 1, 1);
  dim3 GridDim(GridSize, 1, 1);
  uint32_t SMem = (1 << 15);

  cudaFuncSetAttribute(
      reinterpret_cast<void *>(SetLevelStatic<DynGraphContext>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, SMem);

  uint32_t SMemPerWarp = SMem / WarpsN;
  uint32_t SMemElementsPerWarp = SMemPerWarp / sizeof(int);

  void *KernelArgs[] = {static_cast<void *>(&Ctxt),
                        static_cast<void *>(&SourceVertex),
                        static_cast<void *>(&FrontierSrcDev),
                        static_cast<void *>(&FrontierDstDev),
                        static_cast<void *>(&NextFrontierSrcDev),
                        static_cast<void *>(&NextFrontierDstDev),
                        static_cast<void *>(&NextFrontierSizeDev),
                        static_cast<void *>(&DistanceFromSrc),
                        static_cast<void *>(&SMemElementsPerWarp)};

  float ElapsedTime;
  cudaEvent_t Start, Stop;
  cudaStream_t DefaultStream;
  cudaEventCreate(&Start);
  cudaEventCreate(&Stop);
  cudaStreamCreate(&DefaultStream);
  cudaEventRecord(Start, DefaultStream);

  CHECK_ERROR(cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(SetLevelStatic<DynGraphContext>), GridDim,
      BlockDim, KernelArgs, SMem, DefaultStream));

  cudaEventRecord(Stop, DefaultStream);
  cudaEventSynchronize(Stop);
  cudaStreamDestroy(DefaultStream);

  cudaEventElapsedTime(&ElapsedTime, Start, Stop);
  return ElapsedTime;
}

template <typename AllocPolicy>
float BFS<AllocPolicy>::Incremental(uint32_t *EdgesSrcDev,
                                    uint32_t *EdgesDstDev, uint32_t EdgesN) {

  uint32_t *Queued;
  uint32_t *FrontierSrcDev;
  uint32_t *FrontierDstDev;
  uint32_t *FrontierSizeDev;

  uint32_t *NextFrontierSrcDev;
  uint32_t *NextFrontierDstDev;
  uint32_t *NextFrontierSizeDev;

  float ElapsedTime;

  CHECK_ERROR(cudaMalloc(&Queued, sizeof(uint32_t) * VertexN));

  CHECK_ERROR(cudaMalloc(&FrontierSrcDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&FrontierDstDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&FrontierSizeDev, sizeof(uint32_t)));

  CHECK_ERROR(
      cudaMalloc(&NextFrontierSrcDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(
      cudaMalloc(&NextFrontierDstDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&NextFrontierSizeDev, sizeof(uint32_t)));

  CHECK_ERROR(cudaMemset(FrontierSizeDev, 0x00, sizeof(uint32_t)));
  CHECK_ERROR(cudaMemset(NextFrontierSizeDev, 0x00, sizeof(uint32_t)));

  CHECK_ERROR(cudaMemset(Queued, 0x00, sizeof(uint32_t) * VertexN));

  DynGraphContext Ctxt = G.GetDynamicGraphContext();

  uint32_t BlockSize = 512;
  uint32_t GridSize = SMCount;
  uint32_t WarpsN = BlockSize / WarpSize;
  dim3 BlockDim(BlockSize, 1, 1);
  dim3 GridDim(GridSize, 1, 1);
  uint32_t SMem = (1 << 15);

  uint32_t SMemPerWarp = SMem / WarpsN;
  uint32_t SMemElementsPerWarp = SMemPerWarp / sizeof(int);

  void *KernelArgs[] = {static_cast<void *>(&Ctxt),
                        static_cast<void *>(&EdgesSrcDev),
                        static_cast<void *>(&EdgesDstDev),
                        static_cast<void *>(&EdgesN),
                        static_cast<void *>(&FrontierSrcDev),
                        static_cast<void *>(&FrontierDstDev),
                        static_cast<void *>(&NextFrontierSrcDev),
                        static_cast<void *>(&NextFrontierDstDev),
                        static_cast<void *>(&NextFrontierSizeDev),
                        static_cast<void *>(&DistanceFromSrc),
                        static_cast<void *>(&Queued),
                        static_cast<void *>(&SMemElementsPerWarp)};

  cudaFuncSetAttribute(
      reinterpret_cast<void *>(SetLevelDynamicIncremental<DynGraphContext>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemorySize);

  cudaEvent_t Start, Stop;
  cudaStream_t DefaultStream;
  cudaEventCreate(&Start);
  cudaEventCreate(&Stop);
  cudaStreamCreate(&DefaultStream);
  cudaEventRecord(Start, DefaultStream);

  cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(SetLevelDynamicIncremental<DynGraphContext>),
      GridDim, BlockDim, KernelArgs, SMem, DefaultStream);

  cudaEventRecord(Stop, DefaultStream);
  cudaEventSynchronize(Stop);
  cudaStreamDestroy(DefaultStream);

  cudaEventElapsedTime(&ElapsedTime, Start, Stop);

  CHECK_ERROR(cudaFree(FrontierSrcDev));
  CHECK_ERROR(cudaFree(FrontierDstDev));
  CHECK_ERROR(cudaFree(FrontierSizeDev));

  CHECK_ERROR(cudaFree(NextFrontierSrcDev));
  CHECK_ERROR(cudaFree(NextFrontierDstDev));
  CHECK_ERROR(cudaFree(NextFrontierSizeDev));

  CHECK_ERROR(cudaFree(Queued));

  return ElapsedTime;
}

template <typename AllocPolicy>
float BFS<AllocPolicy>::Decremental(uint32_t *EdgesSrcDev,
                                    uint32_t *EdgesDstDev, uint32_t EdgesN) {
  uint32_t *Queued;
  uint32_t *FrontierSrcDev;
  uint32_t *FrontierDstDev;
  uint32_t *FrontierSizeDev;

  uint32_t *NextFrontierSrcDev;
  uint32_t *NextFrontierDstDev;
  uint32_t *NextFrontierSizeDev;

  float ElapsedTime;

  CHECK_ERROR(cudaMalloc(&Queued, sizeof(uint32_t) * VertexN));

  CHECK_ERROR(cudaMalloc(&FrontierSrcDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&FrontierDstDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&FrontierSizeDev, sizeof(uint32_t)));

  CHECK_ERROR(
      cudaMalloc(&NextFrontierSrcDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(
      cudaMalloc(&NextFrontierDstDev, sizeof(uint32_t) * EdgesN * 4 / 3));
  CHECK_ERROR(cudaMalloc(&NextFrontierSizeDev, sizeof(uint32_t)));

  CHECK_ERROR(cudaMemset(FrontierSizeDev, 0x00, sizeof(uint32_t)));
  CHECK_ERROR(cudaMemset(NextFrontierSizeDev, 0x00, sizeof(uint32_t)));

  CHECK_ERROR(cudaMemset(Queued, 0x00, sizeof(uint32_t) * VertexN));

  DynGraphContext Ctxt = G.GetDynamicGraphContext();

  uint32_t BlockSize = 512;
  uint32_t GridSize = SMCount;
  dim3 BlockDim(BlockSize, 1, 1);
  dim3 GridDim(GridSize, 1, 1);
  uint32_t SMem = (1 << 15);

  uint32_t WarpsN = BlockSize / WarpSize;

  uint32_t SMemPerWarp = SMem / WarpsN;
  uint32_t SMemElementsPerWarp = SMemPerWarp / sizeof(int);

  void *KernelArgs[] = {static_cast<void *>(&Ctxt),
                        static_cast<void *>(&SourceVertex),
                        static_cast<void *>(&VertexN),
                        static_cast<void *>(&EdgesSrcDev),
                        static_cast<void *>(&EdgesDstDev),
                        static_cast<void *>(&EdgesN),
                        static_cast<void *>(&FrontierSrcDev),
                        static_cast<void *>(&FrontierDstDev),
                        static_cast<void *>(&NextFrontierSrcDev),
                        static_cast<void *>(&NextFrontierDstDev),
                        static_cast<void *>(&NextFrontierSizeDev),
                        static_cast<void *>(&DistanceFromSrc),
                        static_cast<void *>(&Queued),
                        static_cast<void *>(&SMemElementsPerWarp)};

  cudaFuncSetAttribute(
      reinterpret_cast<void *>(SetLevelDynamicDecremental<DynGraphContext>),
      cudaFuncAttributeMaxDynamicSharedMemorySize, SharedMemorySize);

  cudaEvent_t Start, Stop;
  cudaStream_t DefaultStream;
  cudaEventCreate(&Start);
  cudaEventCreate(&Stop);
  cudaStreamCreate(&DefaultStream);
  cudaEventRecord(Start, DefaultStream);

  cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(SetLevelDynamicDecremental<DynGraphContext>),
      GridDim, BlockDim, KernelArgs, SMem, DefaultStream);

  cudaEventRecord(Stop, DefaultStream);
  cudaEventSynchronize(Stop);
  cudaStreamDestroy(DefaultStream);

  cudaEventElapsedTime(&ElapsedTime, Start, Stop);

  CHECK_ERROR(cudaFree(FrontierSrcDev));
  CHECK_ERROR(cudaFree(FrontierDstDev));
  CHECK_ERROR(cudaFree(FrontierSizeDev));

  CHECK_ERROR(cudaFree(NextFrontierSrcDev));
  CHECK_ERROR(cudaFree(NextFrontierDstDev));
  CHECK_ERROR(cudaFree(NextFrontierSizeDev));

  CHECK_ERROR(cudaFree(Queued));

  return ElapsedTime;
}