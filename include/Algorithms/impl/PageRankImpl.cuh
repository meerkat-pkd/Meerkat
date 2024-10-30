#pragma once

template <typename GraphContextT>
__global__ static void
ComputePageRank(GraphContextT G, uint32_t VertexN, float *Contribution,
                float *NewPageRankValues, float DampingFactor) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      G.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= VertexN)
    return;

  bool ToCompute = false;

  if (ThreadID < VertexN)
    ToCompute = true;

  float Value = 0;
  float PageRankValue = 0;
  uint32_t WorkQueue = 0;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToCompute)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t U = ThreadID - LaneID + Lane;

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::Iterator UIter = (VertexAdjacencies[U]).Begin();
    typename AdjacencyContext::Iterator UEnd = (VertexAdjacencies[U]).End();
    Value = 0;
    while (UIter != UEnd) {
      uint32_t V = *UIter.GetPointer(LaneID);
      bool HasV = ((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK) &&
                  ((V != TOMBSTONE_KEY) && (V != EMPTY_KEY));

      Value += (HasV ? (Contribution[V]) : 0.0f);
      ++UIter;
      __syncwarp();
    }

    for (uint32_t i = 1; i < 32; i *= 2)
      Value += __shfl_xor_sync(0xffffffff, Value, i);

    Value = __shfl_sync(0xFFFFFFFF, Value, 0, 32);

    if (ThreadID == U) {
      PageRankValue = ((1 - DampingFactor) / VertexN) + (DampingFactor * Value);
    }

    if (Lane == LaneID)
      ToCompute = false;
  }

  if (ThreadID < VertexN) {
    NewPageRankValues[ThreadID] = PageRankValue;
  }
}

template <typename GraphContextT>
__global__ static void
CheckPresent(GraphContextT G, uint32_t *Src, uint32_t *Dst, uint32_t EdgeN,
             uint32_t *VertexOutDegree, bool IsIncremental) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;
  int C = IsIncremental ? 1 : -1;

  if ((ThreadID - LaneID) >= EdgeN)
    return;

  bool ToSearch = (ThreadID < EdgeN);
  uint32_t EdgeSrc = 0;
  uint32_t EdgeDst = 0;

  if (ToSearch) {
    EdgeSrc = Src[ThreadID];
    EdgeDst = Dst[ThreadID];
  }

  bool Present = G.SearchEdge(ToSearch, LaneID, EdgeDst, EdgeSrc);
  if (ToSearch && !Present)
    atomicAdd(&VertexOutDegree[EdgeSrc], C);
}

__global__ static void Accumulate(uint32_t *OutDegrees, float *OldPageRanks,
                                  uint32_t VertexN, float *Accumulator) {
  uint32_t Stride = blockDim.x * gridDim.x;
  uint32_t LaneID = threadIdx.x & 0x1F;
  float LocalValue = 0.0f;
  for (uint32_t I = blockIdx.x * blockDim.x + threadIdx.x; I < VertexN;
       I += Stride) {
    uint32_t OutDegree = OutDegrees[I];
    if (OutDegree == 0) {
      LocalValue += OldPageRanks[I] / VertexN;
    }
  }

  for (uint32_t i = 1; i < 32; i *= 2)
    LocalValue += __shfl_xor_sync(0xffffffff, LocalValue, i);

  if (LaneID == 0 && LocalValue != 0.0f)
    atomicAdd(Accumulator, LocalValue);
}

__global__ static void UpdatePageRanks(float *PageRanks, float Value,
                                       uint32_t N) {
  uint32_t Stride = blockDim.x * gridDim.x;
  for (uint32_t I = blockIdx.x * blockDim.x + threadIdx.x; I < N; I += Stride) {
    PageRanks[I] += Value;
  }
}

__global__ static void FindDelta(float *OldPageRanks, float *NewPageRanks,
                                 uint32_t N, float *Delta) {
  uint32_t Stride = blockDim.x * gridDim.x;
  uint32_t LaneID = threadIdx.x & 0x1F;
  float LocalDelta = 0.0f;

  for (uint32_t I = blockIdx.x * blockDim.x + threadIdx.x; I < N; I += Stride)
    LocalDelta += fabs(NewPageRanks[I] - OldPageRanks[I]);

  for (uint32_t i = 1; i < 32; i *= 2)
    LocalDelta += __shfl_xor_sync(0xffffffff, LocalDelta, i);

  if (LocalDelta != 0.0f && LaneID == 0)
    atomicAdd(Delta, LocalDelta);
}

__global__ static void FindContributionPerVertex(float *OldPageRankValues,
                                                 uint32_t *VertexOutDegree,
                                                 float *Contribution,
                                                 uint32_t VertexN) {
  uint32_t Stride = blockDim.x * gridDim.x;
  for (uint32_t I = blockIdx.x * blockDim.x + threadIdx.x; I < VertexN;
       I += Stride) {
    Contribution[I] = ((VertexOutDegree[I] == 0)
                           ? 0.0f
                           : (OldPageRankValues[I] / VertexOutDegree[I]));
  }
}

template <typename AllocPolicy>
float PageRank<AllocPolicy>::PageRankComputation() {
  float Accumulator = 0.0f, Delta = 1.0f, ElapsedTime = 0.0f;
  uint32_t Iterations = 0;
  float *AccumulatorDev, *DeltaDev;
  cudaEvent_t Start, Stop;

  float *NewPageRankValues;

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;

  CHECK_ERROR(cudaMalloc(&AccumulatorDev, sizeof(float)));
  CHECK_ERROR(cudaMalloc(&DeltaDev, sizeof(float)));
  CHECK_ERROR(cudaMalloc(&NewPageRankValues, sizeof(float) * VertexN));

  thrust::fill(thrust::device, NewPageRankValues, NewPageRankValues + VertexN,
               0);

  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  bool ShouldAccumulate = thrust::find(thrust::device, VertexOutDegreeDev,
                                       VertexOutDegreeDev + VertexN,
                                       0) != (VertexOutDegreeDev + VertexN);

  while (Delta > ErrorMargin && Iterations < MaxIter) {
    ++Iterations;

    FindContributionPerVertex<<<256, 256>>>(PageRankValues, VertexOutDegreeDev,
                                            ContributionPerVertex, VertexN);
    cudaDeviceSynchronize();

    ComputePageRank<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
        G.GetDynamicGraphContext(), VertexN, ContributionPerVertex,
        NewPageRankValues, DampingFactor);

    if (ShouldAccumulate) {
      CHECK_ERROR(cudaMemset(AccumulatorDev, 0x00, sizeof(float)));
      Accumulate<<<256, 256>>>(VertexOutDegreeDev, PageRankValues, VertexN,
                               AccumulatorDev);

      cudaDeviceSynchronize();

      CHECK_ERROR(cudaMemcpy(&Accumulator, AccumulatorDev, sizeof(float),
                             cudaMemcpyDeviceToHost));
      UpdatePageRanks<<<256, 256>>>(NewPageRankValues,
                                    Accumulator * DampingFactor, VertexN);
    }
    cudaDeviceSynchronize();

    CHECK_ERROR(cudaMemset(DeltaDev, 0x00, sizeof(float)));
    FindDelta<<<128, 128>>>(PageRankValues, NewPageRankValues, VertexN,
                            DeltaDev);
    cudaDeviceSynchronize();
    CHECK_ERROR(
        cudaMemcpy(&Delta, DeltaDev, sizeof(float), cudaMemcpyDeviceToHost));

    std::swap(PageRankValues, NewPageRankValues);
  }

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);

  CHECK_ERROR(cudaFree(NewPageRankValues));

  cudaEventElapsedTime(&ElapsedTime, Start, Stop);
  return ElapsedTime;
}

template <typename AllocPolicy> float PageRank<AllocPolicy>::Static() {
  return PageRankComputation();
}

template <typename AllocPolicy>
float PageRank<AllocPolicy>::DynamicPageRankComputation(uint32_t *EdgesSrcDev,
                                                        uint32_t *EdgesDstDev,
                                                        uint32_t EdgesN,
                                                        bool IsIncremental) {
  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (EdgesN + ThreadBlockSize - 1) / ThreadBlockSize;

  CheckPresent<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      G.GetDynamicGraphContext(), EdgesSrcDev, EdgesDstDev,
      EdgesN, VertexOutDegreeDev, IsIncremental);
  cudaDeviceSynchronize();

  return PageRankComputation();
}

template <typename AllocPolicy>
float PageRank<AllocPolicy>::Incremental(uint32_t *EdgesSrcDev,
                                         uint32_t *EdgesDstDev,
                                         uint32_t EdgesN) {
  return DynamicPageRankComputation(EdgesSrcDev, EdgesDstDev, EdgesN, true);
}

template <typename AllocPolicy>
float PageRank<AllocPolicy>::Decremental(uint32_t *EdgesSrcDev,
                                         uint32_t *EdgesDstDev,
                                         uint32_t EdgesN) {
  return DynamicPageRankComputation(EdgesSrcDev, EdgesDstDev, EdgesN, false);
}