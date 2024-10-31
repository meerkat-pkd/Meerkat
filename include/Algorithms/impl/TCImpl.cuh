#pragma once

#include <algorithm>
#include <execution>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

template <typename T> __device__ __inline__ void swap(T &First, T &Second) {
  T Temp = First;
  First = Second;
  Second = Temp;
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
CollectEdgesCore(cg::thread_block_tile<32> &Warp, VertexAdjacencies *VA,
                 bool &ToConsider, uint32_t CurrentSrc, uint32_t *SrcPtr,
                 uint32_t *DstPtr, uint32_t *EdgesN, uint32_t *WarpSMQueueSrc,
                 uint32_t *WarpSMQueueDst, uint32_t &WarpSMemQueueLen,
                 uint32_t MaxEdgesInSMem) {
  using AdjacencyContext = typename DynGraphContext::EdgeHashContext;
  using ContainerPolicy = typename DynGraphContext::EdgeContainerPolicy;
  using SlabInfoT = typename ContainerPolicy::SlabInfoT;

  uint32_t WorkQueue = 0;
  while ((WorkQueue = Warp.ballot(ToConsider)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = Warp.shfl(CurrentSrc, Lane);

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
          Warp, HasAdjacentVertex, CurrentVertex, AdjacentVertex, SrcPtr,
          DstPtr, EdgesN, WarpSMQueueSrc, WarpSMQueueDst, WarpSMemQueueLen,
          MaxEdgesInSMem);
      ++First;
    }

    if (Lane == Warp.thread_rank())
      ToConsider = false;
  }
}

template <typename DynGraphContext>
__global__ void CollectEdges(DynGraphContext TheGraphContext, uint32_t VertexN,
                             uint32_t *Src, uint32_t *Dst, uint32_t *EdgesN,
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
  uint32_t LoopLimit = ((VertexN + WARP_SIZE - 1) >> 5) << 5;

  uint32_t WarpSMemQueueLen = 0;

  for (uint32_t I = ThreadId; I < LoopLimit; I += Stride) {
    bool ToConsider = I < VertexN;
    uint32_t CurrentSrc = ToConsider ? I : UINT32_MAX;
    CollectEdgesCore<DynGraphContext>(
        Warp, VertexAdjacencies, ToConsider, CurrentSrc, Src, Dst, EdgesN,
        WarpSMQueueSrc, WarpSMQueueDst, WarpSMemQueueLen, MaxEdgesInSMem);
  }

  FlushSharedMemPartition(Warp, Src, Dst, EdgesN, WarpSMQueueSrc,
                          WarpSMQueueDst, WarpSMemQueueLen);
  Grid.sync();
}

template <typename GraphContextT>
__global__ void TriangleCount_1(GraphContextT G, uint32_t *Src, uint32_t *Dst,
                                uint32_t EdgesN,
                                unsigned long long int *TotalCount) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  using EdgeHashCtxt = typename GraphContextT::EdgeHashContext;
  EdgeHashCtxt *VertexAdjacencies = G.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= EdgesN)
    return;

  bool ToSearch = (ThreadID < EdgesN);
  unsigned long long int Count = 0;
  uint32_t WorkQueue = 0;
  uint32_t U, V;
  uint32_t *Degrees = G.GetVertexDegrees();

  if (ToSearch) {
    U = Src[ThreadID];
    V = Dst[ThreadID];
    uint32_t DegreeSrc = Degrees[U];
    uint32_t DegreeDst = Degrees[V];
    if (DegreeSrc > DegreeDst)
      swap(U, V);
  }

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;
    using Iterator = typename AdjacencyContext::Iterator;

    uint32_t TheU = __shfl_sync(0xFFFFFFFF, U, Lane, 32);
    uint32_t TheV = __shfl_sync(0xFFFFFFFF, V, Lane, 32);
    Iterator UIter = VertexAdjacencies[TheU].Begin();
    Iterator ULast = VertexAdjacencies[TheU].End();

    while (UIter != ULast) {
      uint32_t AdjacentVertex = *UIter.GetPointer(LaneID);
      bool HasAdjacentVertex =
          ((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK) &&
          (AdjacentVertex != TOMBSTONE_KEY) && (AdjacentVertex != EMPTY_KEY) &&
          (AdjacentVertex != TheV);

      bool EdgePresent =
          G.SearchEdge(HasAdjacentVertex, LaneID, AdjacentVertex, TheV);
      if (EdgePresent)
        ++Count;

      ++UIter;
    }

    if (Lane == LaneID)
      ToSearch = false;
  }

  for (uint32_t I = 1; I < 32; I = (I << 1))
    Count += __shfl_xor_sync(0xFFFFFFFF, Count, I);

  if (LaneID == 0 && Count)
    atomicAdd(TotalCount, Count);
}

template <typename GraphContextT>
__global__ void
TriangleCount_2(GraphContextT G1, GraphContextT G2, uint32_t *Src,
                uint32_t *Dst, uint32_t *MappedSrc, uint32_t *MappedDst,
                uint32_t EdgesN, unsigned long long int *TotalCount,
                uint32_t *Map) {

  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  using EdgeHashCtxt = typename GraphContextT::EdgeHashContext;
  EdgeHashCtxt *VertexAdjacencies = G2.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= EdgesN)
    return;

  bool ToSearch = (ThreadID < EdgesN);
  unsigned long long int Count = 0;
  uint32_t WorkQueue = 0;
  uint32_t U, V;
  uint32_t MappedU, MappedV;
  uint32_t *G2Degrees = G2.GetVertexDegrees();

  if (ToSearch) {
    U = Src[ThreadID];
    V = Dst[ThreadID];
    MappedU = MappedSrc[ThreadID];
    MappedV = MappedDst[ThreadID];
  }

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t TheMappedU = __shfl_sync(0xFFFFFFFF, MappedU, Lane, 32);
    uint32_t TheMappedV = __shfl_sync(0xFFFFFFFF, MappedV, Lane, 32);
    uint32_t TheU = __shfl_sync(0xFFFFFFFF, U, Lane, 32);
    uint32_t TheV = __shfl_sync(0xFFFFFFFF, V, Lane, 32);

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;
    using Iterator = typename AdjacencyContext::Iterator;

    Iterator UIter = VertexAdjacencies[TheMappedU].Begin();
    Iterator UEnd = VertexAdjacencies[TheMappedU].End();

    while (UIter != UEnd) {
      uint32_t AdjacentVertex = *UIter.GetPointer(LaneID);
      bool HasAdjacentVertex =
          ((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK) &&
          (AdjacentVertex != TheMappedV) && (AdjacentVertex != TOMBSTONE_KEY) &&
          (AdjacentVertex != EMPTY_KEY);
      uint32_t X = HasAdjacentVertex ? Map[AdjacentVertex]
                                     : static_cast<uint32_t>(INVALID_VERTEX);
      bool EdgePresent = G1.SearchEdge(HasAdjacentVertex, LaneID, TheV, X);
      if (HasAdjacentVertex && EdgePresent) {
        ++Count;
      }

      ++UIter;
    }

    if (Lane == LaneID)
      ToSearch = false;
  }

  for (uint32_t I = 1; I < 32; I = (I << 1))
    Count += __shfl_xor_sync(0xFFFFFFFF, Count, I);

  if (LaneID == 0 && Count)
    atomicAdd(TotalCount, Count);
}

template <typename AllocPolicy> float TC<AllocPolicy>::Static(uint32_t EdgesN) {
  unsigned long long int *TriangleCountDev;
  uint32_t *Src, *Dst;
  uint32_t *EdgesNDev;
  float ElapsedTime[2];

  cudaDeviceProp Properties;
  cudaError_t ErrorStatus = cudaGetDeviceProperties(&Properties, 0);

  if (ErrorStatus != cudaSuccess)
    throw std::runtime_error("cudaGetDeviceProperties() Failed !!");

  uint32_t SMCount = Properties.multiProcessorCount;
  uint32_t WarpSize = Properties.warpSize;
  uint32_t SharedMemorySize = Properties.sharedMemPerBlock;
  uint32_t SMCoreCount =
      ConvertSMVersionToCores(Properties.major, Properties.minor);

  uint32_t BlockSize = 512;
  uint32_t GridSize = SMCount;
  uint32_t WarpsN = BlockSize / WarpSize;
  dim3 BlockDim(BlockSize, 1, 1);
  dim3 GridDim(GridSize, 1, 1);
  uint32_t SMem = (1 << 15);

  uint32_t SMemPerWarp = SMem / WarpsN;
  uint32_t SMemElementsPerWarp = SMemPerWarp / sizeof(int);

  CHECK_ERROR(cudaMalloc(&Src, sizeof(uint32_t) * EdgesN));
  CHECK_ERROR(cudaMalloc(&Dst, sizeof(uint32_t) * EdgesN));
  CHECK_ERROR(cudaMalloc(&TriangleCountDev, sizeof(unsigned long long int)));
  CHECK_ERROR(cudaMemset(TriangleCountDev, 0, sizeof(unsigned long long int)));

  DynGraphContext Ctxt = G.GetDynamicGraphContext();

  void *KernelArgs[] = {static_cast<void *>(&Ctxt),
                        static_cast<void *>(&VertexN),
                        static_cast<void *>(&Src),
                        static_cast<void *>(&Dst),
                        static_cast<void *>(&EdgesNDev),
                        static_cast<void *>(&SMemElementsPerWarp)};

  cudaFuncSetAttribute(reinterpret_cast<void *>(CollectEdges<DynGraphContext>),
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       SharedMemorySize);

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (EdgesN + ThreadBlockSize - 1) / ThreadBlockSize;

  cudaEvent_t Start, Stop;
  cudaStream_t DefaultStream;
  cudaEventCreate(&Start);
  cudaEventCreate(&Stop);
  cudaStreamCreate(&DefaultStream);
  cudaEventRecord(Start, DefaultStream);

  cudaLaunchCooperativeKernel(
      reinterpret_cast<void *>(CollectEdges<DynGraphContext>), GridDim,
      BlockDim, KernelArgs, SMem, DefaultStream);

  cudaEventRecord(Stop, DefaultStream);
  cudaEventSynchronize(Stop);
  cudaStreamDestroy(DefaultStream);

  cudaEventElapsedTime(&ElapsedTime[0], Start, Stop);

  CHECK_ERROR(
      cudaMemcpy(&EdgesN, EdgesNDev, sizeof(uint32_t), cudaMemcpyDeviceToHost));

  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  TriangleCount_1<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      G.GetDynamicGraphContext(), Src, Dst, EdgesN, TriangleCountDev);
  cudaDeviceSynchronize();

  CHECK_ERROR(cudaMemcpy(&TriangleCount, TriangleCountDev,
                         sizeof(unsigned long long int),
                         cudaMemcpyDeviceToHost));
  TriangleCount = TriangleCount / 6;

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&ElapsedTime[1], Start, Stop);

  CHECK_ERROR(cudaFree(Src));
  CHECK_ERROR(cudaFree(Dst));
  CHECK_ERROR(cudaFree(TriangleCountDev));

  return ElapsedTime[0] + ElapsedTime[1];
}

template <typename AllocPolicy>
void TC<AllocPolicy>::ReverseMapping(uint32_t *Src, uint32_t *Dst,
                                     uint32_t Length,
                                     thrust::device_vector<uint32_t> &Mapping,
                                     uint32_t &MappingLength,
                                     uint32_t *MappedSrc, uint32_t *MappedDst) {
  thrust::copy(Src, Src + Length, std::begin(Mapping));
  thrust::copy(Dst, Dst + Length, std::begin(Mapping) + Length);
  thrust::sort(thrust::device, std::begin(Mapping), std::end(Mapping));
  auto Iter =
      thrust::unique(thrust::device, std::begin(Mapping), std::end(Mapping));
  MappingLength = static_cast<uint32_t>(Iter - std::begin(Mapping));

  thrust::host_vector<uint32_t> MappingHost(std::begin(Mapping),
                                            std::end(Mapping));

  uint32_t *SrcHost = new uint32_t[Length];
  uint32_t *DstHost = new uint32_t[Length];
  uint32_t *MappedSrcHost = new uint32_t[Length];
  uint32_t *MappedDstHost = new uint32_t[Length];

  CHECK_ERROR(cudaMemcpy(SrcHost, Src, sizeof(uint32_t) * Length,
                         cudaMemcpyDeviceToHost));
  CHECK_ERROR(cudaMemcpy(DstHost, Dst, sizeof(uint32_t) * Length,
                         cudaMemcpyDeviceToHost));

  std::transform(std::execution::par, SrcHost, SrcHost + Length, MappedSrcHost,
                 [&MappingHost](uint32_t V) {
                   return std::lower_bound(std::begin(MappingHost),
                                           std::end(MappingHost), V) - std::begin(MappingHost);
                 });

  std::transform(std::execution::par, DstHost, DstHost + Length, MappedDstHost,
                 [&MappingHost](uint32_t V) {
                   return std::lower_bound(std::begin(MappingHost),
                                           std::end(MappingHost), V) - std::begin(MappingHost);
                 });

  CHECK_ERROR(cudaMemcpy(MappedSrc, MappedSrcHost, sizeof(uint32_t) * Length,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(MappedDst, MappedDstHost, sizeof(uint32_t) * Length,
                         cudaMemcpyHostToDevice));

  delete[] SrcHost;
  delete[] DstHost;
  delete[] MappedSrcHost;
  delete[] MappedDstHost;
}

template <typename AllocPolicy>
float TC<AllocPolicy>::Incremental(
    uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev, uint32_t EdgeN,
    std::unique_ptr<typename AllocPolicy::DynamicAllocatorT> &Alloc) {
  thrust::device_vector<uint32_t> Mapping =
      thrust::device_vector<uint32_t>(EdgeN);
  uint32_t MappingLength = 0;
  uint32_t *MappedSrcDev, *MappedDstDev;
  float ElapsedTime = 0.0f;

  CHECK_ERROR(cudaMalloc(&MappedSrcDev, sizeof(uint32_t) * EdgeN));
  CHECK_ERROR(cudaMalloc(&MappedDstDev, sizeof(uint32_t) * EdgeN));

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (EdgeN + ThreadBlockSize - 1) / ThreadBlockSize;

  unsigned long long int Count[3];
  unsigned long long int *CountDev;
  cudaEvent_t Start, Stop;

  CHECK_ERROR(cudaMalloc(&CountDev, sizeof(unsigned long long int) * 3));
  CHECK_ERROR(cudaMemset(CountDev, 0x00, sizeof(unsigned long long int) * 3));

  std::vector<uint32_t> _hints(MappingLength, 1);
  float LoadFactor = 0.7;
  GraphT InsGraph{MappingLength, *Alloc, LoadFactor, _hints.data(), 0};

  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  ReverseMapping(EdgesSrcDev, EdgesDstDev, EdgeN, Mapping, MappingLength,
                 MappedSrcDev, MappedDstDev);
  InsGraph.InsertEdges(MappedSrcDev, MappedDstDev, EdgeN);
  cudaDeviceSynchronize();

  TriangleCount_1<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      G.GetDynamicGraphContext(), EdgesSrcDev, EdgesDstDev, EdgeN, CountDev);

  TriangleCount_2<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      G.GetDynamicGraphContext(), InsGraph.GetDynamicGraphContext(),
      EdgesSrcDev, EdgesDstDev, MappedSrcDev, MappedDstDev, EdgeN,
      (CountDev + 1), Mapping.data().get());

  TriangleCount_1<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      InsGraph.GetDynamicGraphContext(), MappedSrcDev, MappedDstDev, EdgeN,
      CountDev + 2);

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&ElapsedTime, Start, Stop);

  CHECK_ERROR(cudaMemcpy(&Count[0], CountDev,
                         sizeof(unsigned long long int) * 3,
                         cudaMemcpyDeviceToHost));

  unsigned long long int TrianglesAdded =
      ((Count[0]) - (Count[1]) + (Count[2]) / 3) / 2;
  TriangleCount += TrianglesAdded;

  CHECK_ERROR(cudaFree(MappedSrcDev));
  CHECK_ERROR(cudaFree(MappedDstDev));
  CHECK_ERROR(cudaFree(CountDev));

  return ElapsedTime;
}

template <typename AllocPolicy>
float TC<AllocPolicy>::Decremental(
    uint32_t *EdgesSrcDev, uint32_t *EdgesDstDev, uint32_t EdgeN,
    std::unique_ptr<typename AllocPolicy::DynamicAllocatorT> &Alloc) {
  thrust::device_vector<uint32_t> Mapping =
      thrust::device_vector<uint32_t>(EdgeN);
  uint32_t MappingLength = 0;
  uint32_t *MappedSrcDev, *MappedDstDev;
  float ElapsedTime = 0.0f;

  CHECK_ERROR(cudaMalloc(&MappedSrcDev, sizeof(uint32_t) * EdgeN));
  CHECK_ERROR(cudaMalloc(&MappedDstDev, sizeof(uint32_t) * EdgeN));

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (EdgeN + ThreadBlockSize - 1) / ThreadBlockSize;

  unsigned long long int Count[3];
  unsigned long long int *CountDev;
  cudaEvent_t Start, Stop;

  CHECK_ERROR(cudaMalloc(&CountDev, sizeof(unsigned long long int) * 3));
  CHECK_ERROR(cudaMemset(CountDev, 0x00, sizeof(unsigned long long int) * 3));

  std::vector<uint32_t> _hints(MappingLength, 1);
  float LoadFactor = 0.0f;
  GraphT DelGraph{MappingLength, *Alloc, LoadFactor, _hints.data(), 0};

  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  ReverseMapping(EdgesSrcDev, EdgesDstDev, EdgeN, Mapping, MappingLength,
                 MappedSrcDev, MappedDstDev);
  DelGraph.InsertEdges(MappedSrcDev, MappedDstDev, EdgeN);
  cudaDeviceSynchronize();

  TriangleCount_1<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      G.GetDynamicGraphContext(), EdgesSrcDev, EdgesDstDev, EdgeN, CountDev);

  TriangleCount_2<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      G.GetDynamicGraphContext(), DelGraph.GetDynamicGraphContext(),
      EdgesSrcDev, EdgesDstDev, MappedSrcDev, MappedDstDev, EdgeN,
      (CountDev + 1), Mapping.data().get());

  TriangleCount_1<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      DelGraph.GetDynamicGraphContext(), MappedSrcDev, MappedDstDev, EdgeN,
      CountDev + 2);

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);
  cudaEventElapsedTime(&ElapsedTime, Start, Stop);

  CHECK_ERROR(cudaMemcpy(&Count[0], CountDev,
                         sizeof(unsigned long long int) * 3,
                         cudaMemcpyDeviceToHost));

  unsigned long long int TrianglesRemoved =
      (Count[0] + Count[1] + Count[2] / 3) >> 1;
  TriangleCount -= TrianglesRemoved;

  CHECK_ERROR(cudaFree(MappedSrcDev));
  CHECK_ERROR(cudaFree(MappedDstDev));
  CHECK_ERROR(cudaFree(CountDev));

  return ElapsedTime;
}