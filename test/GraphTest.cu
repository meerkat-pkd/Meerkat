#include <algorithm>
#include <cstdint>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <gtest/gtest.h>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>

#include "Graph/SlabGraph.cuh"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

static constexpr uint32_t BLOCK_SIZE = 128;

using AllocPolicy = FullAllocatorPolicy<8, 32, 1>;
using Allocator = typename AllocPolicy::DynamicAllocatorT;
using EdgePolicy = UnweightedEdgePolicyT<AllocPolicy>;
using WEdgePolicy = WeightedEdgePolicyMapT<AllocPolicy>;

struct Pair {
  uint32_t x;
  uint32_t y;

  Pair(const Pair &) = default;
  Pair(uint32_t x, uint32_t y) : x(x), y(y) {}

  bool operator<(const Pair &RHS) const {
    return (x < RHS.x) || ((x == RHS.x) && (y < RHS.y));
  }

  bool operator==(const Pair &RHS) const {
    return (x == RHS.x) && (y == RHS.y);
  }

  bool operator!=(const Pair &RHS) const { return !(*this == RHS); }

  std::string DebugString() const {
    std::string TheDebugString;
    std::stringstream Stream(TheDebugString);
    Stream << "[Pair X = " << x << " Y = " << y << " ]";
    return TheDebugString;
  }

  friend void PrintTo(const Pair &ThePair, std::ostream *TheStream) {
    *TheStream << ThePair.DebugString();
  }
};

template <typename EdgePolicy>
using DynGraph = DynamicGraph<EdgePolicy, EdgePolicy::IsWeightedEdge>;

template <typename GraphContextT>
__global__ void CountTombstonesUnweighted(int NumberOfVertices,
                                          GraphContextT TheGraphContext,
                                          uint32_t *TombstoneCount) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;
  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= NumberOfVertices)
    return;

  int Vertex = 0xFFFFFFFF;
  bool ToSearch = false;

  if (ThreadID < NumberOfVertices) {
    Vertex = ThreadID;
    ToSearch = true;
  }

  uint32_t WorkQueue = 0;
  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = ThreadID - LaneID + Lane;

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::Iterator First =
        (VertexAdjacencies[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last =
        (VertexAdjacencies[CurrentVertex]).End();

    uint32_t LocalTombstoneCount = 0;
    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(LaneID);
#if 0
      bool HasAdjacentVertex = (LaneID < 31) &&
                               (AdjacentVertex != TOMBSTONE_KEY) &&
                               (AdjacentVertex != EMPTY_KEY);
#endif
      bool HasTombstone =
          (((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK)) &&
          (AdjacentVertex == TOMBSTONE_KEY);
      if (HasTombstone)
        ++LocalTombstoneCount;
      ++First;
    }

    LocalTombstoneCount +=
        __shfl_down_sync(0xFFFFFFFF, LocalTombstoneCount, 16);
    LocalTombstoneCount += __shfl_down_sync(0xFFFFFFFF, LocalTombstoneCount, 8);
    LocalTombstoneCount += __shfl_down_sync(0xFFFFFFFF, LocalTombstoneCount, 4);
    LocalTombstoneCount += __shfl_down_sync(0xFFFFFFFF, LocalTombstoneCount, 2);
    LocalTombstoneCount += __shfl_down_sync(0xFFFFFFFF, LocalTombstoneCount, 1);

    if (LaneID == 0)
      TombstoneCount[CurrentVertex] = LocalTombstoneCount;
    if (CurrentVertex == ThreadID)
      ToSearch = false;
  }
}

template <typename GraphContextT>
__global__ void CountDegreesUnweighted(int NumberOfVertices,
                                       GraphContextT TheGraphContext,
                                       uint32_t *DegreeCount) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;
  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= NumberOfVertices)
    return;

  int Vertex = 0xFFFFFFFF;
  bool ToSearch = false;

  if (ThreadID < NumberOfVertices) {
    Vertex = ThreadID;
    ToSearch = true;
  }

  uint32_t WorkQueue = 0;
  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = ThreadID - LaneID + Lane;

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::Iterator First =
        (VertexAdjacencies[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last =
        (VertexAdjacencies[CurrentVertex]).End();

    uint32_t LocalDegree = 0;
    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(LaneID);
#if 0
      bool HasAdjacentVertex = (LaneID < 31) &&
                               (AdjacentVertex != TOMBSTONE_KEY) &&
                               (AdjacentVertex != EMPTY_KEY);
#endif
      bool HasAdjacentVertex =
          (((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK)) &&
          (AdjacentVertex != TOMBSTONE_KEY) && (AdjacentVertex != EMPTY_KEY);
      if (HasAdjacentVertex)
        ++LocalDegree;
      ++First;
    }

    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 16);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 8);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 4);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 2);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 1);

    if (LaneID == 0)
      DegreeCount[CurrentVertex] = LocalDegree;
    if (CurrentVertex == ThreadID)
      ToSearch = false;
  }
}

template <typename GraphContextT>
__global__ void CountDegreesUnweightedWarp(int N, GraphContextT TheGraphContext,
                                           uint32_t *BucketVertex,
                                           uint32_t *BucketIndex,
                                           uint32_t *DegreeCount) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = (gridDim.x * blockDim.x);
  uint32_t WarpsN = (ThreadsN >> 5) + (ThreadsN & 0x1F);
  uint32_t LaneId = threadIdx.x & 0x1F;
  uint32_t GlobalWarpID = (ThreadId >> 5);

  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  for (uint32_t U = GlobalWarpID; U < N; U += WarpsN) {
    uint32_t Src = BucketVertex[U];
    uint32_t Index = BucketIndex[U];

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::BucketIterator First =
        (VertexAdjacencies[Src]).BeginAt(Index);
    typename AdjacencyContext::BucketIterator Last =
        (VertexAdjacencies[Src]).EndAt(Index);

    uint32_t LocalDegree = 0;
    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(LaneId);

      bool HasAdjacentVertex =
          (((1 << LaneId) & SlabInfoT::REGULAR_NODE_KEY_MASK)) &&
          (AdjacentVertex != TOMBSTONE_KEY) && (AdjacentVertex != EMPTY_KEY);
      if (HasAdjacentVertex) {
        ++LocalDegree;
      }
      ++First;
    }

    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 16);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 8);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 4);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 2);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 1);
    if (LaneId == 0)
      atomicAdd(&DegreeCount[Src], LocalDegree);
  
  }
}

template <typename GraphContextT>
__global__ void CountDegrees(int NumberOfVertices,
                             GraphContextT TheGraphContext,
                             uint32_t *DegreeCount) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;
  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= NumberOfVertices)
    return;

  int Vertex = 0xFFFFFFFF;
  bool ToSearch = false;

  if (ThreadID < NumberOfVertices) {
    Vertex = ThreadID;
    ToSearch = true;
  }

  uint32_t WorkQueue = 0;
  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = ThreadID - LaneID + Lane;

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::Iterator First =
        (VertexAdjacencies[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last =
        (VertexAdjacencies[CurrentVertex]).End();

    uint32_t LocalDegree = 0;
    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(LaneID);
      bool HasAdjacentVertex =
          (((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK)) &&
          (AdjacentVertex != TOMBSTONE_KEY) && (AdjacentVertex != EMPTY_KEY);
      if (HasAdjacentVertex)
        ++LocalDegree;
      ++First;
    }

    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 16);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 8);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 4);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 2);
    LocalDegree += __shfl_down_sync(0xFFFFFFFF, LocalDegree, 1);

    if (LaneID == 0)
      DegreeCount[CurrentVertex] = LocalDegree;
    if (CurrentVertex == ThreadID)
      ToSearch = false;
  }
}

template <typename GraphContextT>
__global__ void
CollectEdges(int NumberOfVertices, GraphContextT TheGraphContext,
             uint32_t *VertexOffsets, uint32_t *AdjacentVertices) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= NumberOfVertices)
    return;

  uint32_t Vertex = 0xFFFFFFFFu;
  bool ToSearch = false;

  if (ThreadID < NumberOfVertices) {
    Vertex = ThreadID;
    ToSearch = true;
  }

  uint32_t WorkQueue = 0;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = ThreadID - LaneID + Lane;

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::Iterator First =
        (VertexAdjacencies[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last =
        (VertexAdjacencies[CurrentVertex]).End();

    uint32_t AdjVertOffsetBase = VertexOffsets[CurrentVertex];

    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(LaneID);

      bool HasAdjacentVertex =
          (((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK)) &&
          (AdjacentVertex != TOMBSTONE_KEY) && (AdjacentVertex != EMPTY_KEY);

      uint32_t BitSet = __ballot_sync(0xFFFFFFFF, HasAdjacentVertex) &
                        SlabInfoT::REGULAR_NODE_KEY_MASK;

      AdjVertOffsetBase = __shfl_sync(0xFFFFFFFF, AdjVertOffsetBase, 0, 32);
      if (HasAdjacentVertex) {
        uint32_t Offset =
            __popc(__brev(BitSet) & (0xFFFFFFFFu << (32 - LaneID)));
        uint32_t AdjacentVertexOffset = AdjVertOffsetBase + Offset;
        AdjacentVertices[AdjacentVertexOffset] = AdjacentVertex;
      }

      if (LaneID == 0)
        AdjVertOffsetBase += __popc(BitSet);
      ++First;
    }

    if (CurrentVertex == ThreadID)
      ToSearch = false;
  }
}

template <typename GraphContextT>
__global__ void
CollectEdgesWarpScheduled(int N, GraphContextT TheGraphContext,
                          uint32_t *BucketVertex, uint32_t *BucketIndex,
                          uint32_t *VertexOffsets, uint32_t *AdjacentVertices) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = (gridDim.x * blockDim.x);
  uint32_t WarpsN = (ThreadsN >> 5) + (ThreadsN & 0x1F);
  uint32_t LaneId = threadIdx.x & 0x1F;
  uint32_t GlobalWarpID = (ThreadId >> 5);

  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  for (uint32_t U = GlobalWarpID; U < N; U += WarpsN) {
    uint32_t Src = BucketVertex[U];
    uint32_t Index = BucketIndex[U];

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::BucketIterator First =
        (VertexAdjacencies[Src]).BeginAt(Index);
    typename AdjacencyContext::BucketIterator Last =
        (VertexAdjacencies[Src]).EndAt(Index);

    uint32_t AdjVertOffsetBase = VertexOffsets[Src];

    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(LaneId);
      bool HasAdjacentVertex =
          (((1 << LaneId) & SlabInfoT::REGULAR_NODE_KEY_MASK)) &&
          (AdjacentVertex != TOMBSTONE_KEY) && (AdjacentVertex != EMPTY_KEY);

      uint32_t BitSet = __ballot_sync(0xFFFFFFFF, HasAdjacentVertex) &
                        SlabInfoT::REGULAR_NODE_KEY_MASK;

      if (LaneId == 0)
        AdjVertOffsetBase = atomicAdd(&VertexOffsets[Src], __popc(BitSet));

      AdjVertOffsetBase = __shfl_sync(0xFFFFFFFF, AdjVertOffsetBase, 0, 32);
      if (HasAdjacentVertex) {
        uint32_t Offset =
            __popc(__brev(BitSet) & (0xFFFFFFFFu << (32 - LaneId)));
        uint32_t AdjacentVertexOffset = AdjVertOffsetBase + Offset;
        AdjacentVertices[AdjacentVertexOffset] = AdjacentVertex;
      }

      ++First;
    }
  }
}

template <typename GraphContextT>
__global__ void CollectEdges(int NumberOfVertices,
                             GraphContextT TheGraphContext,
                             uint32_t *VertexOffsets,
                             uint32_t *AdjacentVertices, uint32_t *Weights) {
  uint32_t ThreadID = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  typename GraphContextT::EdgeHashContext *VertexAdjacencies =
      TheGraphContext.GetEdgeHashCtxts();

  if ((ThreadID - LaneID) >= NumberOfVertices)
    return;

  uint32_t Vertex = 0xFFFFFFFFu;
  bool ToSearch = false;

  if (ThreadID < NumberOfVertices) {
    Vertex = ThreadID;
    ToSearch = true;
  }

  uint32_t WorkQueue = 0;

  while ((WorkQueue = __ballot_sync(0xFFFFFFFF, ToSearch)) != 0) {
    int Lane = __ffs(WorkQueue) - 1;
    uint32_t CurrentVertex = ThreadID - LaneID + Lane;

    using AdjacencyContext = typename GraphContextT::EdgeHashContext;
    using ContainerPolicy = typename GraphContextT::EdgeContainerPolicy;
    using SlabInfoT = typename ContainerPolicy::SlabInfoT;

    typename AdjacencyContext::Iterator First =
        (VertexAdjacencies[CurrentVertex]).Begin();
    typename AdjacencyContext::Iterator Last =
        (VertexAdjacencies[CurrentVertex]).End();

    uint32_t AdjVertOffsetBase = VertexOffsets[CurrentVertex];

    while (First != Last) {
      uint32_t AdjacentVertex = *First.GetPointer(LaneID);
      uint32_t Weight = __shfl_down_sync(0xFFFFFFFF, AdjacentVertex, 1, 32);

      bool HasAdjacentVertex =
          (((1 << LaneID) & SlabInfoT::REGULAR_NODE_KEY_MASK)) &&
          (AdjacentVertex != TOMBSTONE_KEY) && (AdjacentVertex != EMPTY_KEY);
      uint32_t BitSet = __ballot_sync(0xFFFFFFFF, HasAdjacentVertex) &
                        SlabInfoT::REGULAR_NODE_KEY_MASK;

      AdjVertOffsetBase = __shfl_sync(0xFFFFFFFF, AdjVertOffsetBase, 0, 32);
      if (HasAdjacentVertex) {
        uint32_t Offset =
            __popc(__brev(BitSet) & (0xFFFFFFFFu << (32 - LaneID)));
        uint32_t AdjacentVertexOffset = AdjVertOffsetBase + Offset;
        AdjacentVertices[AdjacentVertexOffset] = AdjacentVertex;
        Weights[AdjacentVertexOffset] = Weight;
      }

      if (LaneID == 0)
        AdjVertOffsetBase += __popc(BitSet);
      ++First;
    }

    if (CurrentVertex == ThreadID)
      ToSearch = false;
  }
}

__global__ void MakeBucketIndexPairs(uint32_t VertexN,
                                     uint32_t *BucketsPerVertex,
                                     uint32_t *BucketOffset,
                                     uint32_t *BucketOffsetToVertex,
                                     uint32_t *BucketOffsetToIndex) {
  uint32_t ThreadId = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t ThreadsN = gridDim.x * blockDim.x;

  for (uint32_t I = ThreadId; I < VertexN; I += ThreadsN) {
    uint32_t Offset = BucketOffset[I];
    uint32_t BucketsN = BucketsPerVertex[I];
    for (uint32_t Index = 0; Index < BucketsN; ++Index) {
      BucketOffsetToVertex[Offset + Index] = I;
      BucketOffsetToIndex[Offset + Index] = Index;
    }
  }
}

struct GraphTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (DegreeCounts == nullptr) {
      std::random_device RandomDevice;
      std::mt19937 MersenneTwister(RandomDevice());
      std::uniform_real_distribution<double> Distribution(0.0, 1.0);

      std::ifstream InputFile("simple");
      InputFile >> VertexN >> EdgeN;

      DegreeCounts = new std::vector<uint32_t>(VertexN);
      std::fill(DegreeCounts->begin(), DegreeCounts->end(), 0);
      DegreeCountsAfterDel = new std::vector<uint32_t>(VertexN);
      std::fill(DegreeCountsAfterDel->begin(), DegreeCountsAfterDel->end(), 0);

      Src = new std::vector<uint32_t>(EdgeN);
      Dst = new std::vector<uint32_t>(EdgeN);
      DelSrc = new std::vector<uint32_t>();
      DelDst = new std::vector<uint32_t>();

      Adjacencies = new std::map<uint32_t, std::set<uint32_t>>();
      AdjacenciesAfterDelete = new std::map<uint32_t, std::set<uint32_t>>();

      CHECK_CUDA_ERROR(cudaMalloc(&SrcDevPtr, sizeof(uint32_t) * EdgeN));
      CHECK_CUDA_ERROR(cudaMalloc(&DstDevPtr, sizeof(uint32_t) * EdgeN));

      uint32_t U, V;
      for (int i = 0; i < EdgeN; ++i) {
        InputFile >> U >> V;
        (*Src)[i] = U;
        (*Dst)[i] = V;
        ++((*DegreeCounts)[U]);
        (*Adjacencies)[U].insert(V);

        double RandomReal = Distribution(MersenneTwister);
        if (RandomReal < 0.45) {
          /* Mark about 45% of the edges for deletion */
          DelSrc->emplace_back(U);
          DelDst->emplace_back(V);
          ++DelEdgeN;
        } else {
          (*AdjacenciesAfterDelete)[U].insert(V);
          ++((*DegreeCountsAfterDel)[U]);
        }
      }
      InputFile.close();

      CHECK_CUDA_ERROR(cudaMemcpy(SrcDevPtr, Src->data(),
                                  sizeof(uint32_t) * EdgeN,
                                  cudaMemcpyHostToDevice));
      CHECK_CUDA_ERROR(cudaMemcpy(DstDevPtr, Dst->data(),
                                  sizeof(uint32_t) * EdgeN,
                                  cudaMemcpyHostToDevice));
      TheAllocator = new Allocator;
      Graph = new DynGraph<EdgePolicy>(VertexN, *TheAllocator, 0.7,
                                       DegreeCounts->data(), 0);
      Graph->InsertEdges(SrcDevPtr, DstDevPtr, EdgeN);
      cudaDeviceSynchronize();
    }

    VertexBucketOffsets = Graph->GetVertexBucketsOffsets();
    BucketsPerVertex = Graph->GetBucketsPerVertex();

    CHECK_CUDA_ERROR(
        cudaMalloc(&VertexBucketOffsetsDev, sizeof(uint32_t) * VertexN));
    CHECK_CUDA_ERROR(
        cudaMalloc(&BucketsPerVertexDev, sizeof(uint32_t) * VertexN));
    CHECK_CUDA_ERROR(cudaMemcpy(VertexBucketOffsetsDev, VertexBucketOffsets,
                                sizeof(uint32_t) * VertexN,
                                cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(BucketsPerVertexDev, BucketsPerVertex,
                                sizeof(uint32_t) * VertexN,
                                cudaMemcpyHostToDevice));

    uint32_t N =
        VertexBucketOffsets[VertexN - 1] + BucketsPerVertex[VertexN - 1];
    CHECK_CUDA_ERROR(cudaMalloc(&V, sizeof(uint32_t) * N));
    CHECK_CUDA_ERROR(cudaMalloc(&Index, sizeof(uint32_t) * N));

    MakeBucketIndexPairs<<<1024, 1024>>>(VertexN, BucketsPerVertexDev,
                                         VertexBucketOffsetsDev, V, Index);
    TotalBuckets = N;
    cudaDeviceSynchronize();
  }

  static void TearDownTestSuite() {
#if 0
    delete DegreeCounts;
    delete Src;
    delete Dst;
    delete Adjacencies;
    delete AdjacenciesAfterDelete;

    CHECK_CUDA_ERROR(cudaFree(SrcDevPtr));
    CHECK_CUDA_ERROR(cudaFree(DstDevPtr));

    delete TheAllocator;
    delete Graph;
#endif
  }

  static std::vector<uint32_t> *DegreeCounts;
  static std::vector<uint32_t> *DegreeCountsAfterDel;
  static std::vector<uint32_t> *Src;
  static std::vector<uint32_t> *Dst;
  static std::vector<uint32_t> *DelSrc;
  static std::vector<uint32_t> *DelDst;

  static uint32_t VertexN;
  static uint32_t EdgeN;
  static uint32_t DelEdgeN;

  static std::map<uint32_t, std::set<uint32_t>> *Adjacencies;
  static std::map<uint32_t, std::set<uint32_t>> *AdjacenciesAfterDelete;

  static uint32_t *SrcDevPtr;
  static uint32_t *DstDevPtr;

  static Allocator *TheAllocator;
  static DynGraph<EdgePolicy> *Graph;

  static uint32_t *VertexBucketOffsets;
  static uint32_t *BucketsPerVertex;
  static uint32_t *VertexBucketOffsetsDev;
  static uint32_t *BucketsPerVertexDev;

  static uint32_t *V;
  static uint32_t *Index;
  static uint32_t TotalBuckets;
};

std::vector<uint32_t> *GraphTest::DegreeCounts = nullptr;
std::vector<uint32_t> *GraphTest::DegreeCountsAfterDel = nullptr;
std::vector<uint32_t> *GraphTest::Src = nullptr;
std::vector<uint32_t> *GraphTest::Dst = nullptr;
std::vector<uint32_t> *GraphTest::DelSrc = nullptr;
std::vector<uint32_t> *GraphTest::DelDst = nullptr;

uint32_t GraphTest::VertexN = 0;
uint32_t GraphTest::EdgeN = 0;
uint32_t GraphTest::DelEdgeN = 0;

std::map<uint32_t, std::set<uint32_t>> *GraphTest::Adjacencies = nullptr;
std::map<uint32_t, std::set<uint32_t>> *GraphTest::AdjacenciesAfterDelete =
    nullptr;

uint32_t *GraphTest::SrcDevPtr = nullptr;
uint32_t *GraphTest::DstDevPtr = nullptr;

Allocator *GraphTest::TheAllocator = nullptr;
DynGraph<EdgePolicy> *GraphTest::Graph = nullptr;

uint32_t *GraphTest::VertexBucketOffsets;
uint32_t *GraphTest::BucketsPerVertex;
uint32_t *GraphTest::VertexBucketOffsetsDev;
uint32_t *GraphTest::BucketsPerVertexDev;

uint32_t *GraphTest::V;
uint32_t *GraphTest::Index;
uint32_t GraphTest::TotalBuckets;

TEST_F(GraphTest, DegreeCountTest) {
  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;

  uint32_t *DegreeCountsDev = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&DegreeCountsDev, sizeof(uint32_t) * VertexN));
  CHECK_CUDA_ERROR(
      cudaMemset(DegreeCountsDev, 0x00, sizeof(uint32_t) * VertexN));

  cudaEvent_t Start, Stop;
  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  CountDegreesUnweighted<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      VertexN, Graph->GetDynamicGraphContext(), DegreeCountsDev);

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);

  float ElapsedTime = 0.0f;
  cudaEventElapsedTime(&ElapsedTime, Start, Stop);
  std::printf("Elapsed time : %f ms\n", ElapsedTime);

  std::vector<uint32_t> Count(VertexN);
  std::fill(Count.begin(), Count.end(), 0);
  CHECK_CUDA_ERROR(cudaMemcpy(Count.data(), DegreeCountsDev,
                              sizeof(uint32_t) * VertexN,
                              cudaMemcpyDeviceToHost));

  for (uint32_t I = 0; I < VertexN; ++I)
    EXPECT_EQ(Count[I], (*DegreeCounts)[I]) << "Failed for Vertex " << I;
  CHECK_CUDA_ERROR(cudaFree(DegreeCountsDev));
};

TEST_F(GraphTest, DegreeCountTestWarpScheduled) {
  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;
  uint32_t *DegreeCountsDev = nullptr;

  CHECK_CUDA_ERROR(cudaMalloc(&DegreeCountsDev, sizeof(uint32_t) * VertexN));
  CHECK_CUDA_ERROR(
      cudaMemset(DegreeCountsDev, 0x00, sizeof(uint32_t) * VertexN));

  cudaEvent_t Start, Stop;
  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  CountDegreesUnweightedWarp<<<1024, 1024>>>(
      TotalBuckets, Graph->GetDynamicGraphContext(), V, Index, DegreeCountsDev);

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);

  float ElapsedTime = 0.0f;
  cudaEventElapsedTime(&ElapsedTime, Start, Stop);
  std::printf("Elapsed time : %f ms\n", ElapsedTime);

  std::vector<uint32_t> Count(VertexN);
  std::fill(Count.begin(), Count.end(), 0);
  CHECK_CUDA_ERROR(cudaMemcpy(Count.data(), DegreeCountsDev,
                              sizeof(uint32_t) * VertexN,
                              cudaMemcpyDeviceToHost));

  for (uint32_t I = 0; I < VertexN; ++I)
    EXPECT_EQ(Count[I], (*DegreeCounts)[I]) << "Failed for Vertex " << I;
  CHECK_CUDA_ERROR(cudaFree(DegreeCountsDev));
};

TEST_F(GraphTest, ElementTest) {
  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;
  ASSERT_EQ(VertexN, DegreeCounts->size());

  std::vector<uint32_t> VertexOffsets(DegreeCounts->size());
  ASSERT_EQ(VertexOffsets.size(), DegreeCounts->size());

  thrust::exclusive_scan(thrust::host, DegreeCounts->begin(),
                         DegreeCounts->end(), std::begin(VertexOffsets), 0);

  uint32_t *AdjacentVertices;
  uint32_t *AdjacentVerticesDev;
  AdjacentVertices = new uint32_t[VertexOffsets.back() + DegreeCounts->back()];
  std::memset(AdjacentVertices, 0x00,
              sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back()));

  CHECK_CUDA_ERROR(cudaMalloc(
      &AdjacentVerticesDev,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));
  CHECK_CUDA_ERROR(cudaMemset(
      AdjacentVerticesDev, 0x00,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));

  uint32_t *VertexOffsetsDev;
  CHECK_ERROR(
      cudaMalloc(&VertexOffsetsDev, sizeof(uint32_t) * VertexOffsets.size()));
  CHECK_ERROR(cudaMemcpy(VertexOffsetsDev, VertexOffsets.data(),
                         sizeof(uint32_t) * VertexOffsets.size(),
                         cudaMemcpyHostToDevice));

  cudaEvent_t Start, Stop;
  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  CollectEdges<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      VertexN, Graph->GetDynamicGraphContext(), VertexOffsetsDev,
      AdjacentVerticesDev);

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);

  float ElapsedTime = 0.0f;
  cudaEventElapsedTime(&ElapsedTime, Start, Stop);
  std::printf("Elapsed time : %f ms\n", ElapsedTime);

  CHECK_CUDA_ERROR(cudaMemcpy(AdjacentVertices, AdjacentVerticesDev,
                              sizeof(uint32_t) *
                                  (VertexOffsets.back() + DegreeCounts->back()),
                              cudaMemcpyDeviceToHost));

  std::map<uint32_t, std::set<uint32_t>> AdjacenciesInDev;
  for (uint32_t I = 0; I < VertexN; ++I) {
    auto Iterator = std::begin(VertexOffsets) + I;
    uint32_t Offset = *Iterator;
    AdjacenciesInDev[I].insert(AdjacentVertices + Offset,
                               AdjacentVertices + Offset + (*DegreeCounts)[I]);
  }

  for (uint32_t I = 0; I < VertexN; ++I) {
    EXPECT_EQ((*Adjacencies)[I].size(), AdjacenciesInDev[I].size())
        << "Adjacencies sizes not matching for Vertex I = " << I;

    auto IterAdj = std::begin((*Adjacencies)[I]),
         EndAdj = std::end((*Adjacencies)[I]);
    auto IterAdjDev = std::begin(AdjacenciesInDev[I]),
         EndAdjDev = std::end(AdjacenciesInDev[I]);

    EXPECT_TRUE(std::equal(IterAdj, EndAdj, IterAdjDev))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
    EXPECT_TRUE(std::equal(IterAdjDev, EndAdjDev, IterAdj))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
  }

  CHECK_CUDA_ERROR(cudaFree(AdjacentVerticesDev));
  CHECK_CUDA_ERROR(cudaFree(VertexOffsetsDev));
  delete AdjacentVertices;
};

TEST_F(GraphTest, ElementTestWarpScheduled) {
  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;
  ASSERT_EQ(VertexN, DegreeCounts->size());

  std::vector<uint32_t> VertexOffsets(DegreeCounts->size());
  ASSERT_EQ(VertexOffsets.size(), DegreeCounts->size());

  thrust::exclusive_scan(thrust::host, DegreeCounts->begin(),
                         DegreeCounts->end(), std::begin(VertexOffsets), 0);

  uint32_t *AdjacentVertices;
  uint32_t *AdjacentVerticesDev;
  AdjacentVertices = new uint32_t[VertexOffsets.back() + DegreeCounts->back()];

  CHECK_CUDA_ERROR(cudaMalloc(
      &AdjacentVerticesDev,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));
  CHECK_CUDA_ERROR(cudaMemset(
      AdjacentVerticesDev, 0x00,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));

  uint32_t *VertexOffsetsDev;
  CHECK_ERROR(
      cudaMalloc(&VertexOffsetsDev, sizeof(uint32_t) * VertexOffsets.size()));
  CHECK_ERROR(cudaMemcpy(VertexOffsetsDev, VertexOffsets.data(),
                         sizeof(uint32_t) * VertexOffsets.size(),
                         cudaMemcpyHostToDevice));

  uint32_t *DegreeCountsDev = nullptr;
  cudaEvent_t Start, Stop;
  cudaEventCreate(&Start);
  cudaEventRecord(Start, 0);

  CollectEdgesWarpScheduled<<<256, 256>>>(
      TotalBuckets, Graph->GetDynamicGraphContext(), V, Index, VertexOffsetsDev,
      AdjacentVerticesDev);

  cudaEventCreate(&Stop);
  cudaEventRecord(Stop, 0);
  cudaEventSynchronize(Stop);

  float ElapsedTime = 0.0f;
  cudaEventElapsedTime(&ElapsedTime, Start, Stop);
  std::printf("Elapsed time : %f ms\n", ElapsedTime);

  CHECK_CUDA_ERROR(cudaMemcpy(AdjacentVertices, AdjacentVerticesDev,
                              sizeof(uint32_t) *
                                  (VertexOffsets.back() + DegreeCounts->back()),
                              cudaMemcpyDeviceToHost));

  std::map<uint32_t, std::set<uint32_t>> AdjacenciesInDev;
  for (uint32_t I = 0; I < VertexN; ++I) {
    auto Iterator = std::begin(VertexOffsets) + I;
    uint32_t Offset = *Iterator;
    AdjacenciesInDev[I].insert(AdjacentVertices + Offset,
                               AdjacentVertices + Offset + (*DegreeCounts)[I]);
  }

  for (uint32_t I = 0; I < VertexN; ++I) {
    EXPECT_EQ((*Adjacencies)[I].size(), AdjacenciesInDev[I].size())
        << "Adjacencies sizes not matching for Vertex I = " << I;

    auto IterAdj = std::begin((*Adjacencies)[I]),
         EndAdj = std::end((*Adjacencies)[I]);
    auto IterAdjDev = std::begin(AdjacenciesInDev[I]),
         EndAdjDev = std::end(AdjacenciesInDev[I]);

    EXPECT_TRUE(std::equal(IterAdj, EndAdj, IterAdjDev))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
    EXPECT_TRUE(std::equal(IterAdjDev, EndAdjDev, IterAdj))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
  }

  CHECK_CUDA_ERROR(cudaFree(AdjacentVerticesDev));
  CHECK_CUDA_ERROR(cudaFree(VertexOffsetsDev));
  delete AdjacentVertices;
};

TEST_F(GraphTest, PartialDelete) {
  uint32_t *DeletedEdgesSrc;
  uint32_t *DeletedEdgesDst;

  CHECK_ERROR(cudaMalloc(&DeletedEdgesSrc, sizeof(uint32_t) * DelEdgeN));
  CHECK_ERROR(cudaMalloc(&DeletedEdgesDst, sizeof(uint32_t) * DelEdgeN));
  CHECK_ERROR(cudaMemcpy(DeletedEdgesSrc, DelSrc->data(),
                         sizeof(uint32_t) * DelEdgeN, cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(DeletedEdgesDst, DelDst->data(),
                         sizeof(uint32_t) * DelEdgeN, cudaMemcpyHostToDevice));

  Graph->DeleteEdges(DeletedEdgesSrc, DeletedEdgesDst, DelEdgeN);
  cudaDeviceSynchronize();

  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;
  ASSERT_EQ(VertexN, DegreeCountsAfterDel->size());

  std::vector<uint32_t> VertexOffsets(DegreeCountsAfterDel->size());
  ASSERT_EQ(VertexOffsets.size(), DegreeCountsAfterDel->size());

  thrust::exclusive_scan(thrust::host, DegreeCountsAfterDel->begin(),
                         DegreeCountsAfterDel->end(), std::begin(VertexOffsets),
                         0);

  uint32_t *AdjacentVertices;
  uint32_t *AdjacentVerticesDev;
  AdjacentVertices =
      new uint32_t[VertexOffsets.back() + DegreeCountsAfterDel->back()];

  CHECK_CUDA_ERROR(cudaMalloc(
      &AdjacentVerticesDev, sizeof(uint32_t) * (VertexOffsets.back() +
                                                DegreeCountsAfterDel->back())));
  CHECK_CUDA_ERROR(
      cudaMemset(AdjacentVerticesDev, 0x00,
                 sizeof(uint32_t) *
                     (VertexOffsets.back() + DegreeCountsAfterDel->back())));

  uint32_t *VertexOffsetsDev;
  CHECK_ERROR(
      cudaMalloc(&VertexOffsetsDev, sizeof(uint32_t) * VertexOffsets.size()));
  CHECK_ERROR(cudaMemcpy(VertexOffsetsDev, VertexOffsets.data(),
                         sizeof(uint32_t) * VertexOffsets.size(),
                         cudaMemcpyHostToDevice));

  CollectEdges<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      VertexN, Graph->GetDynamicGraphContext(), VertexOffsetsDev,
      AdjacentVerticesDev);
  cudaDeviceSynchronize();

  CHECK_CUDA_ERROR(cudaMemcpy(
      AdjacentVertices, AdjacentVerticesDev,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCountsAfterDel->back()),
      cudaMemcpyDeviceToHost));

  std::map<uint32_t, std::set<uint32_t>> AdjacenciesInDev;
  for (uint32_t I = 0; I < VertexN; ++I) {
    auto Iterator = std::begin(VertexOffsets) + I;
    uint32_t Offset = *Iterator;
    AdjacenciesInDev[I].insert(AdjacentVertices + Offset,
                               AdjacentVertices + Offset +
                                   (*DegreeCountsAfterDel)[I]);
  }

  for (uint32_t I = 0; I < VertexN; ++I) {
    EXPECT_EQ((*AdjacenciesAfterDelete)[I].size(), AdjacenciesInDev[I].size())
        << "Adjacencies sizes not matching for Vertex I = " << I;

    auto IterAdj = std::begin((*AdjacenciesAfterDelete)[I]),
         EndAdj = std::end((*AdjacenciesAfterDelete)[I]);
    auto IterAdjDev = std::begin(AdjacenciesInDev[I]),
         EndAdjDev = std::end(AdjacenciesInDev[I]);

    EXPECT_TRUE(std::equal(IterAdj, EndAdj, IterAdjDev))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
    EXPECT_TRUE(std::equal(IterAdjDev, EndAdjDev, IterAdj))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
  }

  CHECK_CUDA_ERROR(cudaFree(AdjacentVerticesDev));
  CHECK_CUDA_ERROR(cudaFree(VertexOffsetsDev));
  delete AdjacentVertices;
}

TEST_F(GraphTest, FullDelete) {
  uint32_t _, DelEdgeN;
  std::ifstream DeletionFile("simple");
  DeletionFile >> _ >> DelEdgeN;

  uint32_t *DelSrc = new uint32_t[DelEdgeN];
  uint32_t *DelDst = new uint32_t[DelEdgeN];

  for (int i = 0; i < DelEdgeN; ++i) {
    DeletionFile >> DelSrc[i];
    DeletionFile >> DelDst[i];
  }

  uint32_t *DeletedEdgesSrc;
  uint32_t *DeletedEdgesDst;

  CHECK_ERROR(cudaMalloc(&DeletedEdgesSrc, sizeof(uint32_t) * DelEdgeN));
  CHECK_ERROR(cudaMalloc(&DeletedEdgesDst, sizeof(uint32_t) * DelEdgeN));
  CHECK_ERROR(cudaMemcpy(DeletedEdgesSrc, DelSrc, sizeof(uint32_t) * DelEdgeN,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(DeletedEdgesDst, DelDst, sizeof(uint32_t) * DelEdgeN,
                         cudaMemcpyHostToDevice));

  Graph->DeleteEdges(DeletedEdgesSrc, DeletedEdgesDst, DelEdgeN);
  cudaDeviceSynchronize();

  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;

  uint32_t *TombstoneCountDev = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&TombstoneCountDev, sizeof(uint32_t) * VertexN));
  CHECK_CUDA_ERROR(
      cudaMemset(TombstoneCountDev, 0x00, sizeof(uint32_t) * VertexN));
  CountTombstonesUnweighted<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      VertexN, Graph->GetDynamicGraphContext(), TombstoneCountDev);
  cudaDeviceSynchronize();

  std::vector<uint32_t> Count(VertexN);
  CHECK_CUDA_ERROR(cudaMemcpy(Count.data(), TombstoneCountDev,
                              sizeof(uint32_t) * VertexN,
                              cudaMemcpyDeviceToHost));

  for (uint32_t I = 0; I < VertexN; ++I)
    EXPECT_EQ(Count[I], (*DegreeCounts)[I]) << "Failed for Vertex " << I;

  CHECK_CUDA_ERROR(cudaFree(DeletedEdgesSrc));
  CHECK_CUDA_ERROR(cudaFree(DeletedEdgesDst));
  CHECK_CUDA_ERROR(cudaFree(TombstoneCountDev));
};

struct WGraphTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    if (DegreeCounts == nullptr) {
      std::ifstream InputFile("simple");
      InputFile >> VertexN >> EdgeN;

      DegreeCounts = new std::vector<uint32_t>(VertexN);
      std::fill(DegreeCounts->begin(), DegreeCounts->end(), 0);
      Src = new std::vector<uint32_t>(EdgeN);
      Dst = new std::vector<uint32_t>(EdgeN);
      Weights = new std::vector<uint32_t>(EdgeN);

      Adjacencies = new std::map<uint32_t, std::set<Pair, std::less<Pair>>>();

      CHECK_CUDA_ERROR(cudaMalloc(&SrcDevPtr, sizeof(uint32_t) * EdgeN));
      CHECK_CUDA_ERROR(cudaMalloc(&DstDevPtr, sizeof(uint32_t) * EdgeN));
      CHECK_CUDA_ERROR(cudaMalloc(&WeightsDevPtr, sizeof(uint32_t) * EdgeN));

      uint32_t U, V, Weight;
      for (int i = 0; i < EdgeN; ++i) {
        InputFile >> U >> V;
        (*Src)[i] = U;
        (*Dst)[i] = V;
        // Weight = (*Weights)[i] = (U >> 1 + V >> 1) >> 2;
        Weight = (*Weights)[i] = 1;
        ++((*DegreeCounts)[U]);

        (*Adjacencies)[U].insert({V, Weight});
      }
      InputFile.close();

      CHECK_CUDA_ERROR(cudaMemcpy(SrcDevPtr, Src->data(),
                                  sizeof(uint32_t) * EdgeN,
                                  cudaMemcpyHostToDevice));
      CHECK_CUDA_ERROR(cudaMemcpy(DstDevPtr, Dst->data(),
                                  sizeof(uint32_t) * EdgeN,
                                  cudaMemcpyHostToDevice));
      CHECK_CUDA_ERROR(cudaMemcpy(WeightsDevPtr, Weights->data(),
                                  sizeof(uint32_t) * EdgeN,
                                  cudaMemcpyHostToDevice));

      TheAllocator = new Allocator;
      Graph = new DynGraph<WEdgePolicy>(VertexN, *TheAllocator, 0.7,
                                        DegreeCounts->data(), 0);
      Graph->InsertEdges(SrcDevPtr, DstDevPtr, EdgeN, WeightsDevPtr);
      cudaDeviceSynchronize();
    }
  }

  static void TearDownTestSuite() {
#if 0
    delete DegreeCounts;
    delete Src;
    delete Dst;
    delete Weights;
    delete Adjacencies;

    CHECK_CUDA_ERROR(cudaFree(SrcDevPtr));
    CHECK_CUDA_ERROR(cudaFree(DstDevPtr));
    CHECK_CUDA_ERROR(cudaFree(WeightsDevPtr));

    delete TheAllocator;
    delete Graph;
#endif
  }

  static std::vector<uint32_t> *DegreeCounts;
  static std::vector<uint32_t> *Src;
  static std::vector<uint32_t> *Dst;
  static std::vector<uint32_t> *Weights;
  static uint32_t VertexN;
  static uint32_t EdgeN;

  static std::map<uint32_t, std::set<Pair, std::less<Pair>>> *Adjacencies;

  static uint32_t *SrcDevPtr;
  static uint32_t *DstDevPtr;
  static uint32_t *WeightsDevPtr;

  static Allocator *TheAllocator;
  static DynGraph<WEdgePolicy> *Graph;
};

std::vector<uint32_t> *WGraphTest::DegreeCounts = nullptr;
std::vector<uint32_t> *WGraphTest::Src = nullptr;
std::vector<uint32_t> *WGraphTest::Dst = nullptr;
std::vector<uint32_t> *WGraphTest::Weights = nullptr;

uint32_t WGraphTest::VertexN = 0;
uint32_t WGraphTest::EdgeN = 0;

std::map<uint32_t, std::set<Pair, std::less<Pair>>> *WGraphTest::Adjacencies =
    nullptr;

uint32_t *WGraphTest::SrcDevPtr = nullptr;
uint32_t *WGraphTest::DstDevPtr = nullptr;
uint32_t *WGraphTest::WeightsDevPtr = nullptr;

Allocator *WGraphTest::TheAllocator = nullptr;
DynGraph<WEdgePolicy> *WGraphTest::Graph = nullptr;

TEST_F(WGraphTest, DegreeCountTest) {
  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;

  uint32_t *DegreeCountsDev = nullptr;
  CHECK_CUDA_ERROR(cudaMalloc(&DegreeCountsDev, sizeof(uint32_t) * VertexN));
  CHECK_CUDA_ERROR(
      cudaMemset(DegreeCountsDev, 0x00, sizeof(uint32_t) * VertexN));
  CountDegrees<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      VertexN, Graph->GetDynamicGraphContext(), DegreeCountsDev);
  cudaDeviceSynchronize();

  std::vector<uint32_t> Count(VertexN);
  std::fill(Count.begin(), Count.end(), 0);
  CHECK_CUDA_ERROR(cudaMemcpy(Count.data(), DegreeCountsDev,
                              sizeof(uint32_t) * VertexN,
                              cudaMemcpyDeviceToHost));

  for (uint32_t I = 0; I < VertexN; ++I)
    EXPECT_EQ(Count[I], (*DegreeCounts)[I]) << "Failed for Vertex " << I;
};

TEST_F(WGraphTest, ElementTest) {
  uint32_t ThreadBlockSize = BLOCK_SIZE;
  uint32_t NumberOfThreadBlocks =
      (VertexN + ThreadBlockSize - 1) / ThreadBlockSize;
  ASSERT_EQ(VertexN, DegreeCounts->size());

  std::vector<uint32_t> VertexOffsets(DegreeCounts->size());
  ASSERT_EQ(VertexOffsets.size(), DegreeCounts->size());

  thrust::exclusive_scan(thrust::host, DegreeCounts->begin(),
                         DegreeCounts->end(), std::begin(VertexOffsets), 0);

  uint32_t *AdjacentVertices;
  uint32_t *AdjacentVerticesDev;
  uint32_t *Weights;
  uint32_t *WeightsDev;
  AdjacentVertices = new uint32_t[VertexOffsets.back() + DegreeCounts->back()];
  Weights = new uint32_t[VertexOffsets.back() + DegreeCounts->back()];

  CHECK_CUDA_ERROR(cudaMalloc(
      &AdjacentVerticesDev,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));
  CHECK_CUDA_ERROR(cudaMemset(
      AdjacentVerticesDev, 0x00,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));
  CHECK_CUDA_ERROR(cudaMalloc(
      &WeightsDev,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));
  CHECK_CUDA_ERROR(cudaMemset(
      WeightsDev, 0x00,
      sizeof(uint32_t) * (VertexOffsets.back() + DegreeCounts->back())));

  uint32_t *VertexOffsetsDev;
  CHECK_ERROR(
      cudaMalloc(&VertexOffsetsDev, sizeof(uint32_t) * VertexOffsets.size()));
  CHECK_ERROR(cudaMemcpy(VertexOffsetsDev, VertexOffsets.data(),
                         sizeof(uint32_t) * VertexOffsets.size(),
                         cudaMemcpyHostToDevice));

  CollectEdges<<<NumberOfThreadBlocks, ThreadBlockSize>>>(
      VertexN, Graph->GetDynamicGraphContext(), VertexOffsetsDev,
      AdjacentVerticesDev, WeightsDev);
  cudaDeviceSynchronize();

  CHECK_CUDA_ERROR(cudaMemcpy(AdjacentVertices, AdjacentVerticesDev,
                              sizeof(uint32_t) *
                                  (VertexOffsets.back() + DegreeCounts->back()),
                              cudaMemcpyDeviceToHost));
  CHECK_CUDA_ERROR(cudaMemcpy(Weights, WeightsDev,
                              sizeof(uint32_t) *
                                  (VertexOffsets.back() + DegreeCounts->back()),
                              cudaMemcpyDeviceToHost));

  std::map<uint32_t, std::set<Pair>> AdjacenciesInDev;
  for (uint32_t I = 0; I < VertexN; ++I) {
    auto Iterator = std::begin(VertexOffsets) + I;
    uint32_t Offset = *Iterator;
    for (uint32_t J = Offset; J < (Offset + (*DegreeCounts)[I]); ++J) {
      AdjacenciesInDev[I].insert(Pair(*(AdjacentVertices + J), *(Weights + J)));
    }
  }

  for (uint32_t I = 0; I < VertexN; ++I) {
    EXPECT_EQ((*Adjacencies)[I].size(), AdjacenciesInDev[I].size())
        << "Adjacencies sizes not matching for Vertex I = " << I;

    auto IterAdj = std::begin((*Adjacencies)[I]),
         EndAdj = std::end((*Adjacencies)[I]);
    auto IterAdjDev = std::begin(AdjacenciesInDev[I]),
         EndAdjDev = std::end(AdjacenciesInDev[I]);

    EXPECT_TRUE(std::equal(IterAdj, EndAdj, IterAdjDev))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
    EXPECT_TRUE(std::equal(IterAdjDev, EndAdjDev, IterAdj))
        << "Adjacencies do not have equal elements for Vertex I = " << I;
  }

  CHECK_CUDA_ERROR(cudaFree(AdjacentVerticesDev));
  CHECK_CUDA_ERROR(cudaFree(VertexOffsetsDev));
  delete AdjacentVertices;
  delete Weights;
};

int main(int argc, char *argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
