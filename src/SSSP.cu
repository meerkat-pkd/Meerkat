#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <iostream>
#include <memory>
#include <vector>

#include "Algorithms/SSSP.cuh"

int main(int argc, char *argv[]) {
  using AllocPolicy = FullAllocatorPolicy<9, 48, 1>;
  using Allocator = typename AllocPolicy::DynamicAllocatorT;

  std::cout << argv[1];
  std::flush(std::cout);

  std::string Filename = std::string(argv[1]);
  std::string Op = std::string(argv[2]);
  std::string UpdateFile = std::string(argv[3]);
  std::string NumBatches = std::string(argv[4]);
  uint32_t Seed = std::atoi(argv[5]);

  std::srand(Seed);

  bool IsWikiTalk = (std::strstr(Filename.c_str(), "wiki.txt") != NULL);
  bool IsOrkut = (std::strstr(Filename.c_str(), "orkut.txt") != NULL);
  uint32_t TheSrc = IsWikiTalk ? 2 : 0;

  std::FILE *GraphFile = std::fopen(Filename.c_str(), "r");
  uint32_t VertexN, EdgeN;
  float ElapsedTime = 0.0f;

  std::fscanf(GraphFile, "%d%d", &VertexN, &EdgeN);

  // initialising degree hints with 1
  std::vector<uint32_t> VertexDegreeHints{
      std::move(std::vector<uint32_t>(VertexN, 1))};
  std::unique_ptr<uint32_t[]> Src{new uint32_t[EdgeN]};
  std::unique_ptr<uint32_t[]> Dst{new uint32_t[EdgeN]};
  std::unique_ptr<uint32_t[]> Wgt{new uint32_t[EdgeN]};

  if (IsOrkut) {
    for (int i = 0; i < EdgeN; ++i) {
      (void)fscanf(GraphFile, "%d%d", &Src[i], &Dst[i]);
      Wgt[i] = 1 + (rand() % 100);
      ++VertexDegreeHints[Src[i]];
    }
  } else {
    for (int i = 0; i < EdgeN; ++i) {
      (void)fscanf(GraphFile, "%d%d", &Src[i], &Dst[i]);
      Wgt[i] = 1 + (rand() % 100);
    }
  }

  double LoadFactor = 0.7;

  uint32_t *SrcDevPtr;
  uint32_t *DstDevPtr;
  uint32_t *WgtDevPtr;

  // initialise GPU variables and memory
  cudaSetDevice(0);
  CHECK_ERROR(cudaMalloc(&SrcDevPtr, sizeof(uint32_t) * EdgeN));
  CHECK_ERROR(cudaMalloc(&DstDevPtr, sizeof(uint32_t) * EdgeN));
  CHECK_ERROR(cudaMalloc(&WgtDevPtr, sizeof(uint32_t) * EdgeN));

  CHECK_ERROR(cudaMemcpy(SrcDevPtr, Src.get(), sizeof(uint32_t) * EdgeN,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(DstDevPtr, Dst.get(), sizeof(uint32_t) * EdgeN,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(WgtDevPtr, Wgt.get(), sizeof(uint32_t) * EdgeN,
                         cudaMemcpyHostToDevice));

  using AllocPolicy = FullAllocatorPolicy<9, 48, 1>;
  using Allocator = typename AllocPolicy::DynamicAllocatorT;
  std::unique_ptr<Allocator> Alloc{new Allocator};

  using DynGraph = typename SSSP<AllocPolicy>::GraphT;
  using DynGraphContext = typename DynGraph::GraphContextT;
  DynGraph Graph{VertexN, *Alloc, LoadFactor, VertexDegreeHints.data(), 0};

  Graph.InsertEdges(SrcDevPtr, DstDevPtr, EdgeN, WgtDevPtr);
  cudaDeviceSynchronize();

  CHECK_ERROR(cudaFree(SrcDevPtr));
  CHECK_ERROR(cudaFree(DstDevPtr));
  CHECK_ERROR(cudaFree(WgtDevPtr));

  SSSP<AllocPolicy> S(Graph, TheSrc, VertexN);
  ElapsedTime = S.Static(EdgeN);

  std::cout << "," << ElapsedTime;
  std::flush(std::cout);

  bool IsInsertion = (Op == "0");
  uint32_t BatchesN = std::stoi(NumBatches);

  for (int File = 0; File < BatchesN; ++File) {
    std::FILE *BatchFile = std::fopen(
        std::string(UpdateFile + ".batch." + std::to_string(File)).c_str(),
        "r");
    uint32_t BatchEdgesN;
    std::fscanf(BatchFile, "%d", &BatchEdgesN);

    uint32_t *BatchEdgesSrc = new uint32_t[BatchEdgesN];
    uint32_t *BatchEdgesDst = new uint32_t[BatchEdgesN];
    uint32_t *BatchEdgesWgt = new uint32_t[BatchEdgesN];
    uint32_t Src, Dst;

    for (uint32_t I = 0; I < BatchEdgesN; ++I) {
      std::fscanf(BatchFile, "%d%d", &Src, &Dst);
      BatchEdgesSrc[I] = Src;
      BatchEdgesDst[I] = Dst;
      BatchEdgesWgt[I] = 1 + (rand() % 100);
    }

    uint32_t *BatchEdgesSrcDev;
    uint32_t *BatchEdgesDstDev;
    uint32_t *BatchEdgesWgtDev;

    CHECK_ERROR(cudaMalloc(&BatchEdgesSrcDev, sizeof(uint32_t) * BatchEdgesN));
    CHECK_ERROR(cudaMemcpy(BatchEdgesSrcDev, BatchEdgesSrc,
                           sizeof(uint32_t) * BatchEdgesN,
                           cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc(&BatchEdgesDstDev, sizeof(uint32_t) * BatchEdgesN));
    CHECK_ERROR(cudaMemcpy(BatchEdgesDstDev, BatchEdgesDst,
                           sizeof(uint32_t) * BatchEdgesN,
                           cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMalloc(&BatchEdgesWgtDev, sizeof(uint32_t) * BatchEdgesN));
    CHECK_ERROR(cudaMemcpy(BatchEdgesWgtDev, BatchEdgesWgt,
                           sizeof(uint32_t) * BatchEdgesN,
                           cudaMemcpyHostToDevice));

    delete[] BatchEdgesSrc;
    delete[] BatchEdgesDst;
    delete[] BatchEdgesWgt;

    bool IsInsertion = (Op == "0");

    if (IsInsertion)
      Graph.InsertEdges(BatchEdgesSrcDev, BatchEdgesDstDev, BatchEdgesN,
                        BatchEdgesWgtDev);
    else
      Graph.DeleteEdges(BatchEdgesSrcDev, BatchEdgesDstDev, BatchEdgesN);
    cudaDeviceSynchronize();

    float ElapsedTime;
    ElapsedTime = IsInsertion
                      ? S.Incremental(BatchEdgesSrcDev, BatchEdgesDstDev,
                                      BatchEdgesWgtDev, BatchEdgesN)
                      : S.Decremental(BatchEdgesSrcDev, BatchEdgesDstDev,
                                      BatchEdgesWgtDev, BatchEdgesN);

    CHECK_ERROR(cudaFree(BatchEdgesSrcDev));
    CHECK_ERROR(cudaFree(BatchEdgesDstDev));
    CHECK_ERROR(cudaFree(BatchEdgesWgtDev));

    std::cout << "," << ElapsedTime;
    std::flush(std::cout);
  }

  std::cout << std::endl;
  return 0;
}