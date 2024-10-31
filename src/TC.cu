#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>

#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "Algorithms/TC.cuh"

int main(int argc, char *argv[]) {
  if (argc < 5) {
    printf("Usage: %s <basegraph-filename> <insertion/deletion[0/1]]> "
           "<update-file> <num-batches> \n",
           argv[0]);
    exit(1);
  }

  std::string Filename = std::string(argv[1]);
  std::string Op = std::string(argv[2]);
  std::string UpdateFile = std::string(argv[3]);
  std::string NumBatches = std::string(argv[4]);

  std::ifstream base_graph(Filename);

  using AllocPolicy = LightAllocatorPolicy<8, 32, 1>;
  using Allocator = typename AllocPolicy::DynamicAllocatorT;
  using GraphT = TC<AllocPolicy>::GraphT;

  cudaSetDevice(0);

  uint32_t VertexN, EdgeN;
  float ElapsedTime = 0.0f;

  base_graph >> VertexN >> EdgeN;

  std::vector<uint32_t> VertexDegreeHints{
      std::move(std::vector<uint32_t>(VertexN, 1))};
  std::cout << Filename;
  std::flush(std::cout);

  uint32_t *Src = new uint32_t[EdgeN];
  uint32_t *Dst = new uint32_t[EdgeN];

  for (int i = 0; i < EdgeN; ++i) {
    base_graph >> Src[i] >> Dst[i];
    ++VertexDegreeHints[Src[i]];
  }

  double LoadFactor = 0.7;

  uint32_t *SrcDevPtr;
  uint32_t *DstDevPtr;

  CHECK_ERROR(cudaMalloc(&SrcDevPtr, sizeof(uint32_t) * EdgeN));
  CHECK_ERROR(cudaMalloc(&DstDevPtr, sizeof(uint32_t) * EdgeN));

  CHECK_ERROR(cudaMemcpy(SrcDevPtr, Src, sizeof(uint32_t) * EdgeN,
                         cudaMemcpyHostToDevice));
  CHECK_ERROR(cudaMemcpy(DstDevPtr, Dst, sizeof(uint32_t) * EdgeN,
                         cudaMemcpyHostToDevice));
  delete[] Src;
  delete[] Dst;

  std::unique_ptr<Allocator> Alloc{new Allocator};
  GraphT Graph{VertexN, *Alloc, LoadFactor, VertexDegreeHints.data(), 0};

  Graph.InsertEdges(SrcDevPtr, DstDevPtr, EdgeN);
  cudaDeviceSynchronize();

  CHECK_ERROR(cudaFree(SrcDevPtr));
  CHECK_ERROR(cudaFree(DstDevPtr));

  TC<AllocPolicy> TCAlgorithm(Graph, VertexN);

  ElapsedTime = TCAlgorithm.Static(EdgeN);
  std::cout << "," << ElapsedTime;
  std::flush(std::cout);

  bool IsInsertion = (Op == "0");

  if (IsInsertion) {
    for (int file = 0; file < std::stoi(NumBatches); file++) {
      std::ifstream insertions(UpdateFile + ".batch." + std::to_string(file));
      uint32_t InsEdgeN;
      insertions >> InsEdgeN;

      uint32_t *InsSrc = new uint32_t[InsEdgeN];
      uint32_t *InsDst = new uint32_t[InsEdgeN];

      for (int I = 0; I < InsEdgeN; ++I)
        insertions >> InsSrc[I] >> InsDst[I];

      uint32_t *EdgesSrcDevPtr;
      uint32_t *EdgesDstDevPtr;

      CHECK_ERROR(cudaMalloc(&EdgesSrcDevPtr, sizeof(uint32_t) * InsEdgeN));
      CHECK_ERROR(cudaMalloc(&EdgesDstDevPtr, sizeof(uint32_t) * InsEdgeN));

      CHECK_ERROR(cudaMemcpy(EdgesSrcDevPtr, InsSrc,
                             sizeof(uint32_t) * InsEdgeN,
                             cudaMemcpyHostToDevice));
      CHECK_ERROR(cudaMemcpy(EdgesDstDevPtr, InsDst,
                             sizeof(uint32_t) * InsEdgeN,
                             cudaMemcpyHostToDevice));

      Graph.InsertEdges(EdgesSrcDevPtr, EdgesDstDevPtr, InsEdgeN);
      cudaDeviceSynchronize();

      delete[] InsSrc;
      delete[] InsDst;

      float ElapsedTime = TCAlgorithm.Incremental(
          EdgesSrcDevPtr, EdgesDstDevPtr, InsEdgeN, Alloc);

      std::cout << "," << ElapsedTime;
      std::flush(std::cout);

      CHECK_ERROR(cudaFree(EdgesSrcDevPtr));
      CHECK_ERROR(cudaFree(EdgesDstDevPtr));
    }
  } else if (Op == "1") { // UK:Deletion operation.
    for (int file = 0; file < std::stoi(NumBatches); file++) {
      std::ifstream deletions(UpdateFile + ".batch." + std::to_string(file));
      uint32_t DelEdgeN;
      deletions >> DelEdgeN;

      uint32_t *DelSrc = new uint32_t[DelEdgeN];
      uint32_t *DelDst = new uint32_t[DelEdgeN];

      for (int I = 0; I < DelEdgeN; ++I)
        deletions >> DelSrc[I] >> DelDst[I];

      uint32_t *EdgesSrcDevPtr;
      uint32_t *EdgesDstDevPtr;

      CHECK_ERROR(cudaMalloc(&EdgesSrcDevPtr, sizeof(uint32_t) * DelEdgeN));
      CHECK_ERROR(cudaMalloc(&EdgesDstDevPtr, sizeof(uint32_t) * DelEdgeN));

      CHECK_ERROR(cudaMemcpy(EdgesSrcDevPtr, DelSrc,
                             sizeof(uint32_t) * DelEdgeN,
                             cudaMemcpyHostToDevice));
      CHECK_ERROR(cudaMemcpy(EdgesDstDevPtr, DelDst,
                             sizeof(uint32_t) * DelEdgeN,
                             cudaMemcpyHostToDevice));

      Graph.DeleteEdges(EdgesSrcDevPtr, EdgesDstDevPtr, DelEdgeN);
      cudaDeviceSynchronize();

      delete[] DelSrc;
      delete[] DelDst;

      float ElapsedTime = TCAlgorithm.Decremental(
          EdgesSrcDevPtr, EdgesDstDevPtr, DelEdgeN, Alloc);

      std::cout << "," << ElapsedTime;
      std::flush(std::cout);

      CHECK_ERROR(cudaFree(EdgesSrcDevPtr));
      CHECK_ERROR(cudaFree(EdgesDstDevPtr));
    }
  }

  std::cout << std::endl;
  return 0;
}
