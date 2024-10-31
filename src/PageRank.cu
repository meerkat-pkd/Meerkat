#include "Algorithms/PageRank.cuh"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <string>
#include <utility>

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::printf("Usage: %s <basegraph-filename> <insertion/deletion[0/1]]> "
                "<update-file> <num-batches> <damping-factor> <error-margin>\n",
                argv[0]);
    std::exit(1);
  }

  std::string Filename = std::string(argv[1]);
  std::string Op = std::string(argv[2]);
  std::string UpdateFile = std::string(argv[3]);
  std::string NumBatches = std::string(argv[4]);
  float DampingFactor = static_cast<float>(std::atof(argv[5]));
  float ErrorMargin = static_cast<float>(std::atof(argv[6]));

  double LoadFactor = 0.7;
  float ElapsedTime;

  std::FILE *F = std::fopen(Filename.c_str(), "r");

  using AllocPolicy = LightAllocatorPolicy<8, 32, 1>;
  using Allocator = typename AllocPolicy::DynamicAllocatorT;
  using GraphT = PageRank<AllocPolicy>::GraphT;

  uint32_t VertexN, EdgeN;

  (void)std::fscanf(F, "%d%d", &VertexN, &EdgeN);
  std::cout << Filename;
  std::flush(std::cout);

  uint32_t *VertexInDegree = new uint32_t[VertexN];
  uint32_t *VertexOutDegree = new uint32_t[VertexN];

  std::fill(VertexInDegree, VertexInDegree + VertexN, 0);
  std::fill(VertexOutDegree, VertexOutDegree + VertexN, 0);

  uint32_t *Src = new uint32_t[EdgeN];
  uint32_t *Dst = new uint32_t[EdgeN];

  for (int i = 0; i < EdgeN; ++i) {
    (void)std::fscanf(F, "%d%d", &Src[i], &Dst[i]);
    ++VertexOutDegree[Src[i]];
  }

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
  GraphT Graph{VertexN, *Alloc, LoadFactor, VertexInDegree, 0};

  Graph.InsertEdges(SrcDevPtr, DstDevPtr, EdgeN);
  cudaDeviceSynchronize();

  CHECK_ERROR(cudaFree(SrcDevPtr));
  CHECK_ERROR(cudaFree(DstDevPtr));
  delete[] VertexInDegree;

  PageRank<AllocPolicy> P(Graph, DampingFactor, ErrorMargin, VertexN,
                          VertexOutDegree, 100);

  delete[] VertexOutDegree;

  // --------------------------- Static --------------------------------------
  ElapsedTime = P.Static();
  std::cout << "," << ElapsedTime;
  std::flush(std::cout);

  // -------------------------------------------------------------------------

  bool IsInsertion = (Op == "0");
  uint32_t BatchesN = std::stoi(NumBatches);

  for (int File = 0; File < BatchesN; ++File) {
    std::ifstream BatchFile(UpdateFile + ".batch." + std::to_string(File));
    uint32_t BatchEdgesN;
    BatchFile >> BatchEdgesN;

    uint32_t Src, Dst;

    std::set<std::pair<uint32_t, uint32_t>> BatchEdges;

    for (uint32_t I = 0; I < BatchEdgesN; ++I) {
      BatchFile >> Src >> Dst;
      BatchEdges.insert(std::make_pair(Src, Dst));
    }

    BatchEdgesN = BatchEdges.size();

    uint32_t *BatchEdgesSrc = new uint32_t[BatchEdgesN];
    uint32_t *BatchEdgesDst = new uint32_t[BatchEdgesN];
    uint32_t *BatchEdgesSrcDev;
    uint32_t *BatchEdgesDstDev;

    CHECK_ERROR(cudaMalloc(&BatchEdgesSrcDev, sizeof(uint32_t) * BatchEdgesN));
    CHECK_ERROR(cudaMalloc(&BatchEdgesDstDev, sizeof(uint32_t) * BatchEdgesN));

    auto Iter = std::begin(BatchEdges);
    for (int I = 0; I < BatchEdgesN; ++I) {
      BatchEdgesSrc[I] = (*Iter).first;
      BatchEdgesDst[I] = (*Iter).second;
      ++Iter;
    }

    CHECK_ERROR(cudaMemcpy(BatchEdgesSrcDev, BatchEdgesSrc,
                           sizeof(uint32_t) * BatchEdgesN,
                           cudaMemcpyHostToDevice));

    CHECK_ERROR(cudaMemcpy(BatchEdgesDstDev, BatchEdgesDst,
                           sizeof(uint32_t) * BatchEdgesN,
                           cudaMemcpyHostToDevice));

    delete[] BatchEdgesSrc;
    delete[] BatchEdgesDst;

    if (IsInsertion)
      Graph.InsertEdges(BatchEdgesSrcDev, BatchEdgesDstDev, BatchEdgesN);
    else
      Graph.DeleteEdges(BatchEdgesSrcDev, BatchEdgesDstDev, BatchEdgesN);
    cudaDeviceSynchronize();

    ElapsedTime =
        IsInsertion
            ? P.Incremental(BatchEdgesSrcDev, BatchEdgesDstDev, BatchEdgesN)
            : P.Decremental(BatchEdgesSrcDev, BatchEdgesDstDev, BatchEdgesN);

    CHECK_ERROR(cudaFree(BatchEdgesSrcDev));
    CHECK_ERROR(cudaFree(BatchEdgesDstDev));

    std::cout << "," << ElapsedTime;
    std::flush(std::cout);
  }

  std::cout << std::endl;

  return 0;
}
