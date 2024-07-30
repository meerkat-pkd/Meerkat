#ifndef SLABGRAPH_INSERT_EDGE_KERNELS_CUH_
#define SLABGRAPH_INSERT_EDGE_KERNELS_CUH_

template <typename VertexTy, typename ValueTy, typename DynamicGraphContextTy,
          typename CountTy = uint32_t>
__global__ void InsertEdgesKernel(VertexTy *SourceVertices,
                                  VertexTy *DestinationVertices,
                                  CountTy NumberOfEdges, ValueTy *EdgeValues,
                                  DynamicGraphContextTy GraphContext) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfEdges)
    return;

  VertexTy SourceVertex{};
  VertexTy DestinationVertex{};
  ValueTy EdgeValue{};
  bool ToInsert = false;

  if (ThreadID < NumberOfEdges) {
    SourceVertex = SourceVertices[ThreadID];
    DestinationVertex = DestinationVertices[ThreadID];
    EdgeValue = (EdgeValues == nullptr) ? ValueTy {} : EdgeValues[ThreadID];
    ToInsert = (SourceVertex != DestinationVertex);
  }

  typename DynamicGraphContextTy::EdgeDynAllocCtxt AllocCtxt(
      GraphContext.GetEdgeDynAllocCtxt());
  AllocCtxt.initAllocator(ThreadID, LaneID);

  GraphContext.InsertEdge(ToInsert, LaneID, SourceVertex, DestinationVertex,
                          EdgeValue, AllocCtxt);
}

template <typename VertexTy, typename DynamicGraphContextTy,
          typename CountTy = uint32_t>
__global__ void
InsertEdgesKernel(VertexTy *SourceVertices, VertexTy *DestinationVertices,
                  CountTy NumberOfEdges, DynamicGraphContextTy GraphContext) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfEdges)
    return;

  VertexTy SourceVertex{};
  VertexTy DestinationVertex{};
  bool ToInsert = false;

  if (ThreadID < NumberOfEdges) {
    SourceVertex = SourceVertices[ThreadID];
    DestinationVertex = DestinationVertices[ThreadID];
    ToInsert = (SourceVertex != DestinationVertex);
  }

  typename DynamicGraphContextTy::EdgeDynAllocCtxt AllocCtxt(
      GraphContext.GetEdgeDynAllocCtxt());
  AllocCtxt.initAllocator(ThreadID, LaneID);

  GraphContext.InsertEdge(ToInsert, LaneID, SourceVertex, DestinationVertex,
                          AllocCtxt);
}

#endif // SLABGRAPH_INSERT_EDGE_KERNELS_CUH_