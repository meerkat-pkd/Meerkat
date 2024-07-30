#ifndef SLABGRAPH_DELETE_EDGE_KERNELS_CUH_
#define SLABGRAPH_DELETE_EDGE_KERNELS_CUH_

template <typename VertexTy, typename ValueTy, typename DynamicGraphContextTy,
          typename CountTy = uint32_t>
__global__ void
DeleteEdgesKernel(VertexTy *SourceVertices, VertexTy *DestinationVertices,
                  CountTy NumberOfEdges, DynamicGraphContextTy GraphContext) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfEdges)
    return;

  VertexTy SourceVertex{};
  VertexTy DestinationVertex{};
  bool ToDelete = false;

  if (ThreadID < NumberOfEdges) {
    SourceVertex = SourceVertices[ThreadID];
    DestinationVertex = DestinationVertices[ThreadID];
    ToDelete = (SourceVertex != DestinationVertex);
  }

  GraphContext.DeleteEdge(ToDelete, LaneID, SourceVertex, DestinationVertex);
}

#endif // SLABGRAPH_DELETE_EDGE_KERNELS_CUH_
