#ifndef SLABGRAPH_UPDATE_EDGE_KERNELS_CUH_
#define SLABGRAPH_UPDATE_EDGE_KERNELS_CUH_

#include <limits>

template <typename VertexTy, typename ValueTy, typename DynamicGraphContextTy,
          typename FilterMapTy, typename CountTy>
__global__ void UpdateEdgesKernel(VertexTy *SourceVertices,
                                  VertexTy *DestinationVertices,
                                  CountTy NumberOfEdges, ValueTy *EdgeValues,
                                  DynamicGraphContextTy GraphContext,
                                  FilterMapTy *EdgeFilterMaps) {
  uint32_t ThreadID = blockDim.x * blockIdx.x + threadIdx.x;
  uint32_t LaneID = threadIdx.x & 0x1F;

  if ((ThreadID - LaneID) >= NumberOfEdges)
    return;

  VertexTy SourceVertex{};
  VertexTy DestinationVertex{};
  ValueTy EdgeValue{};
  bool ToUpdate = false;
  FilterMapTy *EdgeFilterMap = nullptr;

  if (ThreadID < NumberOfEdges) {
    SourceVertex = SourceVertices[ThreadID];
    DestinationVertex = DestinationVertices[ThreadID];
    EdgeValue = (EdgeValues == nullptr) ? EdgeValue : EdgeValues[ThreadID];
    ToUpdate = (SourceVertex != DestinationVertex);
    EdgeFilterMap =
        (EdgeFilterMaps == nullptr) ? nullptr : EdgeFilterMaps[ThreadID];
  }

  GraphContext.UpdateEdge(ToUpdate, LaneID, SourceVertex, DestinationVertex,
                          EdgeValue, EdgeFilterMap);
}

#endif // SLABGRAPH_UPDATE_EDGE_KERNELS_CUH_