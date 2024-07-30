#ifndef SLABGRAPH_IMPL_CUH_
#define SLABGRAPH_IMPL_CUH_

#include "SlabGraph.cuh"

template <typename EdgePolicyT>
void DynamicGraph<EdgePolicyT, true>::InsertEdges(VertexT *SourceVertices,
                                                  VertexT *DestinationVertices,
                                                  CountT NumberOfEdges,
                                                  EdgeValueT *EdgeValues) {

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (NumberOfEdges + ThreadBlockSize - 1) / ThreadBlockSize;

  InsertEdgesKernel<VertexT, EdgeValueT, GraphContextT, CountT>
      <<<NumberOfThreadBlocks, ThreadBlockSize>>>(
          SourceVertices, DestinationVertices, NumberOfEdges, EdgeValues,
          TheGraphContext);
}

template <typename EdgePolicyT>
void DynamicGraph<EdgePolicyT, false>::InsertEdges(VertexT *SourceVertices,
                                                   VertexT *DestinationVertices,
                                                   CountT NumberOfEdges) {

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (NumberOfEdges + ThreadBlockSize - 1) / ThreadBlockSize;

  InsertEdgesKernel<VertexT, GraphContextT, CountT>
      <<<NumberOfThreadBlocks, ThreadBlockSize>>>(
          SourceVertices, DestinationVertices, NumberOfEdges, TheGraphContext);
}

template <typename EdgePolicyT>
void DynamicGraph<EdgePolicyT, true>::DeleteEdges(VertexT *SourceVertices,
                                                  VertexT *DestinationVertices,
                                                  CountT NumberOfEdges) {

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (NumberOfEdges + ThreadBlockSize - 1) / ThreadBlockSize;

  DeleteEdgesKernel<VertexT, EdgeValueT, GraphContextT, CountT>
      <<<NumberOfThreadBlocks, ThreadBlockSize>>>(
          SourceVertices, DestinationVertices, NumberOfEdges, TheGraphContext);
}

template <typename EdgePolicyT>
void DynamicGraph<EdgePolicyT, false>::DeleteEdges(VertexT *SourceVertices,
                                                   VertexT *DestinationVertices,
                                                   CountT NumberOfEdges) {

  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (NumberOfEdges + ThreadBlockSize - 1) / ThreadBlockSize;

  DeleteEdgesKernel<VertexT, EdgeValueT, GraphContextT, CountT>
      <<<NumberOfThreadBlocks, ThreadBlockSize>>>(
          SourceVertices, DestinationVertices, NumberOfEdges, TheGraphContext);
}

template <typename EdgePolicyT>
void DynamicGraph<EdgePolicyT, false>::UpdateSlabPointers() {
  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (NodesMax + ThreadBlockSize - 1) / ThreadBlockSize;

  UpdateSlabListPointers<GraphContextT>
      <<<NumberOfThreadBlocks, ThreadBlockSize>>>(TheGraphContext, NodesMax);
}

template <typename EdgePolicyT>
void DynamicGraph<EdgePolicyT, true>::UpdateSlabPointers() {
  uint32_t ThreadBlockSize = __BLOCK_SIZE__;
  uint32_t NumberOfThreadBlocks =
      (NodesMax + ThreadBlockSize - 1) / ThreadBlockSize;

  UpdateSlabListPointers<GraphContextT>
      <<<NumberOfThreadBlocks, ThreadBlockSize>>>(TheGraphContext, NodesMax);
}

#if 0
  template <typename EdgePolicyT>
  template <typename FilterMapTy>
  typename std::enable_if<std::is_default_constructible<FilterMapTy>::value>::type
  DynamicGraph<EdgePolicyT, true>::UpdateEdges(VertexT *SourceVertices,
                                              VertexT *DestinationVertices,
                                              CountT NumberOfEdges,
                                              EdgeValueT *NewEdgeValues,
                                              FilterMapTy *EdgeFilterMaps) {

    uint32_t ThreadBlockSize = __BLOCK_SIZE__;
    uint32_t NumberOfThreadBlocks =
        (NumberOfEdges + ThreadBlockSize - 1) / ThreadBlockSize;

    UpdateEdgesKernel<VertexT, EdgeValueT, GraphContextT, FilterMapTy, CountT>
        <<<NumberOfThreadBlocks, ThreadBlockSize>>>(
            SourceVertices, DestinationVertices, NumberOfEdges, NewEdgeValues,
            TheGraphContext, EdgeFilterMaps);
  }
#endif

#endif // SLABGRAPH_IMPL_CUH_
