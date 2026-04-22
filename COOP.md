# PathGraph 协作接口

## 1. 入口

- 头文件位置：src/pbrt/pathGraph.h
- 命名空间：pbrt

## 2. 给布朗尼的接口

- PathGraphSnapshot
    - Vertices()
    - LightEdges()
    - ContEdges()
    - NeighborEdges()

## 3. 数据记录类型

- SurfaceVertexRecord
    - 关键字段：vertexId, pathId, depth, pos, normal, wo, albedo, pdfBsdf
    - 稳定标识字段：bsdfType, materialId, shapeId

- LightEdgeRecord
    - 关键字段：fromVertexId, wi, L, fCos, pdfLight, pdfSelectLight, pdfBsdfForWi, misWeight, misMode
    - 其中 fCos 约定为：BSDF(wi) * |n · wi|

- ContEdgeRecord
    - 关键字段：fromVertexId, toVertexId, wi, pdfBsdf, q, invQ
    - 约定：若算法需要统一权重接口，可按默认 weight = 1 处理（当前不单独存字段）

- NeighborEdgeRecord
    - 关键字段：a, b, clusterId
    - 可能用到的：distance, weight

## 4. 约定

- 所有方向向量按世界坐标解释。
- 未启用 RR 时：q = 1，invQ = 1。
- 未启用 MIS 时：misWeight = 1，misMode = 0（uint8_t）。
- 不要依赖裸 BSDF 指针身份；请用 bsdfType/materialId/shapeId。

## 5. 暂不暴露的量

- SurfaceVertex 的 L_in、L_out：口径未冻结，暂不作为采集层字段。
- SurfaceVertex 的 clusterId：聚类后产物，不放入原始顶点记录。
- ContEdge 的 L_e：暂不暴露，避免与发光项 Le 语义混淆。

## 6. 工作流

1. 读取 Vertices/LightEdges/ContEdges。
2. 执行你的聚合/传播算子。
3. 如需聚类，输出独立的 vertexId -> clusterId 映射或生成 NeighborEdge。