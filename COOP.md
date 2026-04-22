# PathGraph 协作文档

## 1. 入口

- 头文件位置：src/pbrt/pathGraph.h
- 命名空间：pbrt

## 2. 接口

- PathGraphSnapshot
    - Vertices()
    - LightEdges()
    - ContEdges()
    - NeighborEdges()
    
## 3. 约定

- 所有方向向量按世界坐标解释。
- 目前ef的brdf项定义为 起点的brdf
- rrQ pdf 和 ef 可能存在各种神秘的约定问题，如果不确定请务必问我
- 不要依赖裸 BSDF 指针 请用 bsdfType/materialId/shapeId。

## 4. 暂未暴露的量

- SurfaceVertex 的 L_in、L_out
- SurfaceVertex 的 clusterId：聚类后产物，不放入原始顶点记录。
