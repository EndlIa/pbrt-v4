# PathGraph 协作文档0423

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
- !!!!!!!!!!!!     edge的vertexA靠近相机 vertexB靠近光源
- !!!!!!!!!!!!     integrator中ray指向光源 wo由hitpoint指向相机 wi由光源指向hitpoint 
- 目前ef的brdf项定义为 起点的brdf
- rrQ pdf 和 ef 可能存在各种神秘的约定问题，如果不确定请务必问我
- 不要依赖裸 BSDF 指针 请用 bsdfType/materialId/shapeId。

## 4. 暂未暴露的量

- SurfaceVertex 的 L_in、L_out
- SurfaceVertex 的 clusterId：聚类后产物，不放入原始顶点记录。

## 5. 其他
- 注意使用 .exr 而非 .png 输出图像以计算MRSE，可使用build/imgtool diff --metric MRSE --reference ref.exr test.exr，使用endlia编写的img_eval/eval.py 计算MRSE 也可
