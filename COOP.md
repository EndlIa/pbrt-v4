# PathGraph 协作文档0424

## 1. 入口

- 头文件位置：src/pbrt/pathGraph.h
- 命名空间：pbrt
- 建图过程在 src/pbrt/cpu/integrators.cpp
- pathGraph创建开关在 src/pbrt/cpu/render.cpp line160左右

- 请在你的代码模块中使用 auto snapshot = pathGraphBuilder.BuildSnapshot(); 来获取pathGraph信息

## 2. 接口

- PathGraphSnapshot
    - Vertices()
    - LightEdges()
    - ContEdges()
    - ContEdgesIntoVertex(uint64_t vertexId) 返回边索引
    - LightEdgesOfVertex(uint64_t vertexId)  返回边索引
    - NeighborsOfVertex(uint64_t vertexId) 返回顶点索引
    

## 3. 约定

- 使用TestOneIntegrator建图，请不要关闭sampleLight
- 所有方向向量按世界坐标解释。
- !!!!!!!!!!!!     edge的vertexA靠近相机 vertexB靠近光源 L_A L_B同理
- !!!!!!!!!!!!     integrator中ray指向光源，而wi/wo的约定与pathGraph中可能并不相同，pathGraph中 wo由hitpoint指向相机 wi由光源指向hitpoint 
- 目前ef的brdf项定义为 起点的brdf
- rrQ pdf 和 ef 可能存在各种神秘的约定问题，如果不确定请务必问我
- 不要依赖裸 BSDF 指针 请用 materialId。
- ContEdgesIntoVertex 查询的是朝向该Vertex传递能量的边
- (VertexID == 0)有保留语义，请从(VertexID == 1) 开始遍历


## 4. 暂未暴露的量
- SurfaceVertex 的 clusterId, 感觉用不上

## 5. 其他
- 注意使用 .exr 而非 .png 输出图像以计算MRSE，可使用build/imgtool diff --metric MRSE --reference ref.exr test.exr，使用endlia编写的img_eval/eval.py 计算MRSE 也可
