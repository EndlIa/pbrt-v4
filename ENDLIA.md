# pathGraph.h 设计思路与约定

本文档记录 `src/pbrt/pathGraph.h` 中 Path Graph 接口的设计意图、数据含义和使用约定。它对应论文 *Path Graphs: Iterative Path Space Filtering* 的工程化骨架：路径追踪阶段只负责捕获图数据，后处理阶段通过 snapshot 读取这些数据，再做聚合、传播和 final gather。

## 设计目标

- **捕获与求解解耦**：Integrator 只通过 `PathGraphSink` 追加路径顶点和边，不直接关心聚类、聚合或迭代求解。
- **多线程友好**：`PathGraphBuilder::GetThreadLocalSink()` 返回线程本地 sink，渲染线程各写各的，最后由 `BuildSnapshot()` 合并。
- **求解快照**：`PathGraphSnapshot` 合并线程本地图数据，后续 direct / indirect / final gather 会在 snapshot 内更新 `L_direct`、`L_indirect`、`L_out`。
- **BSDF 指针约定**：`SurfaceVertex::bsdf` 指向 `PathGraphThreadData` 中保存的 `BSDF` 副本，用于后续聚合 evaluator 评估 `bsdf(v.wo, e.wi)`。为避免引用每个 sample 结束就 reset 的普通 `ScratchBuffer`，path graph 为每个线程持有自己的长期 `ScratchBuffer`；`testone` 在 active path graph 下只构建一次 BSDF，并让渲染和 graph 捕获共用它。这些 BSDF/BxDF 数据在本次 path graph solve 期间保持有效，直到下一次 `PathGraphBuilder::Reset()`。
- **捕获预算**：为避免全图场景把 WSL 内存打爆，当前 path graph 使用约 `2GB` 的捕获预算，并根据 `SurfaceVertex`、`BSDF`、`ContEdge` 以及部分 light/pixel 映射开销估算最大 surface vertex 数。超过预算后停止继续捕获；`testone Render()` 会保留第一遍普通积分器已经写出的图像，并跳过 path graph 后处理与二次写图。
- **轻量接入**：没有 active builder 时使用 `NoopPathGraphSink`，让 Integrator 代码路径保持统一。

## 核心数据结构

### SurfaceVertex

`SurfaceVertex` 表示路径上的一个表面交点。
- `vertexId`：全局顶点 ID。`0` 是保留值，正常表面顶点从 `1` 开始。
- `depth`：路径深度，通常从相机命中后的第一个表面点开始计。
- `pos`、`geometricNormal`、`shadingNormal`：世界坐标下的位置和法线。
- `wo`：从 hit point 指向相机侧/上一跳的出射方向。
-  L direct 直接聚合得到的，间接聚合就不动了
-  L direct = Σ bsdf(v, e.wi) * cos(v, e.wi) * L_light[e] * misWeight[e] / rho[e]
- 当前实现中 `PathGraphSnapshot::AggregateDirectLighting()` 负责直接聚合并写入 `SurfaceVertex::L_direct`。`bsdf(v, e.wi) * cos(v, e.wi)` 可通过 `SurfaceVertex::bsdf` 在 evaluator 中计算；分母使用论文式 marginal density：对当前 LightEdge，在所属 cluster 内所有 vertex 上计算 `edge.lightPMF * edge.light.PDF_Li(vertexCtx, edge.wi, true)` 并求和。每条边在一个 cluster 内的 marginal density 会缓存一次，避免对每个目标 vertex 重复计算。
- `L indirect 间接聚合得到的，会迭代数次
- 当前实现中 `PathGraphSnapshot::AggregateIndirectLighting()` 负责间接聚合并写入 `SurfaceVertex::L_indirect`。它读取 `ContEdge::L_B` 作为 B 端 radiance 缓存；`bsdf(v, e.wi) * cos(v, e.wi)` 同样可通过 `SurfaceVertex::bsdf` 在 evaluator 中计算；分母使用论文式 marginal density：对当前 ContEdge，在所属 cluster 内所有 vertex 上计算 `vertex.bsdf->PDF(vertex.wo, edge.wi)` 并求和。由于 graph、cluster、BSDF 和方向在迭代间不变，第一次 indirect aggregation 会缓存 `fcos / marginalDensity` transfer weights，后续迭代只更新 `ContEdge::L_B` 并执行乘加。
- `AggregateIndirectLighting()` 内部会先用上一轮的 `vertexB.L_indirect` 将每条 ContEdge 的 `L_B` 更新为 `vertexB.L_direct + previous(vertexB.L_indirect)`，再执行 cluster indirect aggregation。
- `AggregateIndirectLighting()` 在每个 cluster 内做能量 clamping：若间接聚合后的 `Σ L_indirect.Average()` 大于输入 continuation radiance 的 `Σ ContEdge::L_B.Average()`，则把该 cluster 的 `L_indirect` 统一缩放到输入能量的 `0.999` 倍，作为收敛保护。
- `PathGraphSnapshot::FinalGather()` 实现论文中的 decorrelation/final gather：在迭代收敛后先把 `SurfaceVertex::L_direct` 拷贝到 `L_out`，再按 `targetClusterSize = 1` 的语义对每个 vertex 只使用自己的 ContEdge 做一次 singleton indirect 聚合。
- LightEdge 直接指向自己的lightEdge ID
- ContEdge 直接指向自己的ContEdge ID
- `materialId`：材质类型或稳定索引，替代裸指针。
- `bsdfFlags`：BSDF 标志，用于区分反射、透射、delta 等行为。
- `bsdf`：可选指针，指向 path graph 线程数据中保存的 BSDF 副本，供直接/间接聚合 evaluator 使用。

### LightEdge

`LightEdge` 表示从某个表面顶点连接到光源的直接光边。

- `vertexA`：边的表面端点；若调用 `AddLightEdge()` 时为 `0`，sink 会自动填成最近一次加入的 surface vertex。
- `wi`：原始采样 vertex 指向光源采样点的方向。direct aggregation 会对 cluster 内每个目标 vertex 按 `pLight` 重新构造方向，避免把原始 `wi` 错用于其他 vertex。
- `pLight`：光源采样点。有限面积光和 delta-position light 的 marginal density / fcos 必须用目标 vertex 到该采样点的方向复算；delta-direction light 保持原始方向。
- `L_B`：光源侧 radiance，可理解为边的 B 端贡献。
- `misWeight`：原 path tracing 直接光估计中的 MIS 权重。
-  lightID （或者别的信息）用来算pdf
- `isDeltaLight`：标记 delta light，当前主要是保留信息。

### ContEdge

`ContEdge` 表示路径继续采样得到的表面到表面边。

- `vertexA`：相机侧端点，也就是当前表面点。
- `vertexB`：光源侧端点。
- `L_B`：B 端 radiance 缓存字段
- `wi`：原始 `vertexA -> vertexB` 的方向。indirect aggregation 会对 cluster 内每个目标 vertex 用 `vertexB.pos - vertex.pos` 重新构造方向，保证 BSDF PDF 和 fcos 对应同一个 B 端采样点。
- 对 specular continuation edge，普通 `BSDF::PDF()` 处在 delta 测度下会返回 0；当前实现仅让该 edge 在原始 `vertexA` 上以采样时记录的 `edge.pdf` 参与 density，避免把 specular delta 事件扩散到 cluster 内其他 vertex。

### Cluster

- 存聚类内的所有点ID
- 存聚类内所有LightEdge的pdf之和
- 存聚类内所有ContEdge的pdf之和
- 当前实现为了避免全图渲染时 `O(N * N/targetClusterSize)` 的最近中心搜索，使用 Morton/grid 顺序把空间相近的 vertex 线性分组；默认 `targetClusterSize = 16`。这是对论文随机中心聚类的轻量工程化替代。
- 额外保存 `centerVertexId`、`center`、聚类内 vertex/edge 的索引，方便后续 aggregation、propagation 和调试。

### 其他

- `PixelVertexMapEntry` 存储每个 pixel/sample 对应的第一个 surface vertexId、采样波长、camera ray weight 和 filter weight，用于从 `SurfaceVertex::L_out` 回写最终图像。当前 `testone Render()` 会在 path graph 流程结束后清空 Film 像素，按该映射重新 `AddSample()` 并 `WriteImage()` 覆盖输出文件。


## 方向与端点约定

- 所有位置、法线、方向都按世界坐标解释。
- wi 和 wo按照适配brdf的约定来
- 对 continuation edge，`vertexA` 靠近相机，`vertexB` 靠近光源。
- 辐射传播方向是 `vertexB -> vertexA`。
- `ContEdgesIntoVertex(vertexId)` 返回向该顶点传递能量的 continuation edge。按照当前布局，它以 `vertexA` 建邻接表。
- `LightEdgesOfVertex(vertexId)` 返回该顶点发出的直接光连接边。

- `vertexId == 0` 有保留语义。遍历求解时从 `1` 开始。



## 当前实现上的注意点


- `testone Render()` 是当前 path graph solve 的入口：构建 snapshot，执行 direct aggregation、indirect aggregation 迭代和 final gather，然后通过 `PixelVertexMapEntry` 回写 Film 并覆盖输出图像。为避免 staircase 这类全图场景在后处理阶段失控，当前间接聚合迭代数限制为 `min(maxDepth, 8)`；若捕获超过 vertex 预算，则跳过 path graph solve，保留普通积分器输出。


## 推荐使用模式

```cpp
PathGraphSink *pathSink = nullptr;
NoopPathGraphSink noopPathSink;
if (PathGraphBuilder *builder = GetActivePathGraphBuilder())
    pathSink = builder->GetThreadLocalSink();
else
    pathSink = &noopPathSink;



后处理阶段：

```cpp
auto snapshot = pathGraphBuilder.BuildSnapshot();
for (const SurfaceVertex &v : snapshot->Vertices()) {
    // read-only traversal
}
```
