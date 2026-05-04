
#pragma once
#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>
#include <atomic>
#include <memory>
#include <vector>


namespace pbrt {

struct SurfaceVertex {
    uint64_t vertexId = 0;
    SampledSpectrum L_in = SampledSpectrum(0.f);
    SampledSpectrum L_out = SampledSpectrum(0.f);
    uint32_t depth = 0;

    Point3f pos;
    Normal3f geometricNormal;
    Normal3f shadingNormal;
    Vector3f wo;

    /// use type index to avoid storing raw BSDF pointers.
    int32_t materialId = -1;

    /// detailed bxdf flags
    BxDFFlags bsdfFlags = BxDFFlags::Unset;
};

struct LightEdge {
    uint64_t vertexA = 0;
    Vector3f wi;
    SampledSpectrum L_B;  ///emit radiance

    Float pdf = 0;
    Float misWeight = 1;

    /// maybe useless ...
    bool isDeltaLight = false;
};

struct ContEdge {
    uint64_t vertexA = 0;
    uint64_t vertexB = 0;
    SampledSpectrum L_B = SampledSpectrum(0.f);
    Vector3f wi;

    Float pdf = 0;
    
    /// maybe useless ...
    BxDFFlags flags = BxDFFlags::Unset;
    Float eta = 1;  // relative refractive index
};

struct Cluster {
    uint32_t clusterId = 0;
    uint64_t centerVertexId = 0;
    Point3f center;

    std::vector<uint64_t> vertexIds;
    std::vector<uint32_t> vertexIndices;

    std::vector<uint32_t> lightEdgeIndices;
    std::vector<uint32_t> contEdgeIndices;

    Float lightEdgePdfSum = 0;
    Float contEdgePdfSum = 0;
};

struct PathGraphThreadData {
    std::vector<SurfaceVertex> vertices;
    std::vector<LightEdge> lightEdges;
    std::vector<ContEdge> contEdges;
    uint64_t lastSurfaceVertexId = 0;

    void Clear();
};

class PathGraphSink {
  public:
    explicit PathGraphSink(PathGraphThreadData *data = nullptr,
                           std::atomic<uint64_t> *nextVertexId = nullptr)
        : data(data), nextVertexId(nextVertexId) {}

    uint64_t AddSurfaceVertex(SurfaceVertex vertex);
    void AddLightEdge(LightEdge edge);
    void AddContEdge(ContEdge edge);
    uint64_t LastSurfaceVertexId() const;

  private:
    PathGraphThreadData *data = nullptr;
    std::atomic<uint64_t> *nextVertexId = nullptr;
};

class NoopPathGraphSink : public PathGraphSink {
  public:
    NoopPathGraphSink() : PathGraphSink(nullptr, nullptr) {}
};

class PathGraphSnapshot {
  public:
    PathGraphSnapshot(std::vector<SurfaceVertex> vertices,
                      std::vector<LightEdge> lightEdges,
                      std::vector<ContEdge> contEdges,
                      uint32_t targetClusterSize = 16);

    pstd::span<const SurfaceVertex> Vertices() const { return vertices; }
    pstd::span<const LightEdge> LightEdges() const { return lightEdges; }
    pstd::span<const ContEdge> ContEdges() const { return contEdges; }
    pstd::span<const Cluster> Clusters() const { return clusters; }

  private:
    void BuildClusters(uint32_t targetClusterSize);

    std::vector<SurfaceVertex> vertices;
    std::vector<LightEdge> lightEdges;
    std::vector<ContEdge> contEdges;
    std::vector<Cluster> clusters;
};

class PathGraphBuilder {
  public:
    PathGraphBuilder();
    ~PathGraphBuilder();

    PathGraphSink *GetThreadLocalSink();
    std::unique_ptr<PathGraphSnapshot> BuildSnapshot(uint32_t targetClusterSize = 16);
    void Reset();

  private:
    class Impl;
    std::unique_ptr<Impl> impl;
};

class ScopedPathGraphBuilder {
  public:
    explicit ScopedPathGraphBuilder(PathGraphBuilder *builder);
    ~ScopedPathGraphBuilder();

    ScopedPathGraphBuilder(const ScopedPathGraphBuilder &) = delete;
    ScopedPathGraphBuilder &operator=(const ScopedPathGraphBuilder &) = delete;

  private:
    PathGraphBuilder *previous = nullptr;
};

PathGraphBuilder *GetActivePathGraphBuilder();

}
