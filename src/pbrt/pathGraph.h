
#pragma once
#include <pbrt/pbrt.h>

#include <pbrt/bsdf.h>
#include <pbrt/base/bxdf.h>
#include <pbrt/lights.h>
#include <pbrt/util/memory.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>
#include <atomic>
#include <memory>
#include <functional>
#include <vector>


namespace pbrt {

struct PathGraphThreadData;

struct SurfaceVertex {
    uint64_t vertexId = 0;
    SampledSpectrum L_out = SampledSpectrum(0.f);
    SampledSpectrum L_direct = SampledSpectrum(0.f);
    SampledSpectrum L_indirect = SampledSpectrum(0.f);
    uint32_t depth = 0;

    Point3f pos;
    Normal3f geometricNormal;
    Normal3f shadingNormal;
    Vector3f wo;

    /// detailed bxdf flags
    BxDFFlags bsdfFlags = BxDFFlags::Unset;

    /// Optional BSDF pointer owned by PathGraphThreadData until the graph solve ends.
    const BSDF *bsdf = nullptr;
};

struct LightEdge {
    uint64_t vertexA = 0;
    Light light;
    Vector3f wi;
    Interaction pLight;
    SampledSpectrum L_B;  /// Emitted radiance along the sampled light ray.

    // PDF for sampling this light-start segment in the emission-ray measure:
    // p(light) * p(position on light) * p(direction from light), i.e. dA_light dω_light.
    // It is not an NEE solid-angle PDF at vertexA and not an area-area segment PDF.
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
};

struct Cluster {
    uint32_t clusterId = 0;

    std::vector<uint32_t> vertexIndices;

    std::vector<uint32_t> lightEdgeIndices;
    std::vector<uint32_t> contEdgeIndices;
};

struct PixelVertexMapEntry {
    Point2i pixel;
    int sampleIndex = 0;
    uint64_t firstVertexId = 0;
    SampledWavelengths lambda;
    SampledSpectrum cameraWeight = SampledSpectrum(1.f);
    Float filterWeight = 1;
};

class PathGraphSink {
  public:
    explicit PathGraphSink(PathGraphThreadData *data = nullptr,
                           std::atomic<uint64_t> *nextVertexId = nullptr,
                           uint64_t maxVertexCount = 0,
                           std::atomic<bool> *truncated = nullptr)
        : data(data),
          nextVertexId(nextVertexId),
          maxVertexCount(maxVertexCount),
          truncated(truncated) {}

    uint64_t AddSurfaceVertex(SurfaceVertex vertex);
    void AddLightEdge(LightEdge edge);
    void AddContEdge(ContEdge edge);
    void BeginPixelSample(Point2i pixel, int sampleIndex,
                          const SampledWavelengths &lambda,
                          SampledSpectrum cameraWeight, Float filterWeight);
    void EndPixelSample();
    ScratchBuffer *GetBSDFScratchBuffer();

  private:
    PathGraphThreadData *data = nullptr;
    std::atomic<uint64_t> *nextVertexId = nullptr;
    uint64_t maxVertexCount = 0;
    std::atomic<bool> *truncated = nullptr;
};

class PathGraphSnapshot {
  public:
    using DirectBSDFEvaluator =
        std::function<SampledSpectrum(const SurfaceVertex &, Vector3f wi)>;
    using IndirectFcosEvaluator =
        std::function<SampledSpectrum(const SurfaceVertex &, const ContEdge &)>;

    PathGraphSnapshot(std::vector<SurfaceVertex> vertices,
                      std::vector<LightEdge> lightEdges,
                      std::vector<ContEdge> contEdges,
                      std::vector<PixelVertexMapEntry> pixelVertexMap,
                      uint32_t targetClusterSize = 16);

    pstd::span<const SurfaceVertex> Vertices() const { return vertices; }
    pstd::span<const LightEdge> LightEdges() const { return lightEdges; }
    pstd::span<const ContEdge> ContEdges() const { return contEdges; }
    pstd::span<const Cluster> Clusters() const { return clusters; }
    pstd::span<const PixelVertexMapEntry> PixelVertexMap() const {
        return pixelVertexMap;
    }

    void AggregateDirectLighting(const DirectBSDFEvaluator &bsdfEvaluator);
    void AggregateIndirectLighting(const IndirectFcosEvaluator &fcosEvaluator);
    void FinalGather(const IndirectFcosEvaluator &indirectFcosEvaluator);

  private:
    void BuildClusters(uint32_t targetClusterSize);
    uint32_t VertexIndexFromId(uint64_t vertexId) const;
    void BuildIndirectTransferWeights(const IndirectFcosEvaluator &fcosEvaluator);

    std::vector<SurfaceVertex> vertices;
    std::vector<LightEdge> lightEdges;
    std::vector<ContEdge> contEdges;
    std::vector<PixelVertexMapEntry> pixelVertexMap;
    std::vector<Cluster> clusters;
    std::vector<uint32_t> clusterByVertexIndex;
    std::vector<uint32_t> contEdgeTargetVertexIndices;
    std::vector<std::vector<SampledSpectrum>> indirectTransferWeights;
    std::vector<SampledSpectrum> previousIndirect;
    bool indirectTransferWeightsValid = false;
};

class PathGraphBuilder {
  public:
    PathGraphBuilder();
    ~PathGraphBuilder();

    PathGraphSink *GetThreadLocalSink();
    std::unique_ptr<PathGraphSnapshot> BuildSnapshot(uint32_t targetClusterSize = 16);
    void Reset();
    bool WasTruncated() const;

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
