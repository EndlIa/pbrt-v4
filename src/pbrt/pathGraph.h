
#pragma once
#include <pbrt/pbrt.h>

#include <pbrt/base/bxdf.h>
#include <pbrt/util/pstd.h>
#include <pbrt/util/spectrum.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>
#include <memory>


namespace pbrt {

struct SurfaceVertex {
    uint64_t vertexId = 0;
    uint64_t pathId = 0;
    uint32_t depth = 0;

    Point3f pos;
    Normal3f geometricNormal;
    Normal3f shadingNormal;
    Vector3f wo;

    /// use type index to avoid storing raw BSDF pointers.
    uint32_t bsdfType = 0;
    int32_t materialId = -1;
    int32_t shapeId = -1;

    /// detailed bxdf flags
    BxDFFlags bsdfFlags = BxDFFlags::Unset;
    bool bsdfRegularized = false;
};

struct LightEdge {
    uint64_t fromVertexId = 0;
    Vector3f wi;
    SampledSpectrum L;  ///emit radiance
    // ef = AbsDot(shadingNormal, wi)
    //    = radiance transfer coefficient at the edge
    SampledSpectrum ef;
    Float pdf = 0;
    Float misWeight = 1;

    ///detailed flags
    bool isDeltaLight = false;
};

struct ContEdge {
    uint64_t fromVertexId = 0;
    uint64_t toVertexId = 0;
    Vector3f wi;

    // ef = BSDF(wo, wi) * AbsDot(shadingNormal, wi)
    //    = radiance transfer coefficient at the edge
    SampledSpectrum ef;

    Float pdf = 0;
    Float rrQ = 0;  // Russian roulette die probability
    
    /// detailed flags
    BxDFFlags flags = BxDFFlags::Unset;
    Float eta = 1;  // relative refractive index
};

struct NeighborEdge {
    uint64_t a = 0;
    uint64_t b = 0;
    int32_t clusterId = -1;

    /// maybe useless ...
    Float distance = 0;
    Float weight = 1;
};

// Read-only snapshot interface 
class PathGraphSnapshot {
  public:
    virtual ~PathGraphSnapshot() = default;
    virtual pstd::span<const SurfaceVertex> Vertices() const = 0;
    virtual pstd::span<const LightEdge> LightEdges() const = 0;
    virtual pstd::span<const ContEdge> ContEdges() const = 0;
    virtual pstd::span<const NeighborEdge> NeighborEdges() const = 0;
};

// Capture interface: called by Integrator 
class PathGraphSink {
  public:
    virtual ~PathGraphSink() = default;
    virtual void BeginPath(uint64_t pathId) = 0;
    virtual void AddSurfaceVertex(const SurfaceVertex &v) = 0;
    virtual void AddLightEdge(const LightEdge &e) = 0;
    virtual void AddContEdge(const ContEdge &e) = 0;
    virtual void EndPath(uint64_t pathId) = 0;
};

class PathGraphBuilder {
  public:
    virtual ~PathGraphBuilder() = default;
    virtual PathGraphSink *GetThreadLocalSink() = 0;
    virtual std::shared_ptr<const PathGraphSnapshot> BuildSnapshot() = 0;
};

// Zero-overhead placeholder when capture is disabled
class NoopPathGraphSink : public PathGraphSink {
  public:
    void BeginPath(uint64_t) override {}
    void AddSurfaceVertex(const SurfaceVertex &) override {}
    void AddLightEdge(const LightEdge &) override {}
    void AddContEdge(const ContEdge &) override {}
    void EndPath(uint64_t) override {}
};

}  