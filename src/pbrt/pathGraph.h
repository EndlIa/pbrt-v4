
#ifndef PBRT_PATHGRAPH_H
#define PBRT_PATHGRAPH_H

#include <pbrt/pbrt.h>

#include <pbrt/util/pstd.h>
#include <pbrt/util/vecmath.h>

#include <cstdint>
#include <memory>

namespace pbrt {

using PathId = uint64_t;
using VertexId = uint64_t;

struct RGB3f {
    Float r = 0;
    Float g = 0;
    Float b = 0;
};

/// x_j: shading point j
struct SurfaceVertexRecord {
    VertexId vertexId = 0;
    PathId pathId = 0;
    uint32_t depth = 0;

    Point3f pos;
    Normal3f normal;
    Vector3f wo;

    RGB3f albedo;

    // Stable identifiers; avoid storing raw BSDF pointers.
    uint32_t bsdfType = 0;
    int32_t materialId = -1;
    int32_t shapeId = -1;

    // Continuation-sampling data (leave as 0 if unavailable).
    Float pdfBsdf = 0;
};

  // NEE connection
struct LightEdgeRecord {
    VertexId fromVertexId = 0;
    Vector3f wi;

    RGB3f L;
    // fCos = BSDF(wi) * |n . wi| for direct reuse in post-processing.
    RGB3f fCos;

    Float pdfLight = 0;
    Float pdfSelectLight = 1;
    Float pdfBsdfForWi = 0;

    // Keep fixed at 1 in the SimplePath stage.
    Float misWeight = 1;
    uint8_t misMode = 0;
};

  // Path continuation edge
struct ContEdgeRecord {
    VertexId fromVertexId = 0;
    VertexId toVertexId = 0;
    Vector3f wi;

    Float pdfBsdf = 0;

    // If RR is disabled, use q=1 and invQ=1.
    Float q = 1;
    Float invQ = 1;
};

  // Adjacency edge after clustering
struct NeighborEdgeRecord {
    VertexId a = 0;
    VertexId b = 0;
    int32_t clusterId = -1;

    Float distance = 0;
    Float weight = 1;
};

// Read-only snapshot interface for post-processing
class PathGraphSnapshot {
  public:
    virtual ~PathGraphSnapshot() = default;

    virtual pstd::span<const SurfaceVertexRecord> Vertices() const = 0;
    virtual pstd::span<const LightEdgeRecord> LightEdges() const = 0;
    virtual pstd::span<const ContEdgeRecord> ContEdges() const = 0;
    virtual pstd::span<const NeighborEdgeRecord> NeighborEdges() const = 0;
};

// Capture interface: called by the Integrator in the main loop
class IPathGraphSink {
  public:
    virtual ~IPathGraphSink() = default;

    virtual void BeginPath(PathId pathId) = 0;
    virtual void AddSurfaceVertex(const SurfaceVertexRecord &v) = 0;
    virtual void AddLightEdge(const LightEdgeRecord &e) = 0;
    virtual void AddContEdge(const ContEdgeRecord &e) = 0;
    virtual void EndPath(PathId pathId) = 0;
};

// Builder interface: implementation may use TLS buffers and merge
class PathGraphBuilder {
  public:
    virtual ~PathGraphBuilder() = default;

    virtual IPathGraphSink *GetThreadLocalSink() = 0;
    virtual std::shared_ptr<const PathGraphSnapshot> BuildSnapshot() = 0;
};

// Zero-overhead placeholder when capture is disabled
class NoopPathGraphSink : public IPathGraphSink {
  public:
    void BeginPath(PathId) override {}
    void AddSurfaceVertex(const SurfaceVertexRecord &) override {}
    void AddLightEdge(const LightEdgeRecord &) override {}
    void AddContEdge(const ContEdgeRecord &) override {}
    void EndPath(PathId) override {}
};

}  

#endif  // PBRT_PATHGRAPH_H