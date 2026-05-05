#include <pbrt/pathGraph.h>

#include <pbrt/util/parallel.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <limits>
#include <utility>

namespace pbrt {

namespace {

std::atomic<PathGraphBuilder *> activePathGraphBuilder{nullptr};

constexpr uint64_t kMaxCapturedVertices = 2000000;

LightSampleContext LightContext(const SurfaceVertex &vertex) {
    return LightSampleContext(Point3fi(vertex.pos), vertex.geometricNormal,
                              vertex.shadingNormal);
}

Float DirectMarginalDensity(const LightEdge &edge, const Cluster &cluster,
                            pstd::span<const SurfaceVertex> vertices) {
    if (!edge.light || edge.lightPMF <= 0)
        return 0;

    Float density = 0;
    for (uint32_t vertexIndex : cluster.vertexIndices) {
        const SurfaceVertex &vertex = vertices[vertexIndex];
        density += edge.lightPMF * edge.light.PDF_Li(LightContext(vertex), edge.wi, true);
    }
    return density;
}

Float IndirectMarginalDensity(const ContEdge &edge, const Cluster &cluster,
                              pstd::span<const SurfaceVertex> vertices) {
    Float density = 0;
    for (uint32_t vertexIndex : cluster.vertexIndices) {
        const SurfaceVertex &vertex = vertices[vertexIndex];
        if (!vertex.bsdf)
            continue;
        density += vertex.bsdf->PDF(vertex.wo, edge.wi);
    }
    return density;
}

uint64_t MortonCode(Point3f p, Point3f pMin, Vector3f extent, uint32_t gridResolution) {
    auto quantize = [&](Float v, Float minValue, Float width) {
        if (width <= 0)
            return uint32_t(0);
        Float normalized = Clamp((v - minValue) / width, 0, Float(0.999999));
        return std::min<uint32_t>(gridResolution - 1, normalized * gridResolution);
    };

    uint32_t x = quantize(p.x, pMin.x, extent.x);
    uint32_t y = quantize(p.y, pMin.y, extent.y);
    uint32_t z = quantize(p.z, pMin.z, extent.z);
    return uint64_t(x) + uint64_t(gridResolution) * (uint64_t(y) +
                                                     uint64_t(gridResolution) * z);
}

}  // namespace

void PathGraphThreadData::Clear() {
    vertices.clear();
    lightEdges.clear();
    contEdges.clear();
    pixelVertexMap.clear();
    bsdfScratchBuffer.Reset();
    bsdfs.clear();
    lastSurfaceVertexId = 0;
    currentFirstVertexId = 0;
    hasCurrentPixelSample = false;
}

uint64_t PathGraphSink::AddSurfaceVertex(SurfaceVertex vertex) {
    if (!data || !nextVertexId)
        return 0;

    vertex.vertexId = nextVertexId->fetch_add(1, std::memory_order_relaxed);
    if (maxVertexCount > 0 && vertex.vertexId > maxVertexCount) {
        if (truncated)
            truncated->store(true, std::memory_order_relaxed);
        data->lastSurfaceVertexId = 0;
        return 0;
    }

    if (vertex.bsdf) {
        data->bsdfs.push_back(*vertex.bsdf);
        vertex.bsdf = &data->bsdfs.back();
    }
    data->lastSurfaceVertexId = vertex.vertexId;
    data->vertices.push_back(vertex);
    if (data->hasCurrentPixelSample && data->currentFirstVertexId == 0) {
        data->currentFirstVertexId = vertex.vertexId;
        data->pixelVertexMap.push_back(
            {data->currentPixel, data->currentSampleIndex, vertex.vertexId,
             data->currentLambda, data->currentCameraWeight, data->currentFilterWeight});
    }
    return vertex.vertexId;
}

void PathGraphSink::AddLightEdge(LightEdge edge) {
    if (!data)
        return;

    if (edge.vertexA == 0)
        edge.vertexA = data->lastSurfaceVertexId;
    if (edge.vertexA == 0)
        return;
    data->lightEdges.push_back(edge);
}

void PathGraphSink::AddContEdge(ContEdge edge) {
    if (!data)
        return;

    if (edge.vertexA == 0)
        edge.vertexA = data->lastSurfaceVertexId;
    if (edge.vertexA == 0 || edge.vertexB == 0)
        return;
    data->contEdges.push_back(edge);
}

uint64_t PathGraphSink::LastSurfaceVertexId() const {
    return data ? data->lastSurfaceVertexId : 0;
}

void PathGraphSink::BeginPixelSample(Point2i pixel, int sampleIndex,
                                     const SampledWavelengths &lambda,
                                     SampledSpectrum cameraWeight, Float filterWeight) {
    if (!data)
        return;

    data->currentPixel = pixel;
    data->currentSampleIndex = sampleIndex;
    data->currentLambda = lambda;
    data->currentCameraWeight = cameraWeight;
    data->currentFilterWeight = filterWeight;
    data->currentFirstVertexId = 0;
    data->hasCurrentPixelSample = true;
}

void PathGraphSink::EndPixelSample() {
    if (!data)
        return;

    data->currentFirstVertexId = 0;
    data->hasCurrentPixelSample = false;
}

ScratchBuffer *PathGraphSink::GetBSDFScratchBuffer() {
    if (truncated && truncated->load(std::memory_order_relaxed))
        return nullptr;
    if (nextVertexId && maxVertexCount > 0 &&
        nextVertexId->load(std::memory_order_relaxed) > maxVertexCount)
        return nullptr;
    return data ? &data->bsdfScratchBuffer : nullptr;
}

PathGraphSnapshot::PathGraphSnapshot(std::vector<SurfaceVertex> vertices,
                                     std::vector<LightEdge> lightEdges,
                                     std::vector<ContEdge> contEdges,
                                     std::vector<PixelVertexMapEntry> pixelVertexMap,
                                     uint32_t targetClusterSize)
    : vertices(std::move(vertices)),
      lightEdges(std::move(lightEdges)),
      contEdges(std::move(contEdges)),
      pixelVertexMap(std::move(pixelVertexMap)) {
    BuildClusters(targetClusterSize);
}

void PathGraphSnapshot::BuildClusters(uint32_t targetClusterSize) {
    clusters.clear();
    clusterByVertexIndex.clear();
    if (vertices.empty())
        return;

    targetClusterSize = std::max<uint32_t>(1, targetClusterSize);
    uint32_t nClusters = std::max<uint32_t>(
        1, (uint32_t)((vertices.size() + targetClusterSize - 1) / targetClusterSize));
    nClusters = std::min<uint32_t>(nClusters, (uint32_t)vertices.size());

    Point3f pMin(std::numeric_limits<Float>::infinity(),
                 std::numeric_limits<Float>::infinity(),
                 std::numeric_limits<Float>::infinity());
    Point3f pMax(-std::numeric_limits<Float>::infinity(),
                 -std::numeric_limits<Float>::infinity(),
                 -std::numeric_limits<Float>::infinity());
    for (const SurfaceVertex &vertex : vertices) {
        pMin.x = std::min(pMin.x, vertex.pos.x);
        pMin.y = std::min(pMin.y, vertex.pos.y);
        pMin.z = std::min(pMin.z, vertex.pos.z);
        pMax.x = std::max(pMax.x, vertex.pos.x);
        pMax.y = std::max(pMax.y, vertex.pos.y);
        pMax.z = std::max(pMax.z, vertex.pos.z);
    }

    Vector3f extent = pMax - pMin;
    uint32_t gridResolution =
        std::max<uint32_t>(1, std::ceil(std::cbrt((Float)nClusters)));

    std::vector<uint32_t> sortedVertexIndices(vertices.size());
    for (uint32_t i = 0; i < sortedVertexIndices.size(); ++i)
        sortedVertexIndices[i] = i;
    std::sort(sortedVertexIndices.begin(), sortedVertexIndices.end(),
              [&](uint32_t a, uint32_t b) {
                  uint64_t mortonA =
                      MortonCode(vertices[a].pos, pMin, extent, gridResolution);
                  uint64_t mortonB =
                      MortonCode(vertices[b].pos, pMin, extent, gridResolution);
                  if (mortonA != mortonB)
                      return mortonA < mortonB;
                  return vertices[a].vertexId < vertices[b].vertexId;
              });

    clusters.resize(nClusters);
    clusterByVertexIndex.resize(vertices.size(), 0);
    for (uint32_t clusterId = 0; clusterId < clusters.size(); ++clusterId) {
        uint32_t begin = clusterId * targetClusterSize;
        uint32_t end = std::min<uint32_t>(begin + targetClusterSize,
                                          sortedVertexIndices.size());
        if (begin >= end)
            break;

        const SurfaceVertex &centerVertex = vertices[sortedVertexIndices[begin]];
        Cluster &cluster = clusters[clusterId];
        cluster.clusterId = clusterId;
        cluster.centerVertexId = centerVertex.vertexId;
        cluster.center = centerVertex.pos;

        for (uint32_t i = begin; i < end; ++i) {
            uint32_t vertexIndex = sortedVertexIndices[i];
            const SurfaceVertex &vertex = vertices[vertexIndex];
            cluster.vertexIds.push_back(vertex.vertexId);
            cluster.vertexIndices.push_back(vertexIndex);
            clusterByVertexIndex[vertexIndex] = clusterId;
        }
    }

    for (uint32_t edgeIndex = 0; edgeIndex < lightEdges.size(); ++edgeIndex) {
        uint32_t vertexIndex = VertexIndexFromId(lightEdges[edgeIndex].vertexA);
        if (vertexIndex == uint32_t(-1))
            continue;
        Cluster &cluster = clusters[clusterByVertexIndex[vertexIndex]];
        cluster.lightEdgeIndices.push_back(edgeIndex);
        cluster.lightEdgePdfSum += lightEdges[edgeIndex].pdf;
    }

    for (uint32_t edgeIndex = 0; edgeIndex < contEdges.size(); ++edgeIndex) {
        uint32_t vertexIndex = VertexIndexFromId(contEdges[edgeIndex].vertexA);
        if (vertexIndex == uint32_t(-1))
            continue;
        Cluster &cluster = clusters[clusterByVertexIndex[vertexIndex]];
        cluster.contEdgeIndices.push_back(edgeIndex);
        cluster.contEdgePdfSum += contEdges[edgeIndex].pdf;
    }
}

uint32_t PathGraphSnapshot::VertexIndexFromId(uint64_t vertexId) const {
    if (vertexId == 0 || vertexId > vertices.size())
        return uint32_t(-1);
    uint32_t index = (uint32_t)(vertexId - 1);
    if (index >= vertices.size() || vertices[index].vertexId != vertexId)
        return uint32_t(-1);
    return index;
}

void PathGraphSnapshot::AggregateDirectLighting(
    const DirectFcosEvaluator &fcosEvaluator) {
    for (SurfaceVertex &vertex : vertices)
        vertex.L_direct = SampledSpectrum(0.f);

    if (!fcosEvaluator)
        return;

    pstd::span<const SurfaceVertex> vertexSpan(vertices);
    for (const Cluster &cluster : clusters) {
        std::vector<Float> marginalDensities(cluster.lightEdgeIndices.size(), 0);
        for (uint32_t i = 0; i < cluster.lightEdgeIndices.size(); ++i) {
            const LightEdge &edge = lightEdges[cluster.lightEdgeIndices[i]];
            if (edge.pdf > 0)
                marginalDensities[i] = DirectMarginalDensity(edge, cluster, vertexSpan);
        }

        for (uint32_t vertexIndex : cluster.vertexIndices) {
            SurfaceVertex &vertex = vertices[vertexIndex];
            SampledSpectrum Ld(0.f);

            for (uint32_t i = 0; i < cluster.lightEdgeIndices.size(); ++i) {
                uint32_t edgeIndex = cluster.lightEdgeIndices[i];
                const LightEdge &edge = lightEdges[edgeIndex];
                if (edge.pdf <= 0)
                    continue;

                Float marginalDensity = marginalDensities[i];
                if (marginalDensity <= 0)
                    continue;

                SampledSpectrum fcos = fcosEvaluator(vertex, edge);
                if (!fcos)
                    continue;

                Ld += fcos * edge.L_B * edge.misWeight / marginalDensity;
            }

            vertex.L_direct = Ld;
        }
    }
}

void PathGraphSnapshot::AggregateIndirectLighting(
    const IndirectFcosEvaluator &fcosEvaluator) {
    std::vector<SampledSpectrum> previousIndirect(vertices.size(), SampledSpectrum(0.f));
    for (uint32_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex)
        previousIndirect[vertexIndex] = vertices[vertexIndex].L_indirect;

    for (SurfaceVertex &vertex : vertices)
        vertex.L_indirect = SampledSpectrum(0.f);

    if (!fcosEvaluator)
        return;

    for (ContEdge &edge : contEdges) {
        uint32_t vertexIndex = VertexIndexFromId(edge.vertexB);
        if (vertexIndex == uint32_t(-1)) {
            edge.L_B = SampledSpectrum(0.f);
            continue;
        }

        const SurfaceVertex &vertexB = vertices[vertexIndex];
        edge.L_B = vertexB.L_direct + previousIndirect[vertexIndex];
    }

    pstd::span<const SurfaceVertex> vertexSpan(vertices);
    for (const Cluster &cluster : clusters) {
        std::vector<Float> marginalDensities(cluster.contEdgeIndices.size(), 0);
        for (uint32_t i = 0; i < cluster.contEdgeIndices.size(); ++i) {
            const ContEdge &edge = contEdges[cluster.contEdgeIndices[i]];
            if (edge.pdf > 0 && edge.L_B)
                marginalDensities[i] = IndirectMarginalDensity(edge, cluster, vertexSpan);
        }

        Float inputEnergy = 0;
        for (uint32_t edgeIndex : cluster.contEdgeIndices)
            inputEnergy += contEdges[edgeIndex].L_B.Average();

        for (uint32_t vertexIndex : cluster.vertexIndices) {
            SurfaceVertex &vertex = vertices[vertexIndex];
            SampledSpectrum Li(0.f);

            for (uint32_t i = 0; i < cluster.contEdgeIndices.size(); ++i) {
                uint32_t edgeIndex = cluster.contEdgeIndices[i];
                const ContEdge &edge = contEdges[edgeIndex];
                if (edge.pdf <= 0 || !edge.L_B)
                    continue;

                Float marginalDensity = marginalDensities[i];
                if (marginalDensity <= 0)
                    continue;

                SampledSpectrum fcos = fcosEvaluator(vertex, edge);
                if (!fcos)
                    continue;

                Li += fcos * edge.L_B / marginalDensity;
            }

            vertex.L_indirect = Li;
        }

        Float outputEnergy = 0;
        for (uint32_t vertexIndex : cluster.vertexIndices)
            outputEnergy += vertices[vertexIndex].L_indirect.Average();

        if (inputEnergy > 0 && outputEnergy > inputEnergy) {
            Float scale = (inputEnergy / outputEnergy) * 0.999f;
            for (uint32_t vertexIndex : cluster.vertexIndices)
                vertices[vertexIndex].L_indirect *= scale;
        }
    }
}

void PathGraphSnapshot::FinalGather(
    const IndirectFcosEvaluator &indirectFcosEvaluator) {
    for (SurfaceVertex &vertex : vertices)
        vertex.L_out = vertex.L_direct;

    if (!indirectFcosEvaluator)
        return;

    for (const ContEdge &edge : contEdges) {
        uint32_t vertexIndex = VertexIndexFromId(edge.vertexA);
        if (vertexIndex == uint32_t(-1))
            continue;

        if (!edge.L_B)
            continue;

        SurfaceVertex &vertex = vertices[vertexIndex];
        Float marginalDensity = vertex.bsdf ? vertex.bsdf->PDF(vertex.wo, edge.wi) : 0;
        if (marginalDensity <= 0)
            continue;

        SampledSpectrum fcos = indirectFcosEvaluator(vertex, edge);
        if (!fcos)
            continue;

        vertex.L_out += fcos * edge.L_B / marginalDensity;
    }
}

class PathGraphBuilder::Impl {
  public:
    ThreadLocal<PathGraphThreadData> threadData;
    ThreadLocal<PathGraphSink> sinks;
    std::atomic<uint64_t> nextVertexId{1};
    std::atomic<bool> truncated{false};

    Impl()
        : threadData([]() { return PathGraphThreadData(); }),
          sinks([this]() {
              return PathGraphSink(&threadData.Get(), &nextVertexId,
                                   kMaxCapturedVertices, &truncated);
          }) {}
};

PathGraphBuilder::PathGraphBuilder() : impl(std::make_unique<Impl>()) {}

PathGraphBuilder::~PathGraphBuilder() = default;

PathGraphSink *PathGraphBuilder::GetThreadLocalSink() {
    return &impl->sinks.Get();
}

std::unique_ptr<PathGraphSnapshot> PathGraphBuilder::BuildSnapshot(
    uint32_t targetClusterSize) {
    std::vector<SurfaceVertex> vertices;
    std::vector<LightEdge> lightEdges;
    std::vector<ContEdge> contEdges;
    std::vector<PixelVertexMapEntry> pixelVertexMap;

    impl->threadData.ForAll([&](const PathGraphThreadData &data) {
        vertices.insert(vertices.end(), data.vertices.begin(), data.vertices.end());
        lightEdges.insert(lightEdges.end(), data.lightEdges.begin(), data.lightEdges.end());
        contEdges.insert(contEdges.end(), data.contEdges.begin(), data.contEdges.end());
        pixelVertexMap.insert(pixelVertexMap.end(), data.pixelVertexMap.begin(),
                              data.pixelVertexMap.end());
    });

    std::sort(vertices.begin(), vertices.end(),
              [](const SurfaceVertex &a, const SurfaceVertex &b) {
                  return a.vertexId < b.vertexId;
              });

    return std::make_unique<PathGraphSnapshot>(std::move(vertices), std::move(lightEdges),
                                               std::move(contEdges), std::move(pixelVertexMap),
                                               targetClusterSize);
}

void PathGraphBuilder::Reset() {
    impl->threadData.ForAll([](PathGraphThreadData &data) { data.Clear(); });
    impl->nextVertexId.store(1, std::memory_order_relaxed);
    impl->truncated.store(false, std::memory_order_relaxed);
}

bool PathGraphBuilder::WasTruncated() const {
    return impl->truncated.load(std::memory_order_relaxed);
}

ScopedPathGraphBuilder::ScopedPathGraphBuilder(PathGraphBuilder *builder) {
    previous = activePathGraphBuilder.exchange(builder, std::memory_order_acq_rel);
}

ScopedPathGraphBuilder::~ScopedPathGraphBuilder() {
    activePathGraphBuilder.store(previous, std::memory_order_release);
}

PathGraphBuilder *GetActivePathGraphBuilder() {
    return activePathGraphBuilder.load(std::memory_order_acquire);
}

}  // namespace pbrt
