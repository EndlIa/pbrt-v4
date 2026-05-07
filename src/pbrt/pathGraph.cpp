#include <pbrt/pathGraph.h>

#include <pbrt/ray.h>
#include <pbrt/util/parallel.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <deque>
#include <limits>
#include <utility>

namespace pbrt {

namespace {

std::atomic<PathGraphBuilder *> activePathGraphBuilder{nullptr};

constexpr uint64_t kPathGraphCaptureBudgetBytes = 2ull * 1024 * 1024 * 1024;
constexpr uint64_t kEstimatedBytesPerCapturedVertex =
    sizeof(SurfaceVertex) + sizeof(BSDF) + sizeof(ContEdge) +
    sizeof(LightEdge) / 4 + sizeof(PixelVertexMapEntry) / 4;
constexpr uint64_t kMaxCapturedVertices =
    kPathGraphCaptureBudgetBytes / kEstimatedBytesPerCapturedVertex;

}  // namespace

struct PathGraphThreadData {
    std::vector<SurfaceVertex> vertices;
    std::vector<LightEdge> lightEdges;
    std::vector<ContEdge> contEdges;
    std::vector<PixelVertexMapEntry> pixelVertexMap;
    ScratchBuffer bsdfScratchBuffer;
    std::deque<BSDF> bsdfs;
    uint64_t lastSurfaceVertexId = 0;
    Point2i currentPixel;
    int currentSampleIndex = 0;
    SampledWavelengths currentLambda;
    SampledSpectrum currentCameraWeight = SampledSpectrum(1.f);
    Float currentFilterWeight = 1;
    uint64_t currentFirstVertexId = 0;
    bool hasCurrentPixelSample = false;

    void Clear();
};

namespace {

LightSampleContext LightContext(const SurfaceVertex &vertex) {
    return LightSampleContext(Point3fi(vertex.pos), vertex.geometricNormal,
                              vertex.shadingNormal);
}

LightSampleContext OffsetLightContext(const SurfaceVertex &vertex) {
    LightSampleContext ctx = LightContext(vertex);
    if (IsReflective(vertex.bsdfFlags) && !IsTransmissive(vertex.bsdfFlags))
        ctx.pi = Point3fi(OffsetRayOrigin(Point3fi(vertex.pos), vertex.geometricNormal,
                                          vertex.wo));
    else if (IsTransmissive(vertex.bsdfFlags) && !IsReflective(vertex.bsdfFlags))
        ctx.pi = Point3fi(OffsetRayOrigin(Point3fi(vertex.pos), vertex.geometricNormal,
                                          -vertex.wo));
    return ctx;
}

Vector3f LightDirectionForVertex(const LightEdge &edge, const SurfaceVertex &vertex) {
    if (edge.light && (edge.light.Type() == LightType::Area ||
                       edge.light.Type() == LightType::DeltaPosition)) {
        Vector3f wi = edge.pLight.p() - vertex.pos;
        if (LengthSquared(wi) > 0)
            return Normalize(wi);
    }
    return edge.wi;
}

Vector3f ContinuationDirectionForVertex(const ContEdge &edge,
                                        const SurfaceVertex &vertex,
                                        pstd::span<const SurfaceVertex> vertices) {
    uint32_t vertexBIndex = edge.vertexB == 0 ? uint32_t(-1) : uint32_t(edge.vertexB - 1);
    if (vertexBIndex < vertices.size() && vertices[vertexBIndex].vertexId == edge.vertexB) {
        Vector3f wi = vertices[vertexBIndex].pos - vertex.pos;
        if (LengthSquared(wi) > 0)
            return Normalize(wi);
    }
    return edge.wi;
}

ContEdge ContEdgeForVertex(const ContEdge &edge, const SurfaceVertex &vertex,
                           pstd::span<const SurfaceVertex> vertices) {
    ContEdge adjusted = edge;
    adjusted.wi = ContinuationDirectionForVertex(edge, vertex, vertices);
    return adjusted;
}

Float DirectMarginalDensity(const LightEdge &edge, const Cluster &cluster,
                            pstd::span<const SurfaceVertex> vertices,
                            pstd::span<Vector3f> cachedDirections,
                            uint32_t edgeOffset, uint32_t nEdges) {
    if (!edge.light || edge.lightPMF <= 0)
        return 0;

    Float density = 0;
    for (uint32_t vertexOffset = 0; vertexOffset < cluster.vertexIndices.size();
         ++vertexOffset) {
        const SurfaceVertex &vertex = vertices[cluster.vertexIndices[vertexOffset]];
        Vector3f wi = LightDirectionForVertex(edge, vertex);
        cachedDirections[vertexOffset * nEdges + edgeOffset] = wi;
        if (edge.isDeltaLight) {
            density += edge.lightPMF;
            continue;
        }
        density += edge.lightPMF * edge.light.PDF_Li(OffsetLightContext(vertex), wi, true);
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
        Vector3f wi = ContinuationDirectionForVertex(edge, vertex, vertices);
        if (IsSpecular(edge.flags)) {
            if (vertex.vertexId == edge.vertexA)
                density += edge.pdf;
        } else {
            density += vertex.bsdf->PDF(vertex.wo, wi);
        }
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
    contEdgeTargetVertexIndices.clear();
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

        Cluster &cluster = clusters[clusterId];
        cluster.clusterId = clusterId;

        for (uint32_t i = begin; i < end; ++i) {
            uint32_t vertexIndex = sortedVertexIndices[i];
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
    }

    for (uint32_t edgeIndex = 0; edgeIndex < contEdges.size(); ++edgeIndex) {
        uint32_t vertexIndex = VertexIndexFromId(contEdges[edgeIndex].vertexA);
        if (vertexIndex == uint32_t(-1))
            continue;
        Cluster &cluster = clusters[clusterByVertexIndex[vertexIndex]];
        cluster.contEdgeIndices.push_back(edgeIndex);
    }

    contEdgeTargetVertexIndices.resize(contEdges.size(), uint32_t(-1));
    for (uint32_t edgeIndex = 0; edgeIndex < contEdges.size(); ++edgeIndex)
        contEdgeTargetVertexIndices[edgeIndex] = VertexIndexFromId(contEdges[edgeIndex].vertexB);
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
    ParallelFor(0, clusters.size(), [&](int64_t clusterIndex) {
        const Cluster &cluster = clusters[clusterIndex];
        uint32_t nVertices = cluster.vertexIndices.size();
        uint32_t nEdges = cluster.lightEdgeIndices.size();
        std::vector<Vector3f> lightDirections(nVertices * nEdges);
        std::vector<Float> marginalDensities(nEdges, 0);
        for (uint32_t edgeOffset = 0; edgeOffset < nEdges; ++edgeOffset) {
            const LightEdge &edge = lightEdges[cluster.lightEdgeIndices[edgeOffset]];
            if (edge.pdf > 0)
                marginalDensities[edgeOffset] =
                    DirectMarginalDensity(edge, cluster, vertexSpan, lightDirections,
                                          edgeOffset, nEdges);
        }

        for (uint32_t vertexOffset = 0; vertexOffset < nVertices; ++vertexOffset) {
            uint32_t vertexIndex = cluster.vertexIndices[vertexOffset];
            SurfaceVertex &vertex = vertices[vertexIndex];
            SampledSpectrum Ld(0.f);

            for (uint32_t edgeOffset = 0; edgeOffset < nEdges; ++edgeOffset) {
                uint32_t edgeIndex = cluster.lightEdgeIndices[edgeOffset];
                const LightEdge &edge = lightEdges[edgeIndex];
                if (edge.pdf <= 0)
                    continue;

                Float marginalDensity = marginalDensities[edgeOffset];
                if (marginalDensity <= 0)
                    continue;

                Vector3f wi = lightDirections[vertexOffset * nEdges + edgeOffset];
                SampledSpectrum fcos = fcosEvaluator(vertex, wi);
                if (!fcos)
                    continue;

                Ld += fcos * edge.L_B * edge.misWeight / marginalDensity;
            }

            vertex.L_direct = Ld;
        }
    });
}

void PathGraphSnapshot::BuildIndirectTransferWeights(
    const IndirectFcosEvaluator &fcosEvaluator) {
    indirectTransferWeights.clear();
    indirectTransferWeights.resize(clusters.size());
    if (!fcosEvaluator) {
        indirectTransferWeightsValid = true;
        return;
    }

    pstd::span<const SurfaceVertex> vertexSpan(vertices);
    ParallelFor(0, clusters.size(), [&](int64_t clusterIndex) {
        const Cluster &cluster = clusters[clusterIndex];
        size_t nVertices = cluster.vertexIndices.size();
        size_t nEdges = cluster.contEdgeIndices.size();
        std::vector<SampledSpectrum> &weights = indirectTransferWeights[clusterIndex];
        weights.assign(nVertices * nEdges, SampledSpectrum(0.f));
        if (nVertices == 0 || nEdges == 0)
            return;

        std::vector<Float> marginalDensities(nEdges, 0);
        for (uint32_t edgeOffset = 0; edgeOffset < nEdges; ++edgeOffset) {
            const ContEdge &edge = contEdges[cluster.contEdgeIndices[edgeOffset]];
            if (edge.pdf > 0)
                marginalDensities[edgeOffset] =
                    IndirectMarginalDensity(edge, cluster, vertexSpan);
        }

        for (uint32_t vertexOffset = 0; vertexOffset < nVertices; ++vertexOffset) {
            const SurfaceVertex &vertex = vertices[cluster.vertexIndices[vertexOffset]];
            for (uint32_t edgeOffset = 0; edgeOffset < nEdges; ++edgeOffset) {
                Float marginalDensity = marginalDensities[edgeOffset];
                if (marginalDensity <= 0)
                    continue;

                ContEdge adjustedEdge =
                    ContEdgeForVertex(contEdges[cluster.contEdgeIndices[edgeOffset]],
                                      vertex, vertexSpan);
                SampledSpectrum fcos = fcosEvaluator(vertex, adjustedEdge);
                if (!fcos)
                    continue;

                weights[vertexOffset * nEdges + edgeOffset] = fcos / marginalDensity;
            }
        }
    });

    indirectTransferWeightsValid = true;
}

void PathGraphSnapshot::AggregateIndirectLighting(
    const IndirectFcosEvaluator &fcosEvaluator) {
    previousIndirect.resize(vertices.size());
    ParallelFor(0, vertices.size(), [&](int64_t vertexIndex) {
        previousIndirect[vertexIndex] = vertices[vertexIndex].L_indirect;
        vertices[vertexIndex].L_indirect = SampledSpectrum(0.f);
    });


    if (!fcosEvaluator)
        return;

    if (!indirectTransferWeightsValid)
        BuildIndirectTransferWeights(fcosEvaluator);

    ParallelFor(0, contEdges.size(), [&](int64_t edgeIndex) {
        ContEdge &edge = contEdges[edgeIndex];
        uint32_t vertexIndex = contEdgeTargetVertexIndices[edgeIndex];
        if (vertexIndex == uint32_t(-1)) {
            edge.L_B = SampledSpectrum(0.f);
            return;
        }

        const SurfaceVertex &vertexB = vertices[vertexIndex];
        edge.L_B = vertexB.L_direct + previousIndirect[vertexIndex];
    });

    ParallelFor(0, clusters.size(), [&](int64_t clusterIndex) {
        const Cluster &cluster = clusters[clusterIndex];
        const std::vector<SampledSpectrum> &weights =
            indirectTransferWeights[clusterIndex];
        size_t nEdges = cluster.contEdgeIndices.size();
        Float inputEnergy = 0;
        for (uint32_t edgeIndex : cluster.contEdgeIndices)
            inputEnergy += contEdges[edgeIndex].L_B.Average();

        for (uint32_t vertexOffset = 0; vertexOffset < cluster.vertexIndices.size();
             ++vertexOffset) {
            uint32_t vertexIndex = cluster.vertexIndices[vertexOffset];
            SurfaceVertex &vertex = vertices[vertexIndex];
            SampledSpectrum Li(0.f);

            for (uint32_t edgeOffset = 0; edgeOffset < nEdges; ++edgeOffset) {
                uint32_t edgeIndex = cluster.contEdgeIndices[edgeOffset];
                const ContEdge &edge = contEdges[edgeIndex];
                if (edge.pdf <= 0 || !edge.L_B)
                    continue;

                const SampledSpectrum &weight =
                    weights[vertexOffset * nEdges + edgeOffset];
                if (!weight)
                    continue;

                Li += weight * edge.L_B;
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
    });
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
        ContEdge adjustedEdge = ContEdgeForVertex(edge, vertex, vertices);
        Float marginalDensity = 0;
        if (IsSpecular(edge.flags))
            marginalDensity = edge.pdf;
        else
            marginalDensity = vertex.bsdf ? vertex.bsdf->PDF(vertex.wo, adjustedEdge.wi) : 0;
        if (marginalDensity <= 0)
            continue;

        SampledSpectrum fcos = indirectFcosEvaluator(vertex, adjustedEdge);
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
