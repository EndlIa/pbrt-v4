#include <pbrt/pathGraph.h>

#include <pbrt/util/parallel.h>

#include <algorithm>
#include <atomic>
#include <random>
#include <unordered_map>
#include <utility>

namespace pbrt {

namespace {

std::atomic<PathGraphBuilder *> activePathGraphBuilder{nullptr};

}  // namespace

void PathGraphThreadData::Clear() {
    vertices.clear();
    lightEdges.clear();
    contEdges.clear();
    pixelVertexMap.clear();
    bsdfs.clear();
    lastSurfaceVertexId = 0;
    currentFirstVertexId = 0;
    hasCurrentPixelSample = false;
}

uint64_t PathGraphSink::AddSurfaceVertex(SurfaceVertex vertex) {
    if (!data || !nextVertexId)
        return 0;

    vertex.vertexId = nextVertexId->fetch_add(1, std::memory_order_relaxed);
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

PathGraphSnapshot::PathGraphSnapshot(std::vector<SurfaceVertex> vertices,
                                     std::vector<LightEdge> lightEdges,
                                     std::vector<ContEdge> contEdges,
                                     std::vector<PixelVertexMapEntry> pixelVertexMap,
                                     std::deque<BSDF> bsdfs,
                                     uint32_t targetClusterSize)
    : vertices(std::move(vertices)),
      lightEdges(std::move(lightEdges)),
      contEdges(std::move(contEdges)),
      pixelVertexMap(std::move(pixelVertexMap)),
      bsdfs(std::move(bsdfs)) {
    BuildClusters(targetClusterSize);
}

void PathGraphSnapshot::BuildClusters(uint32_t targetClusterSize) {
    clusters.clear();
    if (vertices.empty())
        return;

    targetClusterSize = std::max<uint32_t>(1, targetClusterSize);
    uint32_t nClusters =
        std::max<uint32_t>(1, (vertices.size() + targetClusterSize - 1) / targetClusterSize);
    nClusters = std::min<uint32_t>(nClusters, vertices.size());

    std::vector<uint32_t> shuffledVertexIndices(vertices.size());
    for (uint32_t i = 0; i < shuffledVertexIndices.size(); ++i)
        shuffledVertexIndices[i] = i;

    // The paper uses uniformly random centers and assigns each vertex to the
    // nearest center (one k-means iteration). Use a fixed seed for repeatability.
    std::mt19937 rng(0x70677261u);
    std::shuffle(shuffledVertexIndices.begin(), shuffledVertexIndices.end(), rng);

    clusters.resize(nClusters);
    for (uint32_t clusterId = 0; clusterId < nClusters; ++clusterId) {
        const SurfaceVertex &centerVertex = vertices[shuffledVertexIndices[clusterId]];
        Cluster &cluster = clusters[clusterId];
        cluster.clusterId = clusterId;
        cluster.centerVertexId = centerVertex.vertexId;
        cluster.center = centerVertex.pos;
    }

    std::unordered_map<uint64_t, uint32_t> vertexIdToCluster;
    vertexIdToCluster.reserve(vertices.size());

    for (uint32_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex) {
        const SurfaceVertex &vertex = vertices[vertexIndex];
        uint32_t nearestCluster = 0;
        Float nearestDist2 = DistanceSquared(vertex.pos, clusters[0].center);
        for (uint32_t clusterId = 1; clusterId < clusters.size(); ++clusterId) {
            Float dist2 = DistanceSquared(vertex.pos, clusters[clusterId].center);
            if (dist2 < nearestDist2) {
                nearestDist2 = dist2;
                nearestCluster = clusterId;
            }
        }

        Cluster &cluster = clusters[nearestCluster];
        cluster.vertexIds.push_back(vertex.vertexId);
        cluster.vertexIndices.push_back(vertexIndex);
        vertexIdToCluster[vertex.vertexId] = nearestCluster;
    }

    for (uint32_t edgeIndex = 0; edgeIndex < lightEdges.size(); ++edgeIndex) {
        auto iter = vertexIdToCluster.find(lightEdges[edgeIndex].vertexA);
        if (iter == vertexIdToCluster.end())
            continue;
        Cluster &cluster = clusters[iter->second];
        cluster.lightEdgeIndices.push_back(edgeIndex);
        cluster.lightEdgePdfSum += lightEdges[edgeIndex].pdf;
    }

    for (uint32_t edgeIndex = 0; edgeIndex < contEdges.size(); ++edgeIndex) {
        auto iter = vertexIdToCluster.find(contEdges[edgeIndex].vertexA);
        if (iter == vertexIdToCluster.end())
            continue;
        Cluster &cluster = clusters[iter->second];
        cluster.contEdgeIndices.push_back(edgeIndex);
        cluster.contEdgePdfSum += contEdges[edgeIndex].pdf;
    }
}

void PathGraphSnapshot::AggregateDirectLighting(
    const DirectFcosEvaluator &fcosEvaluator) {
    for (SurfaceVertex &vertex : vertices)
        vertex.L_direct = SampledSpectrum(0.f);

    if (!fcosEvaluator)
        return;

    for (const Cluster &cluster : clusters) {
        if (cluster.lightEdgePdfSum <= 0)
            continue;

        for (uint32_t vertexIndex : cluster.vertexIndices) {
            SurfaceVertex &vertex = vertices[vertexIndex];
            SampledSpectrum Ld(0.f);

            for (uint32_t edgeIndex : cluster.lightEdgeIndices) {
                const LightEdge &edge = lightEdges[edgeIndex];
                if (edge.pdf <= 0)
                    continue;

                SampledSpectrum fcos = fcosEvaluator(vertex, edge);
                if (!fcos)
                    continue;

                Ld += fcos * edge.L_B * edge.misWeight / cluster.lightEdgePdfSum;
            }

            vertex.L_direct = Ld;
        }
    }
}

void PathGraphSnapshot::AggregateIndirectLighting(
    const IndirectFcosEvaluator &fcosEvaluator) {
    for (SurfaceVertex &vertex : vertices)
        vertex.L_indirect = SampledSpectrum(0.f);

    if (!fcosEvaluator)
        return;

    std::unordered_map<uint64_t, uint32_t> vertexIdToIndex;
    vertexIdToIndex.reserve(vertices.size());
    for (uint32_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex)
        vertexIdToIndex[vertices[vertexIndex].vertexId] = vertexIndex;

    for (ContEdge &edge : contEdges) {
        auto iter = vertexIdToIndex.find(edge.vertexB);
        if (iter == vertexIdToIndex.end()) {
            edge.L_B = SampledSpectrum(0.f);
            continue;
        }

        const SurfaceVertex &vertexB = vertices[iter->second];
        edge.L_B = vertexB.L_direct + vertexB.L_indirect;
    }

    for (const Cluster &cluster : clusters) {
        if (cluster.contEdgePdfSum <= 0)
            continue;

        for (uint32_t vertexIndex : cluster.vertexIndices) {
            SurfaceVertex &vertex = vertices[vertexIndex];
            SampledSpectrum Li(0.f);

            for (uint32_t edgeIndex : cluster.contEdgeIndices) {
                const ContEdge &edge = contEdges[edgeIndex];
                if (edge.pdf <= 0 || !edge.L_B)
                    continue;

                SampledSpectrum fcos = fcosEvaluator(vertex, edge);
                if (!fcos)
                    continue;

                Li += fcos * edge.L_B / cluster.contEdgePdfSum;
            }

            vertex.L_indirect = Li;
        }
    }
}

void PathGraphSnapshot::FinalGather(
    const IndirectFcosEvaluator &indirectFcosEvaluator) {
    for (SurfaceVertex &vertex : vertices)
        vertex.L_out = vertex.L_direct;

    if (!indirectFcosEvaluator)
        return;

    std::unordered_map<uint64_t, uint32_t> vertexIdToIndex;
    vertexIdToIndex.reserve(vertices.size());
    for (uint32_t vertexIndex = 0; vertexIndex < vertices.size(); ++vertexIndex)
        vertexIdToIndex[vertices[vertexIndex].vertexId] = vertexIndex;

    std::vector<Float> contPdfSums(vertices.size(), 0);

    for (const ContEdge &edge : contEdges) {
        auto iter = vertexIdToIndex.find(edge.vertexA);
        if (iter != vertexIdToIndex.end())
            contPdfSums[iter->second] += edge.pdf;
    }

    for (const ContEdge &edge : contEdges) {
        auto iter = vertexIdToIndex.find(edge.vertexA);
        if (iter == vertexIdToIndex.end())
            continue;

        uint32_t vertexIndex = iter->second;
        if (contPdfSums[vertexIndex] <= 0 || !edge.L_B)
            continue;

        SurfaceVertex &vertex = vertices[vertexIndex];
        SampledSpectrum fcos = indirectFcosEvaluator(vertex, edge);
        if (!fcos)
            continue;

        vertex.L_out += fcos * edge.L_B / contPdfSums[vertexIndex];
    }
}

class PathGraphBuilder::Impl {
  public:
    ThreadLocal<PathGraphThreadData> threadData;
    ThreadLocal<PathGraphSink> sinks;
    std::atomic<uint64_t> nextVertexId{1};

    Impl()
        : threadData([]() { return PathGraphThreadData(); }),
          sinks([this]() { return PathGraphSink(&threadData.Get(), &nextVertexId); }) {}
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
    std::deque<BSDF> bsdfs;
    std::unordered_map<const BSDF *, const BSDF *> bsdfRemap;

    impl->threadData.ForAll([&](const PathGraphThreadData &data) {
        for (const BSDF &bsdf : data.bsdfs) {
            const BSDF *oldPtr = &bsdf;
            bsdfs.push_back(bsdf);
            bsdfRemap[oldPtr] = &bsdfs.back();
        }
        vertices.insert(vertices.end(), data.vertices.begin(), data.vertices.end());
        lightEdges.insert(lightEdges.end(), data.lightEdges.begin(), data.lightEdges.end());
        contEdges.insert(contEdges.end(), data.contEdges.begin(), data.contEdges.end());
        pixelVertexMap.insert(pixelVertexMap.end(), data.pixelVertexMap.begin(),
                              data.pixelVertexMap.end());
    });

    for (SurfaceVertex &vertex : vertices) {
        if (!vertex.bsdf)
            continue;
        auto iter = bsdfRemap.find(vertex.bsdf);
        vertex.bsdf = iter == bsdfRemap.end() ? nullptr : iter->second;
    }

    std::sort(vertices.begin(), vertices.end(),
              [](const SurfaceVertex &a, const SurfaceVertex &b) {
                  return a.vertexId < b.vertexId;
              });

    return std::make_unique<PathGraphSnapshot>(std::move(vertices), std::move(lightEdges),
                                               std::move(contEdges), std::move(pixelVertexMap),
                                               std::move(bsdfs), targetClusterSize);
}

void PathGraphBuilder::Reset() {
    impl->threadData.ForAll([](PathGraphThreadData &data) { data.Clear(); });
    impl->nextVertexId.store(1, std::memory_order_relaxed);
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
