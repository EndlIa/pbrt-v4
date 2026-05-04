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
    lastSurfaceVertexId = 0;
}

uint64_t PathGraphSink::AddSurfaceVertex(SurfaceVertex vertex) {
    if (!data || !nextVertexId)
        return 0;

    vertex.vertexId = nextVertexId->fetch_add(1, std::memory_order_relaxed);
    data->lastSurfaceVertexId = vertex.vertexId;
    data->vertices.push_back(vertex);
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

PathGraphSnapshot::PathGraphSnapshot(std::vector<SurfaceVertex> vertices,
                                     std::vector<LightEdge> lightEdges,
                                     std::vector<ContEdge> contEdges,
                                     uint32_t targetClusterSize)
    : vertices(std::move(vertices)),
      lightEdges(std::move(lightEdges)),
      contEdges(std::move(contEdges)) {
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

    impl->threadData.ForAll([&](const PathGraphThreadData &data) {
        vertices.insert(vertices.end(), data.vertices.begin(), data.vertices.end());
        lightEdges.insert(lightEdges.end(), data.lightEdges.begin(), data.lightEdges.end());
        contEdges.insert(contEdges.end(), data.contEdges.begin(), data.contEdges.end());
    });

    std::sort(vertices.begin(), vertices.end(),
              [](const SurfaceVertex &a, const SurfaceVertex &b) {
                  return a.vertexId < b.vertexId;
              });

    return std::make_unique<PathGraphSnapshot>(std::move(vertices), std::move(lightEdges),
                                               std::move(contEdges), targetClusterSize);
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
