#include <pbrt/pathGraph.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

namespace pbrt {

namespace {

constexpr uint64_t kCameraRootVertexId = 0;
constexpr int kDefaultContIterations = 4;

struct PathGraphSolveState {
    std::vector<SampledSpectrum> directVertexRadiance;
    std::vector<SampledSpectrum> iteratedVertexRadiance;
    std::vector<SampledSpectrum> finalGatherVertexRadiance;
    std::vector<SampledSpectrum> contEdgeRadiance;
};

uint64_t GetMaxVertexId(const PathGraphSnapshot &snapshot) {
    uint64_t maxVertexId = 0;
    for (const SurfaceVertex &vertex : snapshot.Vertices())
        maxVertexId = std::max(maxVertexId, vertex.vertexId);
    for (const LightEdge &edge : snapshot.LightEdges())
        maxVertexId = std::max(maxVertexId, edge.vertexA);
    for (const ContEdge &edge : snapshot.ContEdges()) {
        maxVertexId = std::max(maxVertexId, edge.vertexA);
        maxVertexId = std::max(maxVertexId, edge.vertexB);
    }
    return maxVertexId;
}

// Build C(j) from the snapshot interface. The paper defines C(j) as the
// clustering neighborhood plus j itself. If the current snapshot exposes no
// neighbors, the cluster naturally degenerates to the singleton {j}.
std::vector<uint64_t> GetClusterVertices(const PathGraphSnapshot &snapshot,
                                         uint64_t vertexId, bool useNeighbors) {
    if (vertexId == kCameraRootVertexId)
        return {};

    std::vector<uint64_t> cluster;
    cluster.push_back(vertexId);
    if (!useNeighbors)
        return cluster;

    for (uint64_t neighborId : snapshot.NeighborsOfVertex(vertexId)) {
        if (neighborId == kCameraRootVertexId)
            continue;
        if (std::find(cluster.begin(), cluster.end(), neighborId) == cluster.end())
            cluster.push_back(neighborId);
    }
    return cluster;
}

// The paper's aggregate() first gathers the marginal density rho_e over the
// whole cluster C(j), then scatters the contribution back to every k in C(j).
//
// With the current PathGraphSnapshot interface we only have the sampled edge
// data stored once per edge, not a full hitData re-evaluation API for arbitrary
// neighbors. Therefore this prototype interprets the stored pdf / ef as the
// local edge estimate associated with j and reuses it uniformly across C(j).
Float ComputeClusterPdfDenominator(const PathGraphSnapshot &snapshot, uint64_t vertexId,
                                   Float edgePdf, bool useNeighbors) {
    std::vector<uint64_t> cluster = GetClusterVertices(snapshot, vertexId, useNeighbors);
    if (cluster.empty())
        return 0;
    return edgePdf * cluster.size();
}

void ScatterToCluster(std::vector<SampledSpectrum> *vertexRadiance,
                      const PathGraphSnapshot &snapshot, uint64_t vertexId,
                      SampledSpectrum contribution, bool useNeighbors) {
    std::vector<uint64_t> cluster = GetClusterVertices(snapshot, vertexId, useNeighbors);
    for (uint64_t clusterVertexId : cluster)
        (*vertexRadiance)[clusterVertexId] += contribution;
}

// aggregate(E_L, E_N, L, hitData)
//
// Light edges contribute the direct term L_d once before the continuation-edge
// fixed-point iterations start.
std::vector<SampledSpectrum> AggregateLightEdges(const PathGraphSnapshot &snapshot,
                                                 uint64_t maxVertexId,
                                                 bool useNeighbors) {
    std::vector<SampledSpectrum> vertexRadiance(maxVertexId + 1, SampledSpectrum(0.f));

    for (uint64_t vertexId = 1; vertexId <= maxVertexId; ++vertexId) {
        for (uint32_t edgeIndex : snapshot.LightEdgesOfVertex(vertexId)) {
            const LightEdge &edge = snapshot.LightEdges()[edgeIndex];
            if (edge.pdf <= 0)
                continue;

            Float rho = ComputeClusterPdfDenominator(snapshot, vertexId, edge.pdf,
                                                     useNeighbors);
            if (rho <= 0)
                continue;

            SampledSpectrum contribution =
                edge.ef * edge.L_B * edge.misWeight / rho;
            ScatterToCluster(&vertexRadiance, snapshot, vertexId, contribution,
                             useNeighbors);
        }
    }

    return vertexRadiance;
}

// propagate(E_C, L)
//
// Each continuation edge inherits the outgoing radiance of its light-side
// endpoint. By the COOP.md convention, vertexA is closer to the camera and
// vertexB is closer to the light, so radiance flows from vertexB to vertexA.
std::vector<SampledSpectrum> PropagateContEdges(const PathGraphSnapshot &snapshot,
                                                pstd::span<const SampledSpectrum> vertexRadiance,
                                                uint64_t maxVertexId) {
    std::vector<SampledSpectrum> edgeRadiance(snapshot.ContEdges().size(),
                                              SampledSpectrum(0.f));

    for (size_t edgeIndex = 0; edgeIndex < snapshot.ContEdges().size(); ++edgeIndex) {
        const ContEdge &edge = snapshot.ContEdges()[edgeIndex];
        if (edge.vertexB == kCameraRootVertexId || edge.vertexB > maxVertexId)
            continue;
        edgeRadiance[edgeIndex] = vertexRadiance[edge.vertexB];
    }

    return edgeRadiance;
}

// aggregate(E_C, E_N, L, hitData)
//
// ContEdgesIntoVertex(j) enumerates continuation edges transferring energy into
// j. For this data layout, that means the edge's camera-side endpoint is j.
std::vector<SampledSpectrum> AggregateContEdges(
    const PathGraphSnapshot &snapshot, pstd::span<const SampledSpectrum> contEdgeRadiance,
    uint64_t maxVertexId, bool useNeighbors) {
    std::vector<SampledSpectrum> vertexRadiance(maxVertexId + 1, SampledSpectrum(0.f));

    for (uint64_t vertexId = 1; vertexId <= maxVertexId; ++vertexId) {
        for (uint32_t edgeIndex : snapshot.ContEdgesIntoVertex(vertexId)) {
            const ContEdge &edge = snapshot.ContEdges()[edgeIndex];
            if (edge.pdf <= 0)
                continue;

            Float rho = ComputeClusterPdfDenominator(snapshot, vertexId, edge.pdf,
                                                     useNeighbors);
            if (rho <= 0)
                continue;

            SampledSpectrum contribution = edge.ef * contEdgeRadiance[edgeIndex] / rho;
            ScatterToCluster(&vertexRadiance, snapshot, vertexId, contribution,
                             useNeighbors);
        }
    }

    return vertexRadiance;
}

std::vector<SampledSpectrum> AddVertexRadiance(pstd::span<const SampledSpectrum> a,
                                               pstd::span<const SampledSpectrum> b) {
    size_t n = std::max(a.size(), b.size());
    std::vector<SampledSpectrum> result(n, SampledSpectrum(0.f));
    for (size_t i = 0; i < a.size(); ++i)
        result[i] += a[i];
    for (size_t i = 0; i < b.size(); ++i)
        result[i] += b[i];
    return result;
}

// finalGather(E_C, L_d, L, hitData)
//
// The paper's final gather performs one extra continuation-edge propagation and
// local aggregation without E_N reuse. That is why useNeighbors=false here.
std::vector<SampledSpectrum> FinalGather(const PathGraphSnapshot &snapshot,
                                         pstd::span<const SampledSpectrum> directRadiance,
                                         pstd::span<const SampledSpectrum> iteratedRadiance,
                                         uint64_t maxVertexId) {
    std::vector<SampledSpectrum> propagatedRadiance =
        PropagateContEdges(snapshot, iteratedRadiance, maxVertexId);
    std::vector<SampledSpectrum> localIndirectRadiance =
        AggregateContEdges(snapshot, propagatedRadiance, maxVertexId,
                           /*useNeighbors=*/false);
    return AddVertexRadiance(directRadiance, localIndirectRadiance);
}

PathGraphSolveState SolvePathGraph(const PathGraphSnapshot &snapshot,
                                   int nContIterations) {
    uint64_t maxVertexId = GetMaxVertexId(snapshot);
    PathGraphSolveState state;

    // 1. L_d = aggregate(E_L, E_N, ...)
    state.directVertexRadiance =
        AggregateLightEdges(snapshot, maxVertexId, /*useNeighbors=*/true);

    // 2. Initialize the iterated outgoing radiance with the direct term, then
    // repeatedly apply propagate(E_C, L) followed by aggregate(E_C, E_N, ...).
    state.iteratedVertexRadiance = state.directVertexRadiance;
    for (int iter = 0; iter < std::max(0, nContIterations); ++iter) {
        state.contEdgeRadiance =
            PropagateContEdges(snapshot, state.iteratedVertexRadiance, maxVertexId);
        std::vector<SampledSpectrum> indirectRadiance =
            AggregateContEdges(snapshot, state.contEdgeRadiance, maxVertexId,
                               /*useNeighbors=*/true);
        state.iteratedVertexRadiance =
            AddVertexRadiance(state.directVertexRadiance, indirectRadiance);
    }

    // 3. One extra local gather gives the per-vertex outgoing radiance used by
    // the paper's finalGather() step.
    state.finalGatherVertexRadiance =
        FinalGather(snapshot, state.directVertexRadiance, state.iteratedVertexRadiance,
                    maxVertexId);
    return state;
}

// Immutable snapshot consumed by the later path-graph stages. The builder
// captures per-thread vectors first, then packs them into these contiguous
// arrays plus CSR-style adjacency tables for fast read-only traversal.
class BasicPathGraphSnapshot final : public PathGraphSnapshot {
  public:
    BasicPathGraphSnapshot(std::vector<SurfaceVertex> vertices,
                           std::vector<LightEdge> lightEdges,
                           std::vector<ContEdge> contEdges,
                           std::vector<uint32_t> contEdgeOffsets,
                           std::vector<uint32_t> contEdgeIndices,
                           std::vector<uint32_t> lightEdgeOffsets,
                           std::vector<uint32_t> lightEdgeIndices,
                           std::vector<uint32_t> neighborOffsets,
                           std::vector<uint64_t> neighborVertexIds)
        : vertices(std::move(vertices)),
          lightEdges(std::move(lightEdges)),
          contEdges(std::move(contEdges)),
          contEdgeOffsets(std::move(contEdgeOffsets)),
          contEdgeIndices(std::move(contEdgeIndices)),
          lightEdgeOffsets(std::move(lightEdgeOffsets)),
          lightEdgeIndices(std::move(lightEdgeIndices)),
          neighborOffsets(std::move(neighborOffsets)),
          neighborVertexIds(std::move(neighborVertexIds)) {}

    pstd::span<const SurfaceVertex> Vertices() const override { return vertices; }
    pstd::span<const LightEdge> LightEdges() const override { return lightEdges; }
    pstd::span<const ContEdge> ContEdges() const override { return contEdges; }
    pstd::span<const uint32_t> ContEdgesIntoVertex(uint64_t vertexId) const override {
        return SliceAdjacency(contEdgeOffsets, contEdgeIndices, vertexId);
    }
    pstd::span<const uint32_t> LightEdgesOfVertex(uint64_t vertexId) const override {
        return SliceAdjacency(lightEdgeOffsets, lightEdgeIndices, vertexId);
    }
    pstd::span<const uint64_t> NeighborsOfVertex(uint64_t vertexId) const override {
        return SliceAdjacency(neighborOffsets, neighborVertexIds, vertexId);
    }

  private:
    template <typename T>
    pstd::span<const T> SliceAdjacency(const std::vector<uint32_t> &offsets,
                                       const std::vector<T> &values,
                                       uint64_t vertexId) const {
        // The offset arrays follow the usual CSR convention:
        // [offsets[v], offsets[v + 1]) stores the neighbor/index list of vertex v.
        if (vertexId + 1 >= offsets.size())
            return {};
        uint32_t begin = offsets[vertexId];
        uint32_t end = offsets[vertexId + 1];
        if (begin == end)
            return {};
        return pstd::span<const T>(values.data() + begin, end - begin);
    }

    std::vector<SurfaceVertex> vertices;
    std::vector<LightEdge> lightEdges;
    std::vector<ContEdge> contEdges;
    std::vector<uint32_t> contEdgeOffsets;
    std::vector<uint32_t> contEdgeIndices;
    std::vector<uint32_t> lightEdgeOffsets;
    std::vector<uint32_t> lightEdgeIndices;
    std::vector<uint32_t> neighborOffsets;
    std::vector<uint64_t> neighborVertexIds;
};

PathGraphBuilder *lastBuilderForCluster = nullptr;

}  // namespace

class BasicPathGraphBuilder::Impl {
  public:
    class ThreadLocalSink final : public PathGraphSink {
      public:
        explicit ThreadLocalSink(Impl *owner) : owner(owner) {}

        void BeginPath(uint64_t pathId) override {
            currentPathId = pathId != 0 ? pathId : owner->nextPathId.fetch_add(1);
            inPath = true;
            lastVertexId = kCameraRootVertexId;
        }

        void AddSurfaceVertex(const SurfaceVertex &v) override {
            if (!inPath)
                BeginPath(0);

            SurfaceVertex stored = v;
            stored.pathId = currentPathId;
            stored.vertexId = stored.vertexId != 0 ? stored.vertexId
                                                   : owner->nextVertexId.fetch_add(1);
            if (pendingContEdge) {
                // The continuation edge is emitted at vertexA first and closed
                // once the next surface vertex on the same path is known.
                pendingContEdge->vertexB = stored.vertexId;
                contEdges.push_back(*pendingContEdge);
                pendingContEdge.reset();
            }
            vertices.push_back(stored);
            lastVertexId = stored.vertexId;
        }

        void AddLightEdge(const LightEdge &e) override {
            if (!inPath)
                BeginPath(0);

            LightEdge stored = e;
            if (stored.vertexA == 0)
                stored.vertexA = lastVertexId;
            lightEdges.push_back(stored);
        }

        void AddContEdge(const ContEdge &e) override {
            if (!inPath)
                BeginPath(0);

            ContEdge stored = e;
            if (stored.vertexA == 0)
                stored.vertexA = lastVertexId;
            if (stored.vertexB == 0)
                pendingContEdge = stored;
            else
                contEdges.push_back(stored);
        }

        void EndPath(uint64_t pathId) override {
            if (!inPath)
                return;
            if (pathId != 0 && pathId != currentPathId)
                return;
            inPath = false;
            currentPathId = 0;
            lastVertexId = kCameraRootVertexId;
            pendingContEdge.reset();
        }

        std::vector<SurfaceVertex> vertices;
        std::vector<LightEdge> lightEdges;
        std::vector<ContEdge> contEdges;

      private:
        Impl *owner;
        pstd::optional<ContEdge> pendingContEdge;
        uint64_t currentPathId = 0;
        uint64_t lastVertexId = kCameraRootVertexId;
        bool inPath = false;
    };

    ThreadLocalSink *GetThreadLocalSink() {
        thread_local std::vector<std::unique_ptr<ThreadLocalSink>> threadSinks;
        thread_local std::vector<Impl *> threadOwners;
        for (size_t i = 0; i < threadOwners.size(); ++i)
            if (threadOwners[i] == this)
                return threadSinks[i].get();

        auto sink = std::make_unique<ThreadLocalSink>(this);
        ThreadLocalSink *sinkPtr = sink.get();
        {
            std::lock_guard<std::mutex> lock(sinksMutex);
            sinks.push_back(sinkPtr);
        }
        threadOwners.push_back(this);
        threadSinks.push_back(std::move(sink));
        return sinkPtr;
    }

    std::shared_ptr<const PathGraphSnapshot> BuildSnapshot() {
        std::vector<SurfaceVertex> allVertices;
        std::vector<LightEdge> allLightEdges;
        std::vector<ContEdge> allContEdges;
        {
            std::lock_guard<std::mutex> lock(sinksMutex);
            for (ThreadLocalSink *sink : sinks) {
                allVertices.insert(allVertices.end(), sink->vertices.begin(),
                                   sink->vertices.end());
                allLightEdges.insert(allLightEdges.end(), sink->lightEdges.begin(),
                                     sink->lightEdges.end());
                allContEdges.insert(allContEdges.end(), sink->contEdges.begin(),
                                    sink->contEdges.end());
            }
        }

        uint64_t maxVertexId = 0;
        for (const SurfaceVertex &vertex : allVertices)
            maxVertexId = std::max(maxVertexId, vertex.vertexId);
        for (const LightEdge &edge : allLightEdges)
            maxVertexId = std::max(maxVertexId, edge.vertexA);
        for (const ContEdge &edge : allContEdges) {
            maxVertexId = std::max(maxVertexId, edge.vertexA);
            maxVertexId = std::max(maxVertexId, edge.vertexB);
        }

        // ContEdgesIntoVertex(vertexId) is defined in COOP.md as the set of
        // continuation edges transferring energy into that vertex. Since energy
        // propagates from vertexB to vertexA, the adjacency must be keyed by
        // vertexA rather than vertexB.
        std::vector<uint32_t> contEdgeOffsets(maxVertexId + 2, 0);
        std::vector<uint32_t> lightEdgeOffsets(maxVertexId + 2, 0);
        std::vector<uint32_t> neighborOffsets(maxVertexId + 2, 0);

        for (const ContEdge &edge : allContEdges)
            ++contEdgeOffsets[edge.vertexA + 1];
        for (const LightEdge &edge : allLightEdges)
            ++lightEdgeOffsets[edge.vertexA + 1];

        for (size_t i = 1; i < contEdgeOffsets.size(); ++i)
            contEdgeOffsets[i] += contEdgeOffsets[i - 1];
        for (size_t i = 1; i < lightEdgeOffsets.size(); ++i)
            lightEdgeOffsets[i] += lightEdgeOffsets[i - 1];
        for (size_t i = 1; i < neighborOffsets.size(); ++i)
            neighborOffsets[i] += neighborOffsets[i - 1];

        std::vector<uint32_t> contEdgeIndices(allContEdges.size());
        std::vector<uint32_t> lightEdgeIndices(allLightEdges.size());
        std::vector<uint64_t> neighborVertexIds(neighborOffsets.back());

        std::vector<uint32_t> contEdgeCursor = contEdgeOffsets;
        std::vector<uint32_t> lightEdgeCursor = lightEdgeOffsets;

        for (uint32_t edgeIndex = 0; edgeIndex < allContEdges.size(); ++edgeIndex) {
            const ContEdge &edge = allContEdges[edgeIndex];
            contEdgeIndices[contEdgeCursor[edge.vertexA]++] = edgeIndex;
        }
        for (uint32_t edgeIndex = 0; edgeIndex < allLightEdges.size(); ++edgeIndex) {
            const LightEdge &edge = allLightEdges[edgeIndex];
            lightEdgeIndices[lightEdgeCursor[edge.vertexA]++] = edgeIndex;
        }

        return std::make_shared<BasicPathGraphSnapshot>(
            std::move(allVertices), std::move(allLightEdges), std::move(allContEdges),
            std::move(contEdgeOffsets), std::move(contEdgeIndices),
            std::move(lightEdgeOffsets), std::move(lightEdgeIndices),
            std::move(neighborOffsets), std::move(neighborVertexIds));
    }

    std::atomic<uint64_t> nextPathId{1};
    std::atomic<uint64_t> nextVertexId{1};
    std::mutex sinksMutex;
    std::vector<ThreadLocalSink *> sinks;
};

namespace {

// The renderer clears the "active" builder before calling ClusterPathGraph(),
// so keep a second pointer to the most recently registered builder for the
// post-process stage.
std::mutex activeBuilderMutex;
PathGraphBuilder *activeBuilder = nullptr;

}  // namespace

BasicPathGraphBuilder::BasicPathGraphBuilder() : impl(std::make_unique<Impl>()) {}

BasicPathGraphBuilder::~BasicPathGraphBuilder() = default;

PathGraphSink *BasicPathGraphBuilder::GetThreadLocalSink() { return impl->GetThreadLocalSink(); }

std::shared_ptr<const PathGraphSnapshot> BasicPathGraphBuilder::BuildSnapshot() {
    return impl->BuildSnapshot();
}

PathGraphBuilder *GetActivePathGraphBuilder() {
    std::lock_guard<std::mutex> lock(activeBuilderMutex);
    return activeBuilder;
}

void SetActivePathGraphBuilder(PathGraphBuilder *builder) {
    std::lock_guard<std::mutex> lock(activeBuilderMutex);
    activeBuilder = builder;
    if (builder)
        lastBuilderForCluster = builder;
}

void ClusterPathGraph() {
    PathGraphBuilder *builder = nullptr;
    {
        std::lock_guard<std::mutex> lock(activeBuilderMutex);
        builder = lastBuilderForCluster;
    }
    if (!builder) {
        LOG_VERBOSE("Path graph clustering skipped: no captured builder");
        return;
    }

    auto snapshot = builder->BuildSnapshot();
    if (!snapshot) {
        LOG_VERBOSE("Path graph clustering skipped: snapshot build failed");
        return;
    }

    // The current pathGraph.h interface only exposes graph data and radiance
    // solve inputs. Image reconstruction is intentionally left outside this
    // function; here we only perform the paper's aggregation / propagation /
    // finalGather stages on the captured path graph.
    PathGraphSolveState state = SolvePathGraph(*snapshot, kDefaultContIterations);

    LOG_VERBOSE("Path graph solve finished: %zu vertices, %zu light edges, %zu cont edges, "
                "%d cont iterations",
                snapshot->Vertices().size(), snapshot->LightEdges().size(),
                snapshot->ContEdges().size(), kDefaultContIterations);
    LOG_VERBOSE("Path graph buffers: direct %zu, iterated %zu, final %zu, edge %zu",
                state.directVertexRadiance.size(), state.iteratedVertexRadiance.size(),
                state.finalGatherVertexRadiance.size(), state.contEdgeRadiance.size());
}

}  // namespace pbrt
