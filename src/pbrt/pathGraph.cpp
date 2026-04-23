#include <pbrt/pathGraph.h>

#include <algorithm>
#include <atomic>
#include <mutex>
#include <vector>

namespace pbrt {

namespace {

constexpr uint64_t kCameraRootVertexId = 0;

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
        for (const SurfaceVertex &v : allVertices)
            maxVertexId = std::max(maxVertexId, v.vertexId);
        for (const LightEdge &e : allLightEdges)
            maxVertexId = std::max(maxVertexId, e.vertexA);
        for (const ContEdge &e : allContEdges) {
            maxVertexId = std::max(maxVertexId, e.vertexA);
            maxVertexId = std::max(maxVertexId, e.vertexB);
        }

        std::vector<uint32_t> contEdgeOffsets(maxVertexId + 2, 0);
        std::vector<uint32_t> lightEdgeOffsets(maxVertexId + 2, 0);
        std::vector<uint32_t> neighborOffsets(maxVertexId + 2, 0);

        for (const ContEdge &e : allContEdges)
            ++contEdgeOffsets[e.vertexB + 1];
        for (const LightEdge &e : allLightEdges)
            ++lightEdgeOffsets[e.vertexA + 1];

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
        std::vector<uint32_t> neighborCursor = neighborOffsets;

        for (uint32_t i = 0; i < allContEdges.size(); ++i) {
            const ContEdge &e = allContEdges[i];
            contEdgeIndices[contEdgeCursor[e.vertexB]++] = i;
        }
        for (uint32_t i = 0; i < allLightEdges.size(); ++i) {
            const LightEdge &e = allLightEdges[i];
            lightEdgeIndices[lightEdgeCursor[e.vertexA]++] = i;
        }

        return std::make_shared<BasicPathGraphSnapshot>(
            std::move(allVertices), std::move(allLightEdges), std::move(allContEdges),
            std::move(contEdgeIndices), std::move(lightEdgeOffsets),
            std::move(lightEdgeIndices), std::move(neighborOffsets),
            std::move(neighborVertexIds));
    }

    std::atomic<uint64_t> nextPathId{1};
    std::atomic<uint64_t> nextVertexId{1};
    std::mutex sinksMutex;
    std::vector<ThreadLocalSink *> sinks;
};

namespace {

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
}

}  // namespace pbrt
