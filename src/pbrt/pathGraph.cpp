#include <pbrt/pathGraph.h>

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
                           std::vector<NeighborEdge> neighborEdges)
        : vertices(std::move(vertices)),
          lightEdges(std::move(lightEdges)),
          contEdges(std::move(contEdges)),
          neighborEdges(std::move(neighborEdges)) {}

    pstd::span<const SurfaceVertex> Vertices() const override { return vertices; }
    pstd::span<const LightEdge> LightEdges() const override { return lightEdges; }
    pstd::span<const ContEdge> ContEdges() const override { return contEdges; }
    pstd::span<const NeighborEdge> NeighborEdges() const override {
        return neighborEdges;
    }

  private:
    std::vector<SurfaceVertex> vertices;
    std::vector<LightEdge> lightEdges;
    std::vector<ContEdge> contEdges;
    std::vector<NeighborEdge> neighborEdges;
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
                pendingContEdge->toVertexId = stored.vertexId;
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
            if (stored.fromVertexId == 0)
                stored.fromVertexId = lastVertexId;
            lightEdges.push_back(stored);
        }

        void AddContEdge(const ContEdge &e) override {
            if (!inPath)
                BeginPath(0);

            ContEdge stored = e;
            if (stored.fromVertexId == 0)
                stored.fromVertexId = lastVertexId;
            if (stored.toVertexId == 0)
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
        return std::make_shared<BasicPathGraphSnapshot>(
            std::move(allVertices), std::move(allLightEdges), std::move(allContEdges),
            std::vector<NeighborEdge>{});
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
