using FaceRecognize.Abstractions;
using FaceRecognize.Data;
using Xunit;

namespace FaceRecognize.Core.Tests.Services;

public class VectorStoreTests : IDisposable
{
    private readonly string _dbPath;
    private IVectorStore? _store;

    public VectorStoreTests()
    {
        _dbPath = Path.Combine(Path.GetTempPath(), $"test_{Guid.NewGuid():N}.db");
        _store = new SqliteVectorStore(_dbPath);
    }

    [Fact]
    public void Add_IncreasesCount()
    {
        var face = CreateTestFace("Alice");
        _store!.Add(face);

        Assert.Equal(1, _store.Count);
    }

    [Fact]
    public void Add_And_Search_RoundTrip()
    {
        var embedding1 = CreateDistinctEmbedding(1);
        var embedding2 = CreateDistinctEmbedding(2);

        _store!.Add(CreateTestFace("Alice", embedding1));
        _store.Add(CreateTestFace("Bob", embedding2));

        var (best, score) = _store.Search(embedding1);

        Assert.NotNull(best);
        Assert.Equal("Alice", best!.Name);
    }

    [Fact]
    public void Delete_RemovesFace()
    {
        var face = CreateTestFace("Alice");
        _store!.Add(face);
        Assert.Equal(1, _store.Count);

        _store.Delete(face.Id);
        Assert.Equal(0, _store.Count);
    }

    [Fact]
    public void ClearAll_RemovesAllFaces()
    {
        _store!.Add(CreateTestFace("Alice"));
        _store.Add(CreateTestFace("Bob"));
        Assert.Equal(2, _store.Count);

        _store.ClearAll();
        Assert.Equal(0, _store.Count);
    }

    [Fact]
    public void GetAll_ReturnsAllFaces()
    {
        _store!.Add(CreateTestFace("Alice"));
        _store.Add(CreateTestFace("Bob"));

        var all = _store.GetAll();
        Assert.Equal(2, all.Count);
    }

    [Fact]
    public void Search_ReturnsNullBelowThreshold()
    {
        var embedding1 = new float[512];
        embedding1[0] = 1.0f;

        var embedding2 = new float[512];
        embedding2[1] = 1.0f;

        _store!.Add(CreateTestFace("Alice", embedding1));

        var (best, score) = _store.Search(embedding2, threshold: 0.99f);
        Assert.Null(best);
    }

    private static KnownFace CreateTestFace(string name, float[]? embedding = null)
    {
        return new KnownFace
        {
            Name = name,
            ImagePath = $@"C:\test\{name}.jpg",
            Embedding = embedding ?? CreateDistinctEmbedding(0)
        };
    }

    private static float[] CreateDistinctEmbedding(int seed)
    {
        var rng = new Random(seed * 1000);
        var embedding = new float[512];
        for (int i = 0; i < 512; i++)
            embedding[i] = (float)rng.NextDouble();
        return embedding;
    }

    public void Dispose()
    {
        _store?.Dispose();
        _store = null;
        try { File.Delete(_dbPath); } catch { }
    }
}
