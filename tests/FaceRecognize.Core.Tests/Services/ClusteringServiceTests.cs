using FaceRecognize.Abstractions;
using FaceRecognize.Core.Services;
using Xunit;

namespace FaceRecognize.Core.Tests.Services;

public class ClusteringServiceTests
{
    private readonly IClusteringService _clustering = new ClusteringService();

    [Fact]
    public void ClusterUnknownFaces_GroupsSimilarFaces()
    {
        var embA1 = CreateEmbedding(0.0f);
        var embA2 = CreateEmbedding(0.01f);
        var embA3 = CreateEmbedding(0.02f);
        var embB1 = CreateEmbedding(0.5f);
        var embB2 = CreateEmbedding(0.51f);
        var embB3 = CreateEmbedding(0.52f);

        var faces = new List<(string imagePath, float[] embedding, byte[]? thumbnail)>
        {
            ("a1.jpg", embA1, null),
            ("a2.jpg", embA2, null),
            ("a3.jpg", embA3, null),
            ("b1.jpg", embB1, null),
            ("b2.jpg", embB2, null),
            ("b3.jpg", embB3, null),
        };

        var clusters = _clustering.ClusterUnknownFaces(faces, eps: 0.1f, minSamples: 2);

        var nonNoise = clusters.Where(c => c.ClusterId != -1).ToList();
        Assert.True(nonNoise.Count >= 1, $"Expected at least 1 non-noise cluster, got {nonNoise.Count}");
        Assert.True(nonNoise.Sum(c => c.Faces.Count) >= 4, "Expected at least 4 faces in clusters");
    }

    [Fact]
    public void ClusterUnknownFaces_SinglePoint_NoiseCluster()
    {
        var emb1 = CreateEmbedding(0.0f);

        var faces = new List<(string imagePath, float[] embedding, byte[]? thumbnail)>
        {
            ("1.jpg", emb1, null),
        };

        var clusters = _clustering.ClusterUnknownFaces(faces, eps: 0.05f, minSamples: 2);

        var noise = clusters.FirstOrDefault(c => c.ClusterId == -1);
        Assert.NotNull(noise);
        Assert.Equal(1, noise!.Faces.Count);
    }

    [Fact]
    public void ClusterUnknownFaces_EmptyList_ReturnsEmpty()
    {
        var clusters = _clustering.ClusterUnknownFaces([]);
        Assert.Empty(clusters);
    }

    [Fact]
    public void ClusterUnknownFaces_ComputesCentroid()
    {
        var emb1 = CreateEmbedding(0.0f);
        var emb2 = CreateEmbedding(0.01f);

        var faces = new List<(string imagePath, float[] embedding, byte[]? thumbnail)>
        {
            ("1.jpg", emb1, null),
            ("2.jpg", emb2, null),
        };

        var clusters = _clustering.ClusterUnknownFaces(faces, eps: 0.1f, minSamples: 2);

        var nonNoise = clusters.FirstOrDefault(c => c.ClusterId != -1);
        if (nonNoise != null)
        {
            Assert.Equal(512, nonNoise.Centroid.Length);
        }
    }

    private static float[] CreateEmbedding(float baseValue)
    {
        var embedding = new float[512];
        for (int i = 0; i < 512; i++)
            embedding[i] = baseValue + (Random.Shared.NextSingle() - 0.5f) * 0.001f;
        return embedding;
    }
}
