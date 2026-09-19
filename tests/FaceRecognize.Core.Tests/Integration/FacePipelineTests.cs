using FaceRecognize.Abstractions;
using FaceRecognize.Core.Services;
using FaceRecognize.Data;
using Xunit;

namespace FaceRecognize.Core.Tests.Integration;

public class FacePipelineTests : IDisposable
{
    private readonly FaceRecognizer _recognizer;
    private readonly IClusteringService _clustering = new ClusteringService();
    private readonly string _dbPath;
    private IVectorStore? _vectorStore;

    public FacePipelineTests()
    {
        _recognizer = new FaceRecognizer();
        _dbPath = Path.Combine(Path.GetTempPath(), $"test_pipeline_{Guid.NewGuid():N}.db");
        _vectorStore = new SqliteVectorStore(_dbPath);
    }

    [Fact]
    public void FullPipeline_EnrollAndRecognize()
    {
        if (!TestConfig.IsAvailable) return;

        var personFolder = Path.Combine(TestConfig.TrainDataRoot, "Scarlett Johansson");
        if (!Directory.Exists(personFolder)) return;

        var images = TestConfig.GetImages(personFolder, 5);
        if (images.Length < 3) return;

        for (int i = 0; i < 3; i++)
        {
            var embedding = _recognizer.ExtractEmbedding(images[i]);
            _vectorStore!.Add(new KnownFace
            {
                Name = "Scarlett Johansson",
                ImagePath = images[i],
                Embedding = embedding
            });
        }

        var queryEmbedding = _recognizer.ExtractEmbedding(images[3]);
        var (best, score) = _vectorStore!.Search(queryEmbedding, 0.5f);

        Assert.NotNull(best);
        Assert.Equal("Scarlett Johansson", best!.Name);
        Assert.True(score > 0.3f, $"Match score too low: {score:F4}");
    }

    [Fact]
    public void FullPipeline_ScanDirectory()
    {
        if (!TestConfig.IsAvailable) return;

        var personFolder = Path.Combine(TestConfig.TrainDataRoot, "Scarlett Johansson");
        if (!Directory.Exists(personFolder)) return;

        var images = TestConfig.GetImages(personFolder, 10);
        if (images.Length < 5) return;

        for (int i = 0; i < 3; i++)
        {
            var embedding = _recognizer.ExtractEmbedding(images[i]);
            _vectorStore!.Add(new KnownFace
            {
                Name = "Scarlett Johansson",
                ImagePath = images[i],
                Embedding = embedding
            });
        }

        int recognized = 0;
        for (int i = 3; i < Math.Min(8, images.Length); i++)
        {
            var faces = _recognizer.ExtractAllFaces(images[i]);
            foreach (var face in faces)
            {
                var (best, _) = _vectorStore!.Search(face.Embedding, 0.5f);
                if (best?.Name == "Scarlett Johansson")
                    recognized++;
            }
        }

        Assert.True(recognized > 0, "No faces recognized in scan");
    }

    [Fact]
    public void FullPipeline_UnknownFacesClustered()
    {
        if (!TestConfig.IsAvailable) return;

        var enrollFolder = Path.Combine(TestConfig.TrainDataRoot, "Scarlett Johansson");
        var scanFolder = Path.Combine(TestConfig.TrainDataRoot, "Emma Stone");
        if (!Directory.Exists(enrollFolder) || !Directory.Exists(scanFolder)) return;

        var enrollImages = TestConfig.GetImages(enrollFolder, 2);
        var scanImages = TestConfig.GetImages(scanFolder, 5);
        if (enrollImages.Length < 1 || scanImages.Length < 3) return;

        var embedding = _recognizer.ExtractEmbedding(enrollImages[0]);
        _vectorStore!.Add(new KnownFace
        {
            Name = "Scarlett Johansson",
            ImagePath = enrollImages[0],
            Embedding = embedding
        });

        var unknownEmbeddings = new List<(string path, float[] embedding, byte[]? thumb)>();
        foreach (var img in scanImages)
        {
            try
            {
                var faces = _recognizer.ExtractAllFaces(img);
                foreach (var face in faces)
                {
                    var (best, _) = _vectorStore.Search(face.Embedding, 0.5f);
                    if (best == null)
                        unknownEmbeddings.Add((img, face.Embedding, null));
                }
            }
            catch { }
        }

        if (unknownEmbeddings.Count < 2) return;

        var clusters = _clustering.ClusterUnknownFaces(unknownEmbeddings);

        Assert.True(clusters.Count >= 1, "No clusters formed for unknown faces");
    }

    public void Dispose()
    {
        _recognizer.Dispose();
        _vectorStore?.Dispose();
        _vectorStore = null;
        try { File.Delete(_dbPath); } catch { }
    }
}
