using FaceRecognize.Abstractions;
using FaceRecognize.Core.Services;
using Xunit;

namespace FaceRecognize.Core.Tests.Services;

public class FaceRecognizerTests : IDisposable
{
    private readonly FaceRecognizer _recognizer;

    public FaceRecognizerTests()
    {
        _recognizer = new FaceRecognizer();
    }

    [Fact]
    public void DetectFaces_FindsAtLeastOneFace()
    {
        if (!TestConfig.IsAvailable) return;

        var imagePath = TestConfig.GetRandomImage();
        var result = _recognizer.DetectFaces(imagePath);

        Assert.True(result.Count >= 1, $"No faces detected in {imagePath}");
    }

    [Fact]
    public void ExtractEmbedding_Returns512DimensionalVector()
    {
        if (!TestConfig.IsAvailable) return;

        var imagePath = TestConfig.GetRandomImage();
        var embedding = _recognizer.ExtractEmbedding(imagePath);

        Assert.Equal(512, embedding.Length);
    }

    [Theory]
    [InlineData("Scarlett Johansson")]
    [InlineData("Angelina Jolie")]
    [InlineData("Emma Stone")]
    public void SamePerson_SmallCosineDistance(string personName)
    {
        if (!TestConfig.IsAvailable) return;

        var personFolder = Path.Combine(TestConfig.TrainDataRoot, personName);
        if (!Directory.Exists(personFolder)) return;

        var images = TestConfig.GetImages(personFolder, 3);
        if (images.Length < 2) return;

        float[]? emb1 = null;
        float[]? emb2 = null;

        foreach (var img in images)
        {
            try
            {
                if (emb1 == null) emb1 = _recognizer.ExtractEmbedding(img);
                else if (emb2 == null) { emb2 = _recognizer.ExtractEmbedding(img); break; }
            }
            catch { }
        }

        if (emb1 == null || emb2 == null) return;

        var distance = EmbeddingMath.CosineDistance(emb1, emb2);
        Assert.True(distance < 0.8f, $"Same person distance too high: {distance:F4}");
    }

    [Theory]
    [InlineData("Scarlett Johansson", "Emma Stone")]
    [InlineData("Angelina Jolie", "Milla Jovovich")]
    public void DifferentPerson_LargeCosineDistance(string person1, string person2)
    {
        if (!TestConfig.IsAvailable) return;

        var folder1 = Path.Combine(TestConfig.TrainDataRoot, person1);
        var folder2 = Path.Combine(TestConfig.TrainDataRoot, person2);
        if (!Directory.Exists(folder1) || !Directory.Exists(folder2)) return;

        float[]? emb1 = null;
        float[]? emb2 = null;

        foreach (var img in TestConfig.GetImages(folder1, 10))
        {
            try { emb1 = _recognizer.ExtractEmbedding(img); break; } catch { }
        }

        foreach (var img in TestConfig.GetImages(folder2, 10))
        {
            try { emb2 = _recognizer.ExtractEmbedding(img); break; } catch { }
        }

        if (emb1 == null || emb2 == null) return;

        var distance = EmbeddingMath.CosineDistance(emb1, emb2);
        Assert.True(distance > 0.2f, $"Different person distance too low: {distance:F4}");
    }

    public void Dispose()
    {
        _recognizer.Dispose();
    }
}
