using FaceAiSharp;
using FaceAiSharp.Extensions;
using Microsoft.ML.OnnxRuntime;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceRecognize.Services;

public class FaceRecognizer : IDisposable
{
    private readonly IFaceDetectorWithLandmarks _detector;
    private readonly IFaceEmbeddingsGenerator _embedder;

    public FaceRecognizer()
    {
        var opts = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        _detector = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks(opts);
        _embedder = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator(opts);
    }

    public List<FaceDetectionResult> DetectFaces(string imagePath)
    {
        using var img = Image.Load<Rgb24>(imagePath);
        var faces = _detector.DetectFaces(img);
        return faces.Select(f => new FaceDetectionResult
        {
            Box = f.Box,
            Confidence = f.Confidence,
            Landmarks = f.Landmarks
        }).ToList();
    }

    public float[] ExtractEmbedding(string imagePath)
    {
        using var img = Image.Load<Rgb24>(imagePath);
        var faces = _detector.DetectFaces(img);
        if (faces.Count == 0)
            throw new InvalidOperationException("No faces found in image.");

        var face = faces.First();
        using var cloned = img.Clone();
        _embedder.AlignFaceUsingLandmarks(cloned, face.Landmarks!);
        return _embedder.GenerateEmbedding(cloned);
    }

    public List<FaceWithEmbedding> ExtractAllFaces(string imagePath)
    {
        using var img = Image.Load<Rgb24>(imagePath);
        var faces = _detector.DetectFaces(img);
        var results = new List<FaceWithEmbedding>();

        foreach (var face in faces)
        {
            using var cloned = img.Clone();
            _embedder.AlignFaceUsingLandmarks(cloned, face.Landmarks!);
            var embedding = _embedder.GenerateEmbedding(cloned);
            results.Add(new FaceWithEmbedding
            {
                Box = face.Box,
                Confidence = face.Confidence,
                Embedding = embedding
            });
        }

        return results;
    }

    public static float DotProduct(float[] a, float[] b) => a.Dot(b);

    public static float CosineDistance(float[] a, float[] b) => 1f - a.Dot(b);

    public void Dispose()
    {
        (_detector as IDisposable)?.Dispose();
        (_embedder as IDisposable)?.Dispose();
    }
}

public class FaceDetectionResult
{
    public RectangleF Box { get; set; }
    public float? Confidence { get; set; }
    public IReadOnlyList<PointF>? Landmarks { get; set; }
}

public class FaceWithEmbedding
{
    public RectangleF Box { get; set; }
    public float? Confidence { get; set; }
    public float[] Embedding { get; set; } = [];
}
