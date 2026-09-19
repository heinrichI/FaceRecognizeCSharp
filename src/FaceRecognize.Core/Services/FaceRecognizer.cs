using System.IO;
using FaceAiSharp;
using FaceAiSharp.Extensions;
using FaceRecognize.Abstractions;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceRecognize.Core.Services;

public class FaceRecognizer : IFaceRecognizer
{
    private readonly IFaceDetectorWithLandmarks _detector;
    private readonly IFaceEmbeddingsGenerator _embedder;
    private readonly ThreeDAlignmentService? _alignment3D;

    public bool Is3DAlignmentAvailable => _alignment3D != null;

    public FaceRecognizer(string? landmarkOnnxPath = null)
    {
        var opts = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        _detector = FaceAiSharpBundleFactory.CreateFaceDetectorWithLandmarks(opts);
        _embedder = FaceAiSharpBundleFactory.CreateFaceEmbeddingsGenerator(opts);

        if (landmarkOnnxPath != null && File.Exists(landmarkOnnxPath))
        {
            _alignment3D = new ThreeDAlignmentService(landmarkOnnxPath, opts);
        }
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

        if (_alignment3D != null)
        {
            try
            {
                return ExtractEmbedding3D(imagePath, face.Box);
            }
            catch
            {
                // Fall through to 2D
            }
        }

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
            float[]? embedding = null;

            if (_alignment3D != null)
            {
                try
                {
                    embedding = ExtractEmbedding3D(imagePath, face.Box);
                }
                catch
                {
                    embedding = null;
                }
            }

            if (embedding == null)
            {
                using var cloned = img.Clone();
                _embedder.AlignFaceUsingLandmarks(cloned, face.Landmarks!);
                embedding = _embedder.GenerateEmbedding(cloned);
            }

            results.Add(new FaceWithEmbedding
            {
                Box = face.Box,
                Confidence = face.Confidence,
                Embedding = embedding
            });
        }

        return results;
    }

    public HeadPose? GetHeadPose(string imagePath)
    {
        if (_alignment3D == null)
            return null;

        using var sourceMat = Cv2.ImRead(imagePath);
        if (sourceMat.Empty())
            return null;

        var img = Image.Load<Rgb24>(imagePath);
        var faces = _detector.DetectFaces(img);
        img.Dispose();
        if (faces.Count == 0)
            return null;

        var face = faces.First();
        var bbox = new Rect(
            (int)face.Box.X,
            (int)face.Box.Y,
            (int)face.Box.Width,
            (int)face.Box.Height);

        bbox = ClampRect(bbox, sourceMat.Width, sourceMat.Height);
        return _alignment3D.EstimateHeadPose(sourceMat, bbox);
    }

    private float[] ExtractEmbedding3D(string imagePath, RectangleF box)
    {
        using var sourceMat = Cv2.ImRead(imagePath);
        var bbox = new Rect(
            (int)box.X,
            (int)box.Y,
            (int)box.Width,
            (int)box.Height);

        bbox = ClampRect(bbox, sourceMat.Width, sourceMat.Height);
        using var aligned = _alignment3D!.AlignFace3D(sourceMat, bbox);

        Cv2.ImEncode(".png", aligned, out var buf);
        using var ms = new MemoryStream(buf);
        using var img = Image.Load<Rgb24>(ms);
        return _embedder.GenerateEmbedding(img);
    }

    private static Rect ClampRect(Rect r, int maxWidth, int maxHeight)
    {
        int x = Math.Max(0, r.X);
        int y = Math.Max(0, r.Y);
        int w = Math.Min(r.Width, maxWidth - x);
        int h = Math.Min(r.Height, maxHeight - y);
        return new Rect(x, y, Math.Max(w, 1), Math.Max(h, 1));
    }

    public static float DotProduct(float[] a, float[] b) => a.Dot(b);

    public static float CosineDistance(float[] a, float[] b) => 1f - a.Dot(b);

    public void Dispose()
    {
        (_detector as IDisposable)?.Dispose();
        (_embedder as IDisposable)?.Dispose();
        _alignment3D?.Dispose();
    }
}
