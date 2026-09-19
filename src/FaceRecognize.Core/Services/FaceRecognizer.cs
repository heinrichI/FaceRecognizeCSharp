using System.Diagnostics;
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
            Box = new FaceBox(f.Box.X, f.Box.Y, f.Box.Width, f.Box.Height),
            Confidence = f.Confidence,
            Landmarks = f.Landmarks?.Select(p => (p.X, p.Y)).ToList()
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
                return ExtractEmbedding3D(imagePath, ToFaceBox(face.Box));
            }
            catch
            {
                // Fall through to 2D
            }
        }

        using var cloned = img.Clone();
        _embedder.AlignFaceUsingLandmarks(cloned, face.Landmarks!);
        var embedding = _embedder.GenerateEmbedding(cloned);
        Debug.Assert(embedding is { Length: > 0 }, "Embedding generator returned empty vector");
        return embedding;
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
                    embedding = ExtractEmbedding3D(imagePath, ToFaceBox(face.Box));
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

            Debug.Assert(embedding is { Length: > 0 }, "Face embedding must not be empty");

            results.Add(new FaceWithEmbedding
            {
                Box = new FaceBox(face.Box.X, face.Box.Y, face.Box.Width, face.Box.Height),
                Confidence = face.Confidence,
                Embedding = embedding
            });
        }

        Debug.Assert(results.Count == faces.Count, "ExtractAllFaces result count mismatch");
        return results;
    }

    private float[] ExtractEmbedding3D(string imagePath, FaceBox box)
    {
        using var sourceMat = Cv2.ImRead(imagePath);
        var bbox = new Rect(
            (int)box.X,
            (int)box.Y,
            (int)box.Width,
            (int)box.Height);

        bbox = ClampRect(bbox, sourceMat.Width, sourceMat.Height);
        using var aligned = _alignment3D!.AlignFace3D(sourceMat, bbox);
        Debug.Assert(!aligned.Empty(), "3D alignment produced empty image");

        Cv2.ImEncode(".png", aligned, out var buf);
        using var ms = new MemoryStream(buf);
        using var img = Image.Load<Rgb24>(ms);
        var embedding = _embedder.GenerateEmbedding(img);
        Debug.Assert(embedding is { Length: > 0 }, "Embedding generator returned empty vector");
        return embedding;
    }

    private static Rect ClampRect(Rect r, int maxWidth, int maxHeight)
    {
        int x = Math.Max(0, r.X);
        int y = Math.Max(0, r.Y);
        int w = Math.Min(r.Width, maxWidth - x);
        int h = Math.Min(r.Height, maxHeight - y);
        var rect = new Rect(x, y, Math.Max(w, 1), Math.Max(h, 1));
        Debug.Assert(rect.X >= 0 && rect.Y >= 0 && rect.Width >= 1 && rect.Height >= 1,
            "Clamped rect must be non-negative and non-zero sized");
        return rect;
    }

    private static FaceBox ToFaceBox(SixLabors.ImageSharp.RectangleF box)
        => new(box.X, box.Y, box.Width, box.Height);

    public void Dispose()
    {
        (_detector as IDisposable)?.Dispose();
        (_embedder as IDisposable)?.Dispose();
        _alignment3D?.Dispose();
    }
}
