using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace FaceRecognize.Abstractions;

public interface IFaceRecognizer : IDisposable
{
    List<FaceDetectionResult> DetectFaces(string imagePath);
    float[] ExtractEmbedding(string imagePath);
    List<FaceWithEmbedding> ExtractAllFaces(string imagePath);
    HeadPose? GetHeadPose(string imagePath);
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

public class HeadPose
{
    public double Yaw { get; set; }
    public double Pitch { get; set; }
    public double Roll { get; set; }
}
