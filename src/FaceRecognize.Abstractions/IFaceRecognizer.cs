namespace FaceRecognize.Abstractions;

public interface IFaceRecognizer : IDisposable
{
    bool Is3DAlignmentAvailable { get; }
    float[] ExtractEmbedding(string imagePath);
    List<FaceWithEmbedding> ExtractAllFaces(string imagePath);
}

public class FaceDetectionResult
{
    public FaceBox Box { get; set; }
    public float? Confidence { get; set; }
    public IReadOnlyList<(float X, float Y)>? Landmarks { get; set; }
}

public class FaceWithEmbedding
{
    public FaceBox Box { get; set; }
    public float? Confidence { get; set; }
    public float[] Embedding { get; set; } = [];
}