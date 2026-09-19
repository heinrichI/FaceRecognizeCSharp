namespace FaceRecognize.Abstractions;

public interface IEmbeddingCache : IDisposable
{
    List<CachedFace>? GetCachedFaces(string imagePath, long fileSize, DateTime lastModified);
    void Store(string imagePath, long fileSize, DateTime lastModified, List<CachedFace> faces);
    void Clear();
}

public class CachedFace
{
    public float[] Embedding { get; set; } = [];
    public float X { get; set; }
    public float Y { get; set; }
    public float Width { get; set; }
    public float Height { get; set; }
}
