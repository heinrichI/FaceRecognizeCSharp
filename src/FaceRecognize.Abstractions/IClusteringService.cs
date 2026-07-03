namespace FaceRecognize.Abstractions;

public interface IClusteringService
{
    List<PersonCluster> ClusterUnknownFaces(
        List<(string imagePath, float[] embedding, byte[]? thumbnail)> unknownFaces,
        float eps = 0.42f,
        int minSamples = 2);
}

public class PersonCluster
{
    public int ClusterId { get; set; }
    public List<ClusterFace> Faces { get; set; } = [];
    public float[] Centroid { get; set; } = [];
}

public class ClusterFace
{
    public string ImagePath { get; set; } = string.Empty;
    public float[] Embedding { get; set; } = [];
    public byte[]? Thumbnail { get; set; }
}
