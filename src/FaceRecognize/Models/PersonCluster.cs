namespace FaceRecognize.Models;

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
