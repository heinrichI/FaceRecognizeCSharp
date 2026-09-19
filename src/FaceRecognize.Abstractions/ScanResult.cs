namespace FaceRecognize.Abstractions;

public class ScanResult
{
    public List<RecognizedFace> KnownFaces { get; set; } = [];
    public List<PersonCluster> UnknownClusters { get; set; } = [];
    public int TotalImagesScanned { get; set; }
    public int TotalFacesFound { get; set; }
    public TimeSpan Duration { get; set; }

    /// <summary>True, если сканирование было остановлено пользователем (результат частичный).</summary>
    public bool Cancelled { get; set; }
}

public class RecognizedFace
{
    public string ImagePath { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public float Confidence { get; set; }
    public byte[]? Thumbnail { get; set; }
}