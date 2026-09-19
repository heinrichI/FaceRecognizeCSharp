namespace FaceRecognize.Abstractions;

public interface IConfiguration
{
    string LastScanPath { get; set; }
    double DistanceThreshold { get; set; }
    int ThreadCount { get; set; }
    string ModelsDirectory { get; set; }
    void Save();
}
