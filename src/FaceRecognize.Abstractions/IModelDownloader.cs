namespace FaceRecognize.Abstractions;

public interface IModelDownloader
{
    Task<bool> EnsureModelExistsAsync(string modelPath, string downloadUrl, IProgress<int>? progress = null, CancellationToken ct = default);
    bool IsModelPresent(string modelPath);
}
