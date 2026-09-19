namespace FaceRecognize.Abstractions;

/// <summary>Отчёт о прогрессе сканирования каталога.</summary>
public readonly record struct ScanProgress(int Completed, int Total, string FileName);

/// <summary>
/// Use-case: сканирование каталога — обнаружение лиц, поиск в хранилище
/// известных лиц, DBSCAN-кластеризация оставшихся (неизвестных) лиц.
/// </summary>
public interface IFaceScanService
{
    ScanResult ScanDirectory(
        string directory,
        float threshold,
        int maxDegreeOfParallelism,
        IProgress<ScanProgress>? progress,
        CancellationToken cancellationToken = default);
}