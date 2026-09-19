namespace FaceRecognize.Abstractions;

/// <summary>Запрос: зарегистрировать лицо с изображения как известное с именем <see cref="Name"/>.</summary>
public sealed record AddToKnownRequest(
    string ImagePath,
    string Name,
    float Threshold,
    IReadOnlyList<ClusterFace> CurrentUnknownFaces);

/// <summary>Результат: кто из оставшихся неизвестных лиц теперь распознан и новые кластеры.</summary>
public sealed class AddToKnownResult
{
    public List<RecognizedFace> NewlyMatched { get; init; } = [];
    public List<PersonCluster> RemainingClusters { get; init; } = [];
}

/// <summary>Итог по одному файлу при регистрации: сколько лиц добавлено и причина пропуска (null — успех).</summary>
public sealed record EnrollFileOutcome(string Path, int FacesAdded, string? Error)
{
    public bool Skipped => Error is not null;
}

/// <summary>Результат пакетной регистрации: суммарное число лиц + итоги по каждому файлу.</summary>
public sealed record EnrollByNameResult(int FacesAdded, IReadOnlyList<EnrollFileOutcome> Files)
{
    public IReadOnlyList<EnrollFileOutcome> SkippedFiles => Files.Where(f => f.Skipped).ToList();
}

/// <summary>
/// Use-case: регистрация лиц в хранилище известных — по имени (пакет файлов)
/// или для конкретного изображения с повторной идентификацией и
/// пере-кластеризацией оставшихся неизвестных лиц.
/// </summary>
public interface IFaceEnrollmentService
{
    /// <summary>Последнее имя, использованное при регистрации (держится в памяти сессии, общее для всех диалогов).</summary>
    string LastEnrolledName { get; }

    /// <summary>Отсортированный список известных имён (хранилище + дополнительные, если заданы).</summary>
    List<string> GetExistingNames(IReadOnlyList<string>? extraNames = null);

    /// <summary>Регистрирует все лица из файлов под заданным именем; возвращает итоги по каждому файлу.</summary>
    EnrollByNameResult EnrollByName(string name, IReadOnlyList<string> imagePaths, IProgress<int>? progress = null);

    AddToKnownResult AddToKnown(AddToKnownRequest request);
}