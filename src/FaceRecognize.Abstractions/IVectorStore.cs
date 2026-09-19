namespace FaceRecognize.Abstractions;

public interface IVectorStore : IDisposable
{
    void Add(KnownFace face);
    void Delete(Guid id);
    void ClearAll();
    (KnownFace? best, float score) Search(float[] embedding, float threshold);
    int Count { get; }
    List<KnownFace> GetAll();

    /// <summary>Перечитывает хранилище из файла БД (синхронизирует в-памяти список с диском).</summary>
    void Reload();
}

public class KnownFace
{
    public Guid Id { get; set; } = Guid.NewGuid();
    public string Name { get; set; } = string.Empty;
    public string ImagePath { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public float[] Embedding { get; set; } = [];
    public byte[]? Thumbnail { get; set; }
}
