using System.IO;
using FaceRecognize.Models;
using Microsoft.Data.Sqlite;

namespace FaceRecognize.Services;

public class VectorStore : IDisposable
{
    private readonly string _dbPath;
    private readonly List<KnownFace> _faces = [];
    private readonly object _lock = new();
    private SqliteConnection? _connection;

    public VectorStore(string dbPath)
    {
        _dbPath = dbPath;
        Initialize();
    }

    private void Initialize()
    {
        var dir = Path.GetDirectoryName(_dbPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        _connection = new SqliteConnection($"Data Source={_dbPath}");
        _connection.Open();

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS known_faces (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                embedding BLOB NOT NULL
            )
            """;
        cmd.ExecuteNonQuery();

        LoadAll();
    }

    private void LoadAll()
    {
        using var cmd = _connection!.CreateCommand();
        cmd.CommandText = "SELECT id, name, image_path, created_at, embedding FROM known_faces";

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var face = new KnownFace
            {
                Id = Guid.Parse(reader.GetString(0)),
                Name = reader.GetString(1),
                ImagePath = reader.GetString(2),
                CreatedAt = DateTime.Parse(reader.GetString(3)),
                Embedding = BytesToFloats(reader.GetFieldValue<byte[]>(4))
            };
            _faces.Add(face);
        }
    }

    public void Add(KnownFace face)
    {
        using var cmd = _connection!.CreateCommand();
        cmd.CommandText = """
            INSERT INTO known_faces (id, name, image_path, created_at, embedding)
            VALUES ($id, $name, $image_path, $created_at, $embedding)
            """;
        cmd.Parameters.AddWithValue("$id", face.Id.ToString());
        cmd.Parameters.AddWithValue("$name", face.Name);
        cmd.Parameters.AddWithValue("$image_path", face.ImagePath);
        cmd.Parameters.AddWithValue("$created_at", face.CreatedAt.ToString("O"));
        cmd.Parameters.AddWithValue("$embedding", FloatsToBytes(face.Embedding));
        cmd.ExecuteNonQuery();

        lock (_lock)
        {
            _faces.Add(face);
        }
    }

    public void Delete(Guid id)
    {
        using var cmd = _connection!.CreateCommand();
        cmd.CommandText = "DELETE FROM known_faces WHERE id = $id";
        cmd.Parameters.AddWithValue("$id", id.ToString());
        cmd.ExecuteNonQuery();

        lock (_lock)
        {
            _faces.RemoveAll(f => f.Id == id);
        }
    }

    public void ClearAll()
    {
        using var cmd = _connection!.CreateCommand();
        cmd.CommandText = "DELETE FROM known_faces";
        cmd.ExecuteNonQuery();

        lock (_lock)
        {
            _faces.Clear();
        }
    }

    public (KnownFace? best, float score) Search(float[] embedding, float threshold = 0.58f)
    {
        lock (_lock)
        {
            KnownFace? best = null;
            float bestScore = float.MinValue;

            foreach (var face in _faces)
            {
                var score = FaceRecognizer.DotProduct(face.Embedding, embedding);
                if (score > bestScore)
                {
                    bestScore = score;
                    best = face;
                }
            }

            return bestScore >= threshold ? (best, bestScore) : (null, bestScore);
        }
    }

    public int Count
    {
        get { lock (_lock) { return _faces.Count; } }
    }

    public List<KnownFace> GetAll()
    {
        lock (_lock)
        {
            return [.. _faces];
        }
    }

    private static byte[] FloatsToBytes(float[] floats)
    {
        var bytes = new byte[floats.Length * sizeof(float)];
        Buffer.BlockCopy(floats, 0, bytes, 0, bytes.Length);
        return bytes;
    }

    private static float[] BytesToFloats(byte[] bytes)
    {
        var floats = new float[bytes.Length / sizeof(float)];
        Buffer.BlockCopy(bytes, 0, floats, 0, bytes.Length);
        return floats;
    }

    public void Dispose()
    {
        _connection?.Dispose();
    }
}
