using System.IO;
using FaceRecognize.Abstractions;
using Microsoft.Data.Sqlite;

namespace FaceRecognize.Data;

public class SqliteVectorStore : IVectorStore
{
    private readonly string _dbPath;
    private readonly List<KnownFace> _faces = [];
    private readonly object _lock = new();
    private SqliteConnection? _connection;

    public SqliteVectorStore(string dbPath)
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
                embedding BLOB NOT NULL,
                thumbnail BLOB
            )
            """;
        cmd.ExecuteNonQuery();

        // Migration: add thumbnail column if missing
        try
        {
            using var check = _connection.CreateCommand();
            check.CommandText = "SELECT thumbnail FROM known_faces LIMIT 1";
            check.ExecuteScalar();
        }
        catch
        {
            using var alter = _connection.CreateCommand();
            alter.CommandText = "ALTER TABLE known_faces ADD COLUMN thumbnail BLOB";
            alter.ExecuteNonQuery();
        }

        LoadAll();
    }

    private void LoadAll()
    {
        using var cmd = _connection!.CreateCommand();
        cmd.CommandText = "SELECT id, name, image_path, created_at, embedding, thumbnail FROM known_faces";

        using var reader = cmd.ExecuteReader();
        while (reader.Read())
        {
            var face = new KnownFace
            {
                Id = Guid.Parse(reader.GetString(0)),
                Name = reader.GetString(1),
                ImagePath = reader.GetString(2),
                CreatedAt = DateTime.Parse(reader.GetString(3)),
                Embedding = BytesToFloats(reader.GetFieldValue<byte[]>(4)),
                Thumbnail = reader.IsDBNull(5) ? null : reader.GetFieldValue<byte[]>(5)
            };
            _faces.Add(face);
        }
    }

    public void Add(KnownFace face)
    {
        using var cmd = _connection!.CreateCommand();
        cmd.CommandText = """
            INSERT INTO known_faces (id, name, image_path, created_at, embedding, thumbnail)
            VALUES ($id, $name, $image_path, $created_at, $embedding, $thumbnail)
            """;
        cmd.Parameters.AddWithValue("$id", face.Id.ToString());
        cmd.Parameters.AddWithValue("$name", face.Name);
        cmd.Parameters.AddWithValue("$image_path", face.ImagePath);
        cmd.Parameters.AddWithValue("$created_at", face.CreatedAt.ToString("O"));
        cmd.Parameters.AddWithValue("$embedding", FloatsToBytes(face.Embedding));
        cmd.Parameters.AddWithValue("$thumbnail", (object?)face.Thumbnail ?? DBNull.Value);
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
                var score = DotProduct(face.Embedding, embedding);
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

    private static float DotProduct(float[] a, float[] b)
    {
        float sum = 0;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
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
