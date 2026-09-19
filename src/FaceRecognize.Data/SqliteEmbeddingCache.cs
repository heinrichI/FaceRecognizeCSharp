using System.IO;
using System.Text.Json;
using FaceRecognize.Abstractions;
using Microsoft.Data.Sqlite;

namespace FaceRecognize.Data;

public class SqliteEmbeddingCache : IEmbeddingCache
{
    private readonly SqliteConnection _connection;

    public SqliteEmbeddingCache(string dbPath)
    {
        var dir = Path.GetDirectoryName(dbPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        _connection = new SqliteConnection($"Data Source={dbPath}");
        _connection.Open();

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = """
            CREATE TABLE IF NOT EXISTS embedding_cache (
                image_path TEXT NOT NULL,
                file_size INTEGER NOT NULL,
                last_modified TEXT NOT NULL,
                embeddings_json TEXT NOT NULL,
                PRIMARY KEY (image_path)
            )
            """;
        cmd.ExecuteNonQuery();
    }

    public List<CachedFace>? GetCachedFaces(string imagePath, long fileSize, DateTime lastModified)
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "SELECT embeddings_json FROM embedding_cache WHERE image_path = $path AND file_size = $size AND last_modified = $modified";
        cmd.Parameters.AddWithValue("$path", imagePath);
        cmd.Parameters.AddWithValue("$size", fileSize);
        cmd.Parameters.AddWithValue("$modified", lastModified.ToString("O"));

        var json = cmd.ExecuteScalar() as string;
        if (string.IsNullOrEmpty(json))
            return null;

        try
        {
            // Try new format first (List<CachedFace>)
            var faces = JsonSerializer.Deserialize<List<CachedFace>>(json);
            if (faces != null && faces.Count > 0)
                return faces;

            // Fallback: old format (List<float[]>)
            var embeddings = JsonSerializer.Deserialize<List<float[]>>(json);
            if (embeddings != null && embeddings.Count > 0)
                return embeddings.Select(e => new CachedFace { Embedding = e }).ToList();

            return null;
        }
        catch
        {
            return null;
        }
    }

    public void Store(string imagePath, long fileSize, DateTime lastModified, List<CachedFace> faces)
    {
        var json = JsonSerializer.Serialize(faces);

        using var cmd = _connection.CreateCommand();
        cmd.CommandText = """
            INSERT OR REPLACE INTO embedding_cache (image_path, file_size, last_modified, embeddings_json)
            VALUES ($path, $size, $modified, $json)
            """;
        cmd.Parameters.AddWithValue("$path", imagePath);
        cmd.Parameters.AddWithValue("$size", fileSize);
        cmd.Parameters.AddWithValue("$modified", lastModified.ToString("O"));
        cmd.Parameters.AddWithValue("$json", json);
        cmd.ExecuteNonQuery();
    }

    public void Clear()
    {
        using var cmd = _connection.CreateCommand();
        cmd.CommandText = "DELETE FROM embedding_cache";
        cmd.ExecuteNonQuery();
    }

    public void Dispose()
    {
        _connection?.Dispose();
    }
}
