using System.IO;
using System.Text.Json;
using FaceRecognize.Abstractions;

namespace FaceRecognize.Data;

public class JsonConfiguration : IConfiguration
{
    private readonly string _filePath;
    private static readonly JsonSerializerOptions JsonOptions = new()
    {
        WriteIndented = true
    };

    public string LastScanPath { get; set; } = string.Empty;
    public double DistanceThreshold { get; set; } = 0.58;
    public int ThreadCount { get; set; } = Environment.ProcessorCount;
    public string ModelsDirectory { get; set; } = string.Empty;

    public JsonConfiguration(string filePath)
    {
        _filePath = filePath;
        Load();
    }

    private void Load()
    {
        if (!File.Exists(_filePath))
            return;

        try
        {
            var json = File.ReadAllText(_filePath);
            var data = JsonSerializer.Deserialize<ConfigData>(json, JsonOptions);
            if (data != null)
            {
                LastScanPath = data.LastScanPath ?? string.Empty;
                DistanceThreshold = data.DistanceThreshold;
                ThreadCount = data.ThreadCount;
                ModelsDirectory = data.ModelsDirectory ?? string.Empty;
            }
        }
        catch
        {
            // Use defaults on error
        }
    }

    public void Save()
    {
        var dir = Path.GetDirectoryName(_filePath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        var data = new ConfigData
        {
            LastScanPath = LastScanPath,
            DistanceThreshold = DistanceThreshold,
            ThreadCount = ThreadCount,
            ModelsDirectory = ModelsDirectory
        };

        var json = JsonSerializer.Serialize(data, JsonOptions);
        File.WriteAllText(_filePath, json);
    }

    private class ConfigData
    {
        public string? LastScanPath { get; set; }
        public double DistanceThreshold { get; set; } = 0.58;
        public int ThreadCount { get; set; } = Environment.ProcessorCount;
        public string? ModelsDirectory { get; set; }
    }
}
