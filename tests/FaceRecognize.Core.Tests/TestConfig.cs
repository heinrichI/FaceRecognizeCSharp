namespace FaceRecognize.Core.Tests;

public static class TestConfig
{
    public static string TrainDataRoot => @"F:\FaceTrain";

    public static string GetRandomPersonFolder()
    {
        var dirs = Directory.GetDirectories(TrainDataRoot);
        return dirs[Random.Shared.Next(dirs.Length)];
    }

    public static string GetRandomImage(string? personFolder = null)
    {
        personFolder ??= GetRandomPersonFolder();
        var files = Directory.GetFiles(personFolder, "*.*", SearchOption.AllDirectories)
            .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            .ToArray();
        return files[Random.Shared.Next(files.Length)];
    }

    public static string GetPersonName(string folderPath)
    {
        return Path.GetFileName(folderPath);
    }

    public static string[] GetImages(string personFolder, int count)
    {
        var files = Directory.GetFiles(personFolder, "*.*", SearchOption.AllDirectories)
            .Where(f => f.EndsWith(".jpg", StringComparison.OrdinalIgnoreCase) ||
                        f.EndsWith(".png", StringComparison.OrdinalIgnoreCase))
            .ToArray();
        return files.Take(count).ToArray();
    }

    public static bool IsAvailable => Directory.Exists(TrainDataRoot);
}
