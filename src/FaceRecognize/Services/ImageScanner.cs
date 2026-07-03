using System.IO;

namespace FaceRecognize.Services;

public static class ImageScanner
{
    private static readonly string[] Extensions =
        [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"];

    public static List<string> ScanDirectory(string directory)
    {
        if (!Directory.Exists(directory))
            return [];

        var images = new List<string>();
        foreach (var ext in Extensions)
        {
            images.AddRange(Directory.GetFiles(directory, $"*{ext}", SearchOption.AllDirectories));
        }
        return images;
    }
}
