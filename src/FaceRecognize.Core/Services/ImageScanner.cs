using System.Diagnostics;
using System.IO;
using FaceRecognize.Abstractions;

namespace FaceRecognize.Core.Services;

public class ImageScanner : IImageScanner
{
    private static readonly string[] Extensions =
        [".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tiff", ".tif"];

    public List<string> ScanDirectory(string directory)
    {
        if (!Directory.Exists(directory))
            return [];

        var images = new List<string>();
        foreach (var ext in Extensions)
        {
            images.AddRange(Directory.GetFiles(directory, $"*{ext}", SearchOption.AllDirectories));
        }

        Debug.Assert(images.Count == images.Distinct().Count(), "Duplicate image paths in scan results");
        return images;
    }
}
