using FaceRecognize.Abstractions;
using FaceRecognize.Core.Services;
using Xunit;

namespace FaceRecognize.Core.Tests.Services;

public class ImageScannerTests
{
    private readonly IImageScanner _scanner = new ImageScanner();

    [Fact]
    public void ScanDirectory_FindsImages()
    {
        if (!TestConfig.IsAvailable) return;

        var result = _scanner.ScanDirectory(TestConfig.TrainDataRoot);
        Assert.True(result.Count > 0, "No images found in train directory");
    }

    [Fact]
    public void ScanDirectory_SkipsNonImages()
    {
        var tempDir = Path.Combine(Path.GetTempPath(), $"test_scan_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            File.WriteAllText(Path.Combine(tempDir, "test.txt"), "hello");
            File.WriteAllText(Path.Combine(tempDir, "test.jpg"), "fake jpg");
            File.WriteAllText(Path.Combine(tempDir, "test.png"), "fake png");

            var result = _scanner.ScanDirectory(tempDir);

            Assert.Equal(2, result.Count);
            Assert.All(result, f => Assert.True(f.EndsWith(".jpg") || f.EndsWith(".png")));
        }
        finally
        {
            Directory.Delete(tempDir, true);
        }
    }

    [Fact]
    public void ScanDirectory_NonExistent_ReturnsEmpty()
    {
        var result = _scanner.ScanDirectory(@"C:\nonexistent_path_12345");
        Assert.Empty(result);
    }
}
