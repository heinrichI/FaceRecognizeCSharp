namespace FaceRecognize.Abstractions;

public interface IImageScanner
{
    List<string> ScanDirectory(string directory);
}
