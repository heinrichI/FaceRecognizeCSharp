using FaceRecognize.Abstractions;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;

namespace FaceRecognize.Core.Services;

/// <summary>JPEG-обложка региона лица с 30% контекстной границей.</summary>
public static class FaceThumbnail
{
    public static byte[]? Create(string imagePath, FaceBox box, int outputSize)
    {
        try
        {
            using var original = Image.Load(imagePath);

            // Add 30% border around face for context
            var borderX = (int)(box.Width * 0.3f);
            var borderY = (int)(box.Height * 0.3f);

            int cropX = Math.Max(0, (int)box.X - borderX);
            int cropY = Math.Max(0, (int)box.Y - borderY);
            int cropW = Math.Min(original.Width - cropX, (int)box.Width + borderX * 2);
            int cropH = Math.Min(original.Height - cropY, (int)box.Height + borderY * 2);

            original.Mutate(ctx =>
            {
                ctx.Crop(new SixLabors.ImageSharp.Rectangle(cropX, cropY, Math.Max(cropW, 1), Math.Max(cropH, 1)));
                ctx.Resize(new ResizeOptions
                {
                    Size = new SixLabors.ImageSharp.Size(outputSize, outputSize),
                    Mode = ResizeMode.Crop
                });
            });

            using var ms = new MemoryStream();
            original.Save(ms, new JpegEncoder());
            return ms.ToArray();
        }
        catch
        {
            return null;
        }
    }
}