using System.Diagnostics;
using System.IO;
using System.Net.Http;
using FaceRecognize.Abstractions;

namespace FaceRecognize.Core.Services;

public class ModelDownloader : IModelDownloader
{
    private static readonly HttpClient Http = new() { Timeout = TimeSpan.FromMinutes(10) };

    public bool IsModelPresent(string modelPath)
    {
        return File.Exists(modelPath) && new FileInfo(modelPath).Length > 1024;
    }

    public async Task<bool> EnsureModelExistsAsync(
        string modelPath,
        string downloadUrl,
        IProgress<int>? progress = null,
        CancellationToken ct = default)
    {
        Debug.Assert(!string.IsNullOrEmpty(modelPath), "modelPath must not be empty");

        if (IsModelPresent(modelPath))
            return true;

        var dir = Path.GetDirectoryName(modelPath);
        if (!string.IsNullOrEmpty(dir))
            Directory.CreateDirectory(dir);

        var tempPath = modelPath + ".downloading";

        try
        {
            using var response = await Http.GetAsync(downloadUrl, HttpCompletionOption.ResponseHeadersRead, ct);
            response.EnsureSuccessStatusCode();

            var totalBytes = response.Content.Headers.ContentLength ?? -1L;
            await using var contentStream = await response.Content.ReadAsStreamAsync(ct);
            await using var fileStream = new FileStream(tempPath, FileMode.Create, FileAccess.Write, FileShare.None, 81920, true);

            var buffer = new byte[81920];
            long downloaded = 0;
            int bytesRead;

            while ((bytesRead = await contentStream.ReadAsync(buffer, ct)) > 0)
            {
                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), ct);
                downloaded += bytesRead;

                if (totalBytes > 0)
                {
                    var pct = Math.Clamp((int)(downloaded * 100 / totalBytes), 0, 100);
                    Debug.Assert(pct is >= 0 and <= 100, "Download progress out of range");
                    progress?.Report(pct);
                }
            }

            await fileStream.FlushAsync(ct);
            fileStream.Close();

            // Atomic rename
            if (File.Exists(modelPath))
                File.Delete(modelPath);
            File.Move(tempPath, modelPath);

            Debug.Assert(IsModelPresent(modelPath), "Model file missing or too small after successful download");
            return true;
        }
        catch
        {
            try { File.Delete(tempPath); } catch { }
            return false;
        }
    }
}
