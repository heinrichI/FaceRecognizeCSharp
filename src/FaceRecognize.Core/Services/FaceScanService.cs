using System.Diagnostics;
using System.IO;
using FaceRecognize.Abstractions;

namespace FaceRecognize.Core.Services;

public class FaceScanService : IFaceScanService
{
    private readonly IImageScanner _imageScanner;
    private readonly IEmbeddingCache _embeddingCache;
    private readonly IFaceRecognizer _recognizer;
    private readonly IVectorStore _vectorStore;
    private readonly IClusteringService _clusteringService;

    public FaceScanService(
        IImageScanner imageScanner,
        IEmbeddingCache embeddingCache,
        IFaceRecognizer recognizer,
        IVectorStore vectorStore,
        IClusteringService clusteringService)
    {
        _imageScanner = imageScanner;
        _embeddingCache = embeddingCache;
        _recognizer = recognizer;
        _vectorStore = vectorStore;
        _clusteringService = clusteringService;
    }

    public ScanResult ScanDirectory(
        string directory,
        float threshold,
        int maxDegreeOfParallelism,
        IProgress<ScanProgress>? progress,
        CancellationToken cancellationToken = default)
    {
        Debug.Assert(maxDegreeOfParallelism >= 1, "maxDegreeOfParallelism must be >= 1");
        Debug.Assert(!string.IsNullOrWhiteSpace(directory), "Scan directory must not be empty");

        var images = _imageScanner.ScanDirectory(directory);
        Debug.Assert(!images.Any(string.IsNullOrWhiteSpace), "Scanner must not return invalid paths");

        var results = new List<RecognizedFace>();
        var unknownEmbeddings = new List<(string path, float[] embedding, byte[]? thumbnail)>();
        var lockObj = new object();
        int processed = 0;
        int faceCount = 0;
        bool cancelled = false;

        var parallelOptions = new ParallelOptions
        {
            MaxDegreeOfParallelism = maxDegreeOfParallelism,
            CancellationToken = cancellationToken
        };

        try
        {
            Parallel.ForEach(images, parallelOptions, imgPath =>
            {
                cancellationToken.ThrowIfCancellationRequested();

                var current = Interlocked.Increment(ref processed);
                progress?.Report(new ScanProgress(current, images.Count, Path.GetFileName(imgPath)));

                try
                {
                    var fileInfo = new FileInfo(imgPath);
                    var fileSize = fileInfo.Length;
                    var lastModified = fileInfo.LastWriteTimeUtc;

                    var cachedFaces = _embeddingCache.GetCachedFaces(imgPath, fileSize, lastModified);
                    CachedFace[] faceArray;

                    if (cachedFaces is { Count: > 0 })
                    {
                        faceArray = cachedFaces.ToArray();
                    }
                    else
                    {
                        var detected = _recognizer.ExtractAllFaces(imgPath);
                        faceArray = detected.Select(f => new CachedFace
                        {
                            Embedding = f.Embedding,
                            X = f.Box.X,
                            Y = f.Box.Y,
                            Width = f.Box.Width,
                            Height = f.Box.Height
                        }).ToArray();
                        if (faceArray.Length > 0)
                            _embeddingCache.Store(imgPath, fileSize, lastModified, faceArray.ToList());
                    }

                    foreach (var cachedFace in faceArray)
                    {
                        var (best, score) = _vectorStore.Search(cachedFace.Embedding, threshold);
                        var box = new FaceBox(cachedFace.X, cachedFace.Y, cachedFace.Width, cachedFace.Height);
                        var thumb = FaceThumbnail.Create(imgPath, box, 128);

                        lock (lockObj)
                        {
                            if (best != null)
                            {
                                results.Add(new RecognizedFace
                                {
                                    ImagePath = imgPath,
                                    Name = best.Name,
                                    Confidence = score,
                                    Thumbnail = thumb
                                });
                            }
                            else
                            {
                                unknownEmbeddings.Add((imgPath, cachedFace.Embedding, thumb));
                            }

                            faceCount++;
                        }
                    }
                }
                catch
                {
                    // Skip unprocessable images
                }
            });
        }
        catch (OperationCanceledException)
        {
            cancelled = true;
        }

        Debug.Assert(faceCount == results.Count + unknownEmbeddings.Count, "Face count conservation violated during scan");
        Debug.Assert(results.All(r => r.Confidence >= threshold), "Known face result below match threshold");
        if (!cancelled)
            Debug.Assert(processed == images.Count, "Not all images were processed");

        var clusters = _clusteringService.ClusterUnknownFaces(unknownEmbeddings);
        Debug.Assert(clusters.Sum(c => c.Faces.Count) == unknownEmbeddings.Count,
            "Clustering lost or duplicated faces");

        return new ScanResult
        {
            KnownFaces = results,
            UnknownClusters = clusters,
            TotalImagesScanned = processed,
            TotalFacesFound = faceCount,
            Cancelled = cancelled
        };
    }
}