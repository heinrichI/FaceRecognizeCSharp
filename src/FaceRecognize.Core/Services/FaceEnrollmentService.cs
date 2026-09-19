using System.Diagnostics;
using System.IO;
using FaceRecognize.Abstractions;

namespace FaceRecognize.Core.Services;

public class FaceEnrollmentService : IFaceEnrollmentService
{
    private readonly IVectorStore _vectorStore;
    private readonly IFaceRecognizer _recognizer;
    private readonly IClusteringService _clusteringService;

    public FaceEnrollmentService(
        IVectorStore vectorStore,
        IFaceRecognizer recognizer,
        IClusteringService clusteringService)
    {
        _vectorStore = vectorStore;
        _recognizer = recognizer;
        _clusteringService = clusteringService;
    }

    public List<string> GetExistingNames(IReadOnlyList<string>? extraNames = null)
    {
        var names = _vectorStore.GetAll().Select(f => f.Name);
        if (extraNames is { Count: > 0 })
            names = names.Concat(extraNames);

        return names.Distinct().Order().ToList();
    }

    public int EnrollByName(string name, IReadOnlyList<string> imagePaths, IProgress<int>? progress = null)
    {
        Debug.Assert(!string.IsNullOrWhiteSpace(name), "Person name must not be empty");

        int added = 0;
        for (int i = 0; i < imagePaths.Count; i++)
        {
            var path = imagePaths[i];
            try
            {
                var faces = _recognizer.ExtractAllFaces(path);
                foreach (var face in faces)
                {
                    Debug.Assert(face.Embedding is { Length: > 0 }, "Face embedding must not be empty");
                    var thumb = FaceThumbnail.Create(path, face.Box, 128);
                    _vectorStore.Add(new KnownFace
                    {
                        Name = name,
                        ImagePath = path,
                        Embedding = face.Embedding,
                        Thumbnail = thumb
                    });
                    added++;
                }
            }
            catch
            {
                // Skip unprocessable files
            }

            progress?.Report(i + 1);
        }

        return added;
    }

    public AddToKnownResult AddToKnown(AddToKnownRequest request)
    {
        Debug.Assert(!string.IsNullOrWhiteSpace(request.Name), "Person name must not be empty");
        Debug.Assert(File.Exists(request.ImagePath), $"Image not found: {request.ImagePath}");

        var detectedFaces = _recognizer.ExtractAllFaces(request.ImagePath);
        Debug.Assert(detectedFaces.Count > 0, $"No faces found in {request.ImagePath}");
        var detected = detectedFaces.First();

        var thumb = FaceThumbnail.Create(request.ImagePath, detected.Box, 128);
        var countBefore = _vectorStore.Count;
        _vectorStore.Add(new KnownFace
        {
            Name = request.Name,
            ImagePath = request.ImagePath,
            Embedding = detected.Embedding,
            Thumbnail = thumb
        });
        Debug.Assert(_vectorStore.Count == countBefore + 1, "Vector store count did not increase by 1 after Add");

        // Re-check: which of the remaining unknown faces now match the updated store?
        var newlyMatched = new List<RecognizedFace>();
        var stillUnknown = new List<(string path, float[] embedding, byte[]? thumbnail)>();
        int skipped = 0;

        foreach (var unknown in request.CurrentUnknownFaces)
        {
            if (unknown.ImagePath == request.ImagePath || unknown.Embedding is not { Length: > 0 })
            {
                skipped++;
                continue;
            }

            var (best, score) = _vectorStore.Search(unknown.Embedding, request.Threshold);
            if (best != null)
            {
                newlyMatched.Add(new RecognizedFace
                {
                    ImagePath = unknown.ImagePath,
                    Name = best.Name,
                    Confidence = score,
                    Thumbnail = unknown.Thumbnail
                });
            }
            else
            {
                stillUnknown.Add((unknown.ImagePath, unknown.Embedding, unknown.Thumbnail));
            }
        }

        Debug.Assert(newlyMatched.Count + stillUnknown.Count + skipped == request.CurrentUnknownFaces.Count,
            "Every current unknown face must be classified exactly once");

        var clusters = _clusteringService.ClusterUnknownFaces(stillUnknown);
        Debug.Assert(clusters.Sum(c => c.Faces.Count) == stillUnknown.Count,
            "Re-clustering lost or duplicated faces");

        return new AddToKnownResult
        {
            NewlyMatched = newlyMatched,
            RemainingClusters = clusters
        };
    }
}