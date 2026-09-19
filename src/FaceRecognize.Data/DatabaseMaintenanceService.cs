using System.Diagnostics;
using System.IO;
using FaceRecognize.Abstractions;

namespace FaceRecognize.Data;

public class DatabaseMaintenanceService : IDatabaseMaintenanceService
{
    private readonly IVectorStore _vectorStore;

    public DatabaseMaintenanceService(IVectorStore vectorStore)
    {
        _vectorStore = vectorStore;
    }

    public List<string> GetOrphanedEntries()
    {
        var allFaces = _vectorStore.GetAll();
        return allFaces
            .Where(f => !File.Exists(f.ImagePath))
            .Select(f => f.ImagePath)
            .ToList();
    }

    public int RemoveOrphanedEntries()
    {
        var allFaces = _vectorStore.GetAll();
        int removed = 0;

        foreach (var face in allFaces)
        {
            if (!File.Exists(face.ImagePath))
            {
                _vectorStore.Delete(face.Id);
                removed++;
            }
        }

        Debug.Assert(_vectorStore.GetAll().All(f => File.Exists(f.ImagePath)),
            "Orphaned entries still present after cleanup");

        return removed;
    }
}
