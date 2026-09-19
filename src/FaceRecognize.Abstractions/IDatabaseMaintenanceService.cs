namespace FaceRecognize.Abstractions;

public interface IDatabaseMaintenanceService
{
    int RemoveOrphanedEntries();
    List<string> GetOrphanedEntries();
}
