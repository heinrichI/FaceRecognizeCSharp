using FaceRecognize.Abstractions;
using Microsoft.Extensions.DependencyInjection;

namespace FaceRecognize.Data.Extensions;

public static class ServiceCollectionExtensions
{
    public static void AddData(this IServiceCollection services, string dbPath, string configPath)
    {
        var cachePath = System.IO.Path.Combine(
            System.IO.Path.GetDirectoryName(dbPath)!,
            "embedding_cache.db");

        services.AddSingleton<IVectorStore>(sp => new SqliteVectorStore(dbPath));
        services.AddSingleton<IConfiguration>(sp => new JsonConfiguration(configPath));
        services.AddSingleton<IEmbeddingCache>(sp => new SqliteEmbeddingCache(cachePath));
        services.AddSingleton<IDatabaseMaintenanceService>(sp =>
            new DatabaseMaintenanceService(sp.GetRequiredService<IVectorStore>()));
    }
}
