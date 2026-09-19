using FaceRecognize.Abstractions;
using Microsoft.Extensions.DependencyInjection;

namespace FaceRecognize.Data.Extensions;

public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Регистрирует data-слой. <see cref="IConfiguration"/> регистрирует composition root
    /// (единственный экземпляр) — здесь конфигурация не дублируется.
    /// </summary>
    public static void AddData(this IServiceCollection services, string dbPath)
    {
        var cachePath = System.IO.Path.Combine(
            System.IO.Path.GetDirectoryName(dbPath)!,
            "embedding_cache.db");

        services.AddSingleton<IVectorStore>(sp => new SqliteVectorStore(dbPath));
        services.AddSingleton<IEmbeddingCache>(sp => new SqliteEmbeddingCache(cachePath));
        services.AddSingleton<IDatabaseMaintenanceService>(sp =>
            new DatabaseMaintenanceService(sp.GetRequiredService<IVectorStore>()));
    }
}