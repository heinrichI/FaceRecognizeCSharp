using FaceRecognize.Abstractions;
using Microsoft.Extensions.DependencyInjection;

namespace FaceRecognize.Data.Extensions;

public static class ServiceCollectionExtensions
{
    public static void AddData(this IServiceCollection services, string dbPath)
    {
        services.AddSingleton<IVectorStore>(sp => new SqliteVectorStore(dbPath));
    }
}
