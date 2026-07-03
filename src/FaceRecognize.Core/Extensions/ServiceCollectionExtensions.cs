using FaceRecognize.Abstractions;
using FaceRecognize.Core.Services;
using Microsoft.Extensions.DependencyInjection;

namespace FaceRecognize.Core.Extensions;

public static class ServiceCollectionExtensions
{
    public static void AddFaceRecognition(this IServiceCollection services, string? landmarkOnnxPath = null)
    {
        services.AddSingleton<IFaceRecognizer>(sp => new FaceRecognizer(landmarkOnnxPath));
        services.AddSingleton<IClusteringService, ClusteringService>();
        services.AddSingleton<IImageScanner, ImageScanner>();
    }
}
