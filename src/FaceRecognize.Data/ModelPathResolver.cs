using System.IO;

namespace FaceRecognize.Data;

/// <summary>Единственное место разрешения пути 3D-модели (lm_model3_opt.onnx).</summary>
public static class ModelPathResolver
{
    public const string ModelFileName = "lm_model3_opt.onnx";

    /// <param name="modelsDirectory">Конфигурируемая директория из конфигурации (пусто — не задана).</param>
    /// <param name="basePath">Базовая директория приложения (fallback: <c>basePath/Models</c>).</param>
    public static string Resolve(string? modelsDirectory, string basePath)
        => !string.IsNullOrWhiteSpace(modelsDirectory)
            ? Path.Combine(modelsDirectory, ModelFileName)
            : Path.Combine(basePath, "Models", ModelFileName);
}