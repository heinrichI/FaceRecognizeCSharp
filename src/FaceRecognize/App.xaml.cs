using System.IO;
using System.Windows;
using FaceRecognize.Abstractions;
using FaceRecognize.Core.Extensions;
using FaceRecognize.Data;
using FaceRecognize.Data.Extensions;
using FaceRecognize.ViewModels;
using Microsoft.Extensions.DependencyInjection;

namespace FaceRecognize;

public partial class App : Application
{
    public static IServiceProvider ServiceProvider { get; private set; } = null!;

    protected override void OnStartup(StartupEventArgs e)
    {
        base.OnStartup(e);

        var basePath = AppDomain.CurrentDomain.BaseDirectory;
        var dbPath = Path.Combine(basePath, "faces.db");
        var configPath = Path.Combine(basePath, "config.json");

        // Load config first to get ModelsDirectory
        var config = new JsonConfiguration(configPath);

        var modelPath = !string.IsNullOrEmpty(config.ModelsDirectory)
            ? Path.Combine(config.ModelsDirectory, "lm_model3_opt.onnx")
            : Path.Combine(basePath, "Models", "lm_model3_opt.onnx");

        var services = new ServiceCollection();
        services.AddFaceRecognition(File.Exists(modelPath) ? modelPath : null);
        services.AddData(dbPath, configPath);
        services.AddSingleton<MainViewModel>();

        ServiceProvider = services.BuildServiceProvider();

        var mainWindow = new MainWindow
        {
            DataContext = ServiceProvider.GetRequiredService<MainViewModel>()
        };
        mainWindow.Show();
    }
}
