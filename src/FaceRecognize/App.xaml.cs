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

        // Единственный экземпляр конфигурации: контейнер — единый источник правды.
        var configuration = new JsonConfiguration(configPath);
        var modelPath = ModelPathResolver.Resolve(configuration.ModelsDirectory, basePath);

        var services = new ServiceCollection();
        services.AddSingleton<IConfiguration>(configuration);
        services.AddFaceRecognition(File.Exists(modelPath) ? modelPath : null);
        services.AddData(dbPath);
        services.AddSingleton<MainViewModel>();

        ServiceProvider = services.BuildServiceProvider();

        var mainWindow = new MainWindow
        {
            DataContext = ServiceProvider.GetRequiredService<MainViewModel>()
        };
        mainWindow.Show();
    }

    protected override void OnExit(ExitEventArgs e)
    {
        // Контейнер владеет жизненным циклом синглтонов (store, cache, recognizer)
        // и закрывает их SQLite-соединения при выходе.
        (ServiceProvider as IDisposable)?.Dispose();
        base.OnExit(e);
    }
}