using System.IO;
using System.Windows;
using FaceRecognize.Core.Extensions;
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

        var services = new ServiceCollection();

        var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "lm_model3_opt.onnx");
        var dbPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "faces.db");

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
}
