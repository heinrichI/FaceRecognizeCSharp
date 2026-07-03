using System.Windows;
using FaceRecognize.ViewModels;

namespace FaceRecognize;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();
        Closing += (_, _) =>
        {
            if (DataContext is IDisposable disposable)
                disposable.Dispose();
        };
    }
}
