using System.Windows;
using FaceRecognize.ViewModels;

namespace FaceRecognize;

public partial class MainWindow : Window
{
    public MainWindow()
    {
        InitializeComponent();

        // БД может измениться извне во время работы (другой процесс/вручную) —
        // при активации окна перечитываем её с диска.
        Activated += (_, _) =>
        {
            if (DataContext is MainViewModel vm)
                vm.RefreshFromDisk();
        };

        Closing += (_, _) =>
        {
            if (DataContext is IDisposable disposable)
                disposable.Dispose();
        };
    }
}
