using System.Windows;
using FaceRecognize.Abstractions;
using FaceRecognize.ViewModels;

namespace FaceRecognize.Views;

public partial class ResultsWindow : Window
{
    public ResultsWindow(ScanResult result)
    {
        InitializeComponent();
        DataContext = new ResultsViewModel(result);
    }
}
