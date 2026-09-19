using System.Windows;
using FaceRecognize.Abstractions;
using FaceRecognize.ViewModels;

namespace FaceRecognize.Views;

public partial class ResultsWindow : Window
{
    public ResultsWindow(ScanResult result, IVectorStore vectorStore, IFaceRecognizer recognizer, IClusteringService clusteringService)
    {
        InitializeComponent();
        DataContext = new ResultsViewModel(result, vectorStore, recognizer, clusteringService);
    }
}
