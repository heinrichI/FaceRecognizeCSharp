using System.Windows;
using FaceRecognize.Abstractions;
using FaceRecognize.ViewModels;

namespace FaceRecognize.Views;

public partial class ResultsWindow : Window
{
    public ResultsWindow(ScanResult result, IFaceEnrollmentService enrollmentService, float distanceThreshold)
    {
        InitializeComponent();
        DataContext = new ResultsViewModel(result, enrollmentService, distanceThreshold);
    }
}