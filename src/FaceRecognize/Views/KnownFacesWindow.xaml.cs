using System.Windows;
using FaceRecognize.Abstractions;
using FaceRecognize.ViewModels;

namespace FaceRecognize.Views;

public partial class KnownFacesWindow : Window
{
    public KnownFacesWindow(IVectorStore vectorStore)
    {
        InitializeComponent();
        DataContext = new KnownFacesViewModel(vectorStore);
    }
}
