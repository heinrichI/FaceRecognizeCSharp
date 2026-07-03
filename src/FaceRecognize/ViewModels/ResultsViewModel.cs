using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using FaceRecognize.Models;

namespace FaceRecognize.ViewModels;

public partial class ResultsViewModel : ObservableObject
{
    [ObservableProperty] private int _totalImagesScanned;
    [ObservableProperty] private int _totalFacesFound;

    public ObservableCollection<RecognizedFaceViewModel> KnownFaces { get; } = [];
    public ObservableCollection<PersonClusterViewModel> UnknownClusters { get; } = [];

    public ResultsViewModel(ScanResult result)
    {
        TotalImagesScanned = result.TotalImagesScanned;
        TotalFacesFound = result.TotalFacesFound;

        foreach (var face in result.KnownFaces)
        {
            KnownFaces.Add(new RecognizedFaceViewModel
            {
                ImagePath = face.ImagePath,
                Name = face.Name,
                Confidence = face.Confidence
            });
        }

        foreach (var cluster in result.UnknownClusters)
        {
            UnknownClusters.Add(new PersonClusterViewModel
            {
                ClusterId = cluster.ClusterId,
                FaceCount = cluster.Faces.Count,
                Faces = cluster.Faces.Select(f => new ClusterFaceViewModel
                {
                    ImagePath = f.ImagePath
                }).ToList()
            });
        }
    }
}

public class RecognizedFaceViewModel
{
    public string ImagePath { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public float Confidence { get; set; }
}

public class PersonClusterViewModel
{
    public int ClusterId { get; set; }
    public int FaceCount { get; set; }
    public List<ClusterFaceViewModel> Faces { get; set; } = [];
}

public class ClusterFaceViewModel
{
    public string ImagePath { get; set; } = string.Empty;
}
