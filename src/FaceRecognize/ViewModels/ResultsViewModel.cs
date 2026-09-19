using System.Collections.ObjectModel;
using System.Diagnostics;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using FaceRecognize.Abstractions;

namespace FaceRecognize.ViewModels;

public partial class ResultsViewModel : ObservableObject
{
    private readonly IFaceEnrollmentService _enrollmentService;
    private readonly float _distanceThreshold;
    private readonly List<RecognizedFaceViewModel> _allKnownFaces = [];

    [ObservableProperty] private int _totalImagesScanned;
    [ObservableProperty] private int _totalFacesFound;
    [ObservableProperty] private DisplayMode _selectedDisplayMode = DisplayMode.All;
    [ObservableProperty] private bool _isKnownVisible = true;
    [ObservableProperty] private bool _isUnknownVisible = true;
    [ObservableProperty] private bool _isExclusionPanelVisible;

    public DisplayMode[] AvailableDisplayModes { get; } = [DisplayMode.All, DisplayMode.KnownOnly, DisplayMode.UnknownOnly];

    public ObservableCollection<KnownPersonClusterViewModel> KnownFaces { get; } = [];
    public ObservableCollection<PersonClusterViewModel> UnknownClusters { get; } = [];
    public ObservableCollection<SelectableNameViewModel> KnownPeopleNames { get; } = [];

    partial void OnSelectedDisplayModeChanged(DisplayMode value)
    {
        IsKnownVisible = value is DisplayMode.All or DisplayMode.KnownOnly;
        IsUnknownVisible = value is DisplayMode.All or DisplayMode.UnknownOnly;
        IsExclusionPanelVisible = value == DisplayMode.UnknownOnly;
        RefreshKnownFaces();
    }

    public ResultsViewModel(ScanResult result, IFaceEnrollmentService enrollmentService, float distanceThreshold)
    {
        _enrollmentService = enrollmentService;
        _distanceThreshold = distanceThreshold;

        TotalImagesScanned = result.TotalImagesScanned;
        TotalFacesFound = result.TotalFacesFound;

        foreach (var face in result.KnownFaces)
        {
            _allKnownFaces.Add(new RecognizedFaceViewModel
            {
                ImagePath = face.ImagePath,
                Name = face.Name,
                Confidence = face.Confidence,
                Thumbnail = face.Thumbnail
            });
        }

        RefreshKnownPeopleNames();
        RefreshKnownFaces();

        foreach (var cluster in result.UnknownClusters)
        {
            UnknownClusters.Add(new PersonClusterViewModel
            {
                ClusterId = cluster.ClusterId,
                FaceCount = cluster.Faces.Count,
                Faces = new ObservableCollection<ClusterFaceViewModel>(cluster.Faces.Select(f => new ClusterFaceViewModel
                {
                    ImagePath = f.ImagePath,
                    Embedding = f.Embedding,
                    Thumbnail = f.Thumbnail
                }))
            });
        }
    }

    private void RefreshKnownPeopleNames()
    {
        var existingNames = _allKnownFaces.Select(f => f.Name).Distinct().ToList();

        // Remove names no longer present
        for (int i = KnownPeopleNames.Count - 1; i >= 0; i--)
        {
            if (!existingNames.Contains(KnownPeopleNames[i].Name))
                KnownPeopleNames.RemoveAt(i);
        }

        // Add new names
        foreach (var name in existingNames)
        {
            if (!KnownPeopleNames.Any(n => n.Name == name))
            {
                var vm = new SelectableNameViewModel(name);
                vm.PropertyChanged += (_, _) => RefreshKnownFaces();
                KnownPeopleNames.Add(vm);
            }
        }
    }

    private void RefreshKnownFaces()
    {
        var excluded = new HashSet<string>(
            KnownPeopleNames.Where(n => n.IsExcluded).Select(n => n.Name));

        KnownFaces.Clear();
        var filtered = _allKnownFaces.Where(f => !excluded.Contains(f.Name));

        var grouped = filtered.GroupBy(f => f.Name).OrderBy(g => g.Key);
        int clusterId = 0;
        foreach (var group in grouped)
        {
            KnownFaces.Add(new KnownPersonClusterViewModel
            {
                ClusterId = clusterId++,
                PersonName = group.Key,
                Faces = new ObservableCollection<RecognizedFaceViewModel>(group)
            });
        }
    }

    private IEnumerable<string> GetExistingNames()
        => _enrollmentService.GetExistingNames(_allKnownFaces.Select(f => f.Name).ToList());

    [RelayCommand]
    private void AddToKnown(string? imagePath)
    {
        if (string.IsNullOrEmpty(imagePath)) return;

        var existingNames = GetExistingNames();

        var name = Views.InputDialog.ShowDialog(
            "Add to Known Faces",
            "Enter or select name for this face:",
            existingNames,
            _enrollmentService.LastEnrolledName);

        if (string.IsNullOrWhiteSpace(name)) return;

        try
        {
            // Состояние «неизвестные лица» передаётся в use-case как DTO-снапшот;
            // сам use-case ничего не хранит между вызовами.
            var currentUnknownFaces = UnknownClusters
                .SelectMany(c => c.Faces)
                .Select(f => new ClusterFace
                {
                    ImagePath = f.ImagePath,
                    Embedding = f.Embedding ?? [],
                    Thumbnail = f.Thumbnail
                })
                .ToList();

            var response = _enrollmentService.AddToKnown(
                new AddToKnownRequest(imagePath, name, _distanceThreshold, currentUnknownFaces));

            // Кликнутое лицо становится известным
            _allKnownFaces.Add(new RecognizedFaceViewModel
            {
                ImagePath = imagePath,
                Name = name,
                Confidence = 1.0f
            });

            // Новые совпадения среди оставшихся неизвестных
            foreach (var matched in response.NewlyMatched)
            {
                _allKnownFaces.Add(new RecognizedFaceViewModel
                {
                    ImagePath = matched.ImagePath,
                    Name = matched.Name,
                    Confidence = matched.Confidence,
                    Thumbnail = matched.Thumbnail
                });
            }

            // Пере-кластеризованные оставшиеся неизвестные
            UnknownClusters.Clear();
            foreach (var cluster in response.RemainingClusters)
            {
                UnknownClusters.Add(new PersonClusterViewModel
                {
                    ClusterId = cluster.ClusterId,
                    FaceCount = cluster.Faces.Count,
                    Faces = new ObservableCollection<ClusterFaceViewModel>(cluster.Faces.Select(f => new ClusterFaceViewModel
                    {
                        ImagePath = f.ImagePath,
                        Embedding = f.Embedding,
                        Thumbnail = f.Thumbnail
                    }))
                });
            }

            RefreshKnownPeopleNames();
            RefreshKnownFaces();
            Debug.Assert(KnownPeopleNames.Any(n => n.Name == name), "Added name missing from people list");
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Error: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }

    [RelayCommand]
    private void ShowInfo(string? imagePath)
    {
        if (string.IsNullOrEmpty(imagePath)) return;

        var info = $"Image: {imagePath}\nFile size: {new System.IO.FileInfo(imagePath).Length / 1024} KB";
        MessageBox.Show(info, "Face Info", MessageBoxButton.OK, MessageBoxImage.Information);
    }

    [RelayCommand]
    private void OpenImage(string? imagePath)
    {
        if (string.IsNullOrEmpty(imagePath) || !System.IO.File.Exists(imagePath)) return;

        try
        {
            System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo
            {
                FileName = imagePath,
                UseShellExecute = true
            });
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Cannot open image: {ex.Message}", "Error", MessageBoxButton.OK, MessageBoxImage.Error);
        }
    }
}

public class RecognizedFaceViewModel
{
    public string ImagePath { get; set; } = string.Empty;
    public string Name { get; set; } = string.Empty;
    public float Confidence { get; set; }
    public byte[]? Thumbnail { get; set; }
}

public class KnownPersonClusterViewModel
{
    public int ClusterId { get; set; }
    public string PersonName { get; set; } = string.Empty;
    public ObservableCollection<RecognizedFaceViewModel> Faces { get; set; } = [];
}

public partial class PersonClusterViewModel : ObservableObject
{
    [ObservableProperty] private int _clusterId;
    [ObservableProperty] private int _faceCount;
    public ObservableCollection<ClusterFaceViewModel> Faces { get; set; } = [];
}

public class ClusterFaceViewModel
{
    public string ImagePath { get; set; } = string.Empty;
    public float[]? Embedding { get; set; }
    public byte[]? Thumbnail { get; set; }
}

public partial class SelectableNameViewModel : ObservableObject
{
    public string Name { get; }

    [ObservableProperty] private bool _isExcluded;

    public SelectableNameViewModel(string name)
    {
        Name = name;
    }
}