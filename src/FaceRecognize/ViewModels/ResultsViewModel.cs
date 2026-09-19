using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using FaceRecognize.Abstractions;
using SixLabors.ImageSharp.Processing;

namespace FaceRecognize.ViewModels;

public partial class ResultsViewModel : ObservableObject
{
    private readonly IVectorStore _vectorStore;
    private readonly IFaceRecognizer _recognizer;
    private readonly IClusteringService _clusteringService;
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

    public ResultsViewModel(ScanResult result, IVectorStore vectorStore, IFaceRecognizer recognizer, IClusteringService clusteringService)
    {
        _vectorStore = vectorStore;
        _recognizer = recognizer;
        _clusteringService = clusteringService;

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
    {
        var fromResults = _allKnownFaces.Select(f => f.Name);
        var fromDb = _vectorStore.GetAll().Select(f => f.Name);
        return fromResults.Concat(fromDb).Distinct().Order();
    }

    [RelayCommand]
    private void AddToKnown(string? imagePath)
    {
        if (string.IsNullOrEmpty(imagePath)) return;

        var name = Views.InputDialog.ShowDialog(
            "Add to Known Faces",
            "Enter or select name for this face:",
            GetExistingNames());

        if (string.IsNullOrWhiteSpace(name)) return;

        try
        {
            var detectedFaces = _recognizer.ExtractAllFaces(imagePath);
            if (detectedFaces.Count == 0) return;

            var detected = detectedFaces.First();
            var thumb = CreateFaceThumbnail(imagePath, detected.Box.X, detected.Box.Y, detected.Box.Width, detected.Box.Height, 128);
            var face = new KnownFace
            {
                Name = name,
                ImagePath = imagePath,
                Embedding = detected.Embedding,
                Thumbnail = thumb
            };
            _vectorStore.Add(face);

            // Add to in-memory known faces
            _allKnownFaces.Add(new RecognizedFaceViewModel
            {
                ImagePath = imagePath,
                Name = name,
                Confidence = 1.0f
            });

            // Collect all remaining unknown embeddings with thumbnails
            var remainingUnknowns = new List<(string path, float[] embedding, byte[]? thumb)>();
            foreach (var cluster in UnknownClusters)
            {
                foreach (var faceInCluster in cluster.Faces)
                {
                    if (faceInCluster.ImagePath != imagePath && faceInCluster.Embedding != null)
                    {
                        remainingUnknowns.Add((faceInCluster.ImagePath, faceInCluster.Embedding, faceInCluster.Thumbnail));
                    }
                }
            }

            // Re-check: which of the remaining unknowns now match the updated vector store?
            var newlyMatched = new List<(string path, string name, float score)>();
            var stillUnknown = new List<(string path, float[] embedding, byte[]? thumb)>();

            foreach (var (path, emb, faceThumb) in remainingUnknowns)
            {
                var (best, score) = _vectorStore.Search(emb, 0.58f);
                if (best != null)
                {
                    newlyMatched.Add((path, best.Name, score));
                }
                else
                {
                    stillUnknown.Add((path, emb, faceThumb));
                }
            }

            // Add newly matched to known
            foreach (var (path, matchedName, score) in newlyMatched)
            {
                _allKnownFaces.Add(new RecognizedFaceViewModel
                {
                    ImagePath = path,
                    Name = matchedName,
                    Confidence = score
                });
            }

            // Re-cluster remaining unknowns
            UnknownClusters.Clear();
            if (stillUnknown.Count > 0)
            {
                var clusterInput = stillUnknown.Select(u => (u.path, u.embedding, u.thumb)).ToList();
                var clusters = _clusteringService.ClusterUnknownFaces(clusterInput);

                foreach (var cluster in clusters)
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

            // Update names list and refresh
            RefreshKnownPeopleNames();
            RefreshKnownFaces();
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

    private static byte[]? CreateThumbnail(string imagePath, int maxSize)
    {
        try
        {
            using var original = SixLabors.ImageSharp.Image.Load(imagePath);
            original.Mutate(x => x.Resize(new SixLabors.ImageSharp.Processing.ResizeOptions
            {
                Size = new SixLabors.ImageSharp.Size(maxSize, maxSize),
                Mode = SixLabors.ImageSharp.Processing.ResizeMode.Crop
            }));

            using var ms = new MemoryStream();
            original.Save(ms, new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder());
            return ms.ToArray();
        }
        catch
        {
            return null;
        }
    }

    private static byte[]? CreateFaceThumbnail(string imagePath, float x, float y, float width, float height, int outputSize)
    {
        try
        {
            using var original = SixLabors.ImageSharp.Image.Load(imagePath);

            var borderX = (int)(width * 0.3f);
            var borderY = (int)(height * 0.3f);

            int cropX = Math.Max(0, (int)x - borderX);
            int cropY = Math.Max(0, (int)y - borderY);
            int cropW = Math.Min(original.Width - cropX, (int)width + borderX * 2);
            int cropH = Math.Min(original.Height - cropY, (int)height + borderY * 2);

            original.Mutate(ctx => ctx.Crop(new SixLabors.ImageSharp.Rectangle(cropX, cropY, cropW, cropH)));
            original.Mutate(ctx => ctx.Resize(new SixLabors.ImageSharp.Processing.ResizeOptions
            {
                Size = new SixLabors.ImageSharp.Size(outputSize, outputSize),
                Mode = SixLabors.ImageSharp.Processing.ResizeMode.Crop
            }));

            using var ms = new MemoryStream();
            original.Save(ms, new SixLabors.ImageSharp.Formats.Jpeg.JpegEncoder());
            return ms.ToArray();
        }
        catch
        {
            return null;
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
