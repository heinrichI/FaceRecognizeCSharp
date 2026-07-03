using System.Collections.ObjectModel;
using System.IO;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using FaceRecognize.Abstractions;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Processing;

namespace FaceRecognize.ViewModels;

public partial class MainViewModel : ObservableObject, IDisposable
{
    private readonly IFaceRecognizer _recognizer;
    private readonly IVectorStore _vectorStore;
    private readonly IClusteringService _clusteringService;
    private readonly IImageScanner _imageScanner;

    [ObservableProperty] private string _statusMessage = "Ready";
    [ObservableProperty] private string _newFaceName = string.Empty;
    [ObservableProperty] private string _scanDirectoryPath = string.Empty;
    [ObservableProperty] private bool _isProcessing;
    [ObservableProperty] private int _knownFaceCount;

    public ObservableCollection<KnownFaceViewModel> KnownFaces { get; } = [];

    public MainViewModel(
        IFaceRecognizer recognizer,
        IVectorStore vectorStore,
        IClusteringService clusteringService,
        IImageScanner imageScanner)
    {
        _recognizer = recognizer;
        _vectorStore = vectorStore;
        _clusteringService = clusteringService;
        _imageScanner = imageScanner;

        KnownFaceCount = _vectorStore.Count;
        RefreshKnownFaces();

        var modelPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "lm_model3_opt.onnx");
        if (!File.Exists(modelPath))
        {
            MessageBox.Show(
                "3D-модель лендмарков не найдена (lm_model3_opt.onnx).\n" +
                "Выравнивание лиц будет работать в 2D-режиме.\n" +
                "Скачайте модель в папку Models/ для 3D-нормализации.",
                "Предупреждение", MessageBoxButton.OK, MessageBoxImage.Warning);
        }
    }

    [RelayCommand]
    private async Task AddKnownFaceAsync()
    {
        if (string.IsNullOrWhiteSpace(NewFaceName))
        {
            StatusMessage = "Enter a name for the face.";
            return;
        }

        var dialog = new Microsoft.Win32.OpenFileDialog
        {
            Filter = "Images|*.jpg;*.jpeg;*.png;*.bmp;*.gif;*.webp|All files|*.*",
            Multiselect = true
        };

        if (dialog.ShowDialog() != true) return;

        IsProcessing = true;
        StatusMessage = "Processing...";

        try
        {
            int added = 0;
            foreach (var path in dialog.FileNames)
            {
                var embedding = _recognizer.ExtractEmbedding(path);
                var face = new KnownFace
                {
                    Name = NewFaceName,
                    ImagePath = path,
                    Embedding = embedding
                };
                _vectorStore.Add(face);
                added++;
            }

            KnownFaceCount = _vectorStore.Count;
            StatusMessage = $"Added {added} face(s) for '{NewFaceName}'.";
            NewFaceName = string.Empty;
            RefreshKnownFaces();
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private async Task ScanDirectoryAsync()
    {
        if (string.IsNullOrWhiteSpace(ScanDirectoryPath) || !Directory.Exists(ScanDirectoryPath))
        {
            StatusMessage = "Enter a valid directory path.";
            return;
        }

        IsProcessing = true;
        StatusMessage = "Scanning directory...";

        try
        {
            var images = _imageScanner.ScanDirectory(ScanDirectoryPath);
            StatusMessage = $"Found {images.Count} images. Processing...";

            var results = new List<RecognizedFace>();
            var unknownEmbeddings = new List<(string path, float[] embedding, byte[]? thumb)>();
            int faceCount = 0;

            foreach (var imgPath in images)
            {
                try
                {
                    var faces = _recognizer.ExtractAllFaces(imgPath);
                    foreach (var face in faces)
                    {
                        faceCount++;
                        var (best, score) = _vectorStore.Search(face.Embedding);

                        if (best != null)
                        {
                            results.Add(new RecognizedFace
                            {
                                ImagePath = imgPath,
                                Name = best.Name,
                                Confidence = score
                            });
                        }
                        else
                        {
                            var thumb = CreateThumbnail(imgPath, 64);
                            unknownEmbeddings.Add((imgPath, face.Embedding, thumb));
                        }
                    }
                }
                catch (Exception ex)
                {
                    StatusMessage = $"Error processing {Path.GetFileName(imgPath)}: {ex.Message}";
                }
            }

            var clusters = _clusteringService.ClusterUnknownFaces(unknownEmbeddings);

            var scanResult = new ScanResult
            {
                KnownFaces = results,
                UnknownClusters = clusters,
                TotalImagesScanned = images.Count,
                TotalFacesFound = faceCount
            };

            StatusMessage = $"Done: {results.Count} known, {unknownEmbeddings.Count} unknown in {faceCount} faces.";

            var resultsWindow = new Views.ResultsWindow(scanResult);
            resultsWindow.Show();
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
        }
    }

    [RelayCommand]
    private void ClearAll()
    {
        _vectorStore.ClearAll();
        KnownFaceCount = 0;
        KnownFaces.Clear();
        StatusMessage = "All known faces cleared.";
    }

    [RelayCommand]
    private void BrowseDirectory()
    {
        var dialog = new Microsoft.Win32.OpenFolderDialog
        {
            Title = "Select directory with images"
        };

        if (dialog.ShowDialog() == true)
        {
            ScanDirectoryPath = dialog.FolderName;
        }
    }

    private void RefreshKnownFaces()
    {
        KnownFaces.Clear();
        foreach (var face in _vectorStore.GetAll())
        {
            KnownFaces.Add(new KnownFaceViewModel
            {
                Name = face.Name,
                ImagePath = face.ImagePath,
                CreatedAt = face.CreatedAt
            });
        }
    }

    private static byte[]? CreateThumbnail(string imagePath, int maxSize)
    {
        try
        {
            using var original = SixLabors.ImageSharp.Image.Load(imagePath);
            original.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new SixLabors.ImageSharp.Size(maxSize, maxSize),
                Mode = SixLabors.ImageSharp.Processing.ResizeMode.Crop
            }));

            using var ms = new MemoryStream();
            original.Save(ms, new JpegEncoder());
            return ms.ToArray();
        }
        catch
        {
            return null;
        }
    }

    public void Dispose()
    {
        _recognizer.Dispose();
        _vectorStore.Dispose();
    }
}

public class KnownFaceViewModel
{
    public string Name { get; set; } = string.Empty;
    public string ImagePath { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
}
