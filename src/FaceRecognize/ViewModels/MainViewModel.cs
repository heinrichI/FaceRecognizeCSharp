using System.Collections.ObjectModel;
using System.IO;
using System.Threading;
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
    private readonly IConfiguration _configuration;
    private readonly IDatabaseMaintenanceService _maintenanceService;
    private readonly IEmbeddingCache _embeddingCache;
    private readonly IModelDownloader _modelDownloader;
    private CancellationTokenSource? _scanCts;

    [ObservableProperty] private string _statusMessage = "Ready";
    [ObservableProperty] private string _newFaceName = string.Empty;
    [ObservableProperty] private string _scanDirectoryPath = string.Empty;
    [ObservableProperty] private bool _isProcessing;
    [ObservableProperty] private int _knownFaceCount;

    // Progress bar
    [ObservableProperty] private int _progressValue;
    [ObservableProperty] private int _progressMaximum;
    [ObservableProperty] private string _progressText = string.Empty;
    [ObservableProperty] private Visibility _progressVisible = Visibility.Collapsed;

    // Configuration bindings
    [ObservableProperty] private string _modelsDirectory = string.Empty;
    [ObservableProperty] private double _distanceThreshold = 0.58;
    [ObservableProperty] private int _threadCount = Environment.ProcessorCount;
    [ObservableProperty] private string _alignmentStatus = string.Empty;

    public ObservableCollection<KnownFaceViewModel> KnownFaces { get; } = [];

    public MainViewModel(
        IFaceRecognizer recognizer,
        IVectorStore vectorStore,
        IClusteringService clusteringService,
        IImageScanner imageScanner,
        IConfiguration configuration,
        IDatabaseMaintenanceService maintenanceService,
        IEmbeddingCache embeddingCache,
        IModelDownloader modelDownloader)
    {
        _recognizer = recognizer;
        _vectorStore = vectorStore;
        _clusteringService = clusteringService;
        _imageScanner = imageScanner;
        _configuration = configuration;
        _maintenanceService = maintenanceService;
        _embeddingCache = embeddingCache;
        _modelDownloader = modelDownloader;

        // Load saved config
        ModelsDirectory = _configuration.ModelsDirectory;
        DistanceThreshold = _configuration.DistanceThreshold;
        ThreadCount = _configuration.ThreadCount;
        ScanDirectoryPath = _configuration.LastScanPath;

        KnownFaceCount = _vectorStore.Count;
        RefreshKnownFaces();

        UpdateAlignmentStatus();
        CheckAndOfferModelDownload();
    }

    private string GetModelPath()
    {
        return !string.IsNullOrEmpty(ModelsDirectory)
            ? Path.Combine(ModelsDirectory, "lm_model3_opt.onnx")
            : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "lm_model3_opt.onnx");
    }

    private const string ModelDownloadUrl = "https://github.com/AlfredoRamos/FaceRecognition/raw/main/models/lm_model3_opt.onnx";

    private async void CheckAndOfferModelDownload()
    {
        if (_recognizer.Is3DAlignmentAvailable)
            return;

        var modelPath = GetModelPath();
        if (_modelDownloader.IsModelPresent(modelPath))
            return;

        var result = MessageBox.Show(
            "3D landmark model (lm_model3_opt.onnx) not found.\n\n" +
            "Download it now? (~5 MB)\n" +
            "Required for 3D face alignment.",
            "Download Model",
            MessageBoxButton.YesNo,
            MessageBoxImage.Question);

        if (result == MessageBoxResult.Yes)
        {
            await DownloadModelAsync();
        }
    }

    [RelayCommand]
    private async Task DownloadModelAsync()
    {
        var modelPath = GetModelPath();
        IsProcessing = true;
        ProgressVisible = Visibility.Visible;
        ProgressMaximum = 100;
        ProgressValue = 0;
        StatusMessage = "Downloading 3D model...";

        try
        {
            var progress = new Progress<int>(p =>
            {
                ProgressValue = p;
                ProgressText = $"Downloading... {p}%";
            });

            var success = await _modelDownloader.EnsureModelExistsAsync(modelPath, ModelDownloadUrl, progress);

            if (success)
            {
                StatusMessage = "Model downloaded successfully. Restart required for 3D alignment.";
                UpdateAlignmentStatus();
            }
            else
            {
                StatusMessage = "Model download failed.";
            }
        }
        catch (Exception ex)
        {
            StatusMessage = $"Download error: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
            ProgressVisible = Visibility.Collapsed;
        }
    }

    private void UpdateAlignmentStatus()
    {
        if (_recognizer.Is3DAlignmentAvailable)
        {
            AlignmentStatus = "3D: Active";
        }
        else
        {
            var modelPath = !string.IsNullOrEmpty(ModelsDirectory)
                ? Path.Combine(ModelsDirectory, "lm_model3_opt.onnx")
                : Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models", "lm_model3_opt.onnx");

            if (File.Exists(modelPath))
                AlignmentStatus = "3D: Model found but not loaded (restart required)";
            else
                AlignmentStatus = $"3D: Not available (model not found at {Path.GetDirectoryName(modelPath)})";
        }
    }

    partial void OnModelsDirectoryChanged(string value)
    {
        SaveConfig();
        UpdateAlignmentStatus();
        StatusMessage = "Models directory changed. Restart required for 3D alignment.";
    }
    partial void OnDistanceThresholdChanged(double value) => SaveConfig();
    partial void OnThreadCountChanged(int value) => SaveConfig();
    partial void OnScanDirectoryPathChanged(string value) => _configuration.LastScanPath = value;

    private void SaveConfig()
    {
        _configuration.ModelsDirectory = ModelsDirectory;
        _configuration.DistanceThreshold = DistanceThreshold;
        _configuration.ThreadCount = ThreadCount;
        _configuration.Save();
    }

    [RelayCommand]
    private void AddKnownFace()
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
        ProgressVisible = Visibility.Visible;
        ProgressMaximum = dialog.FileNames.Length;
        ProgressValue = 0;

        try
        {
            int added = 0;
            foreach (var path in dialog.FileNames)
            {
                ProgressText = Path.GetFileName(path);
                ProgressValue = added + 1;

                var faces = _recognizer.ExtractAllFaces(path);
                foreach (var face in faces)
                {
                    var thumb = CreateFaceThumbnail(path, face.Box.X, face.Box.Y, face.Box.Width, face.Box.Height, 128);
                    var knownFace = new KnownFace
                    {
                        Name = NewFaceName,
                        ImagePath = path,
                        Embedding = face.Embedding,
                        Thumbnail = thumb
                    };
                    _vectorStore.Add(knownFace);
                    added++;
                }
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
            ProgressVisible = Visibility.Collapsed;
        }
    }

    [RelayCommand]
    private void StopScan()
    {
        _scanCts?.Cancel();
    }

    [RelayCommand]
    private async Task ScanDirectoryAsync()
    {
        if (string.IsNullOrWhiteSpace(ScanDirectoryPath) || !Directory.Exists(ScanDirectoryPath))
        {
            StatusMessage = "Enter a valid directory path.";
            return;
        }

        _scanCts = new CancellationTokenSource();
        var token = _scanCts.Token;
        IsProcessing = true;
        ProgressVisible = Visibility.Visible;
        StatusMessage = "Scanning directory...";

        try
        {
            var images = _imageScanner.ScanDirectory(ScanDirectoryPath);
            ProgressMaximum = images.Count;
            ProgressValue = 0;
            StatusMessage = $"Found {images.Count} images. Processing...";

            var results = new List<RecognizedFace>();
            var unknownEmbeddings = new List<(string path, float[] embedding, byte[]? thumb)>();
            int faceCount = 0;
            int processed = 0;
            var lockObj = new object();
            bool cancelled = false;

            var parallelOptions = new ParallelOptions
            {
                MaxDegreeOfParallelism = ThreadCount,
                CancellationToken = token
            };

            try
            {
                await Task.Run(() =>
                {
                    Parallel.ForEach(images, parallelOptions, imgPath =>
                    {
                        var current = Interlocked.Increment(ref processed);
                        Application.Current.Dispatcher.Invoke(() =>
                        {
                            ProgressValue = current;
                            ProgressText = Path.GetFileName(imgPath);
                        });

                        try
                        {
                            var fileInfo = new FileInfo(imgPath);
                            var fileSize = fileInfo.Length;
                            var lastModified = fileInfo.LastWriteTimeUtc;

                            var cachedFaces = _embeddingCache.GetCachedFaces(imgPath, fileSize, lastModified);
                            CachedFace[] faceArray;

                            if (cachedFaces != null)
                            {
                                faceArray = cachedFaces.ToArray();
                            }
                            else
                            {
                                var detected = _recognizer.ExtractAllFaces(imgPath);
                                faceArray = detected.Select(f => new CachedFace
                                {
                                    Embedding = f.Embedding,
                                    X = f.Box.X,
                                    Y = f.Box.Y,
                                    Width = f.Box.Width,
                                    Height = f.Box.Height
                                }).ToArray();
                                if (faceArray.Length > 0)
                                    _embeddingCache.Store(imgPath, fileSize, lastModified, faceArray.ToList());
                            }

                            foreach (var cachedFace in faceArray)
                            {
                                var (best, score) = _vectorStore.Search(cachedFace.Embedding, (float)DistanceThreshold);

                                if (best != null)
                                {
                                    var thumb = CreateFaceThumbnail(imgPath, cachedFace.X, cachedFace.Y, cachedFace.Width, cachedFace.Height, 128);
                                    lock (lockObj)
                                    {
                                        results.Add(new RecognizedFace
                                        {
                                            ImagePath = imgPath,
                                            Name = best.Name,
                                            Confidence = score,
                                            Thumbnail = thumb
                                        });
                                        Interlocked.Increment(ref faceCount);
                                    }
                                }
                                else
                                {
                                    var thumb = CreateFaceThumbnail(imgPath, cachedFace.X, cachedFace.Y, cachedFace.Width, cachedFace.Height, 128);
                                    lock (lockObj)
                                    {
                                        unknownEmbeddings.Add((imgPath, cachedFace.Embedding, thumb));
                                        Interlocked.Increment(ref faceCount);
                                    }
                                }
                            }
                        }
                        catch
                        {
                            // Skip unprocessable images
                        }
                    });
                });
            }
            catch (OperationCanceledException)
            {
                cancelled = true;
            }

            var clusters = _clusteringService.ClusterUnknownFaces(unknownEmbeddings);

            var scanResult = new ScanResult
            {
                KnownFaces = results,
                UnknownClusters = clusters,
                TotalImagesScanned = processed,
                TotalFacesFound = faceCount
            };

            StatusMessage = cancelled
                ? $"Stopped: {results.Count} known, {unknownEmbeddings.Count} unknown in {faceCount} faces."
                : $"Done: {results.Count} known, {unknownEmbeddings.Count} unknown in {faceCount} faces.";

            var resultsWindow = new Views.ResultsWindow(scanResult, _vectorStore, _recognizer, _clusteringService);
            resultsWindow.Show();
        }
        catch (Exception ex)
        {
            StatusMessage = $"Error: {ex.Message}";
        }
        finally
        {
            IsProcessing = false;
            ProgressVisible = Visibility.Collapsed;
            _scanCts?.Dispose();
            _scanCts = null;
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

    private static byte[]? CreateFaceThumbnail(string imagePath, float x, float y, float width, float height, int outputSize)
    {
        try
        {
            using var original = SixLabors.ImageSharp.Image.Load(imagePath);

            // Add 30% border around face for context
            var borderX = (int)(width * 0.3f);
            var borderY = (int)(height * 0.3f);

            int cropX = Math.Max(0, (int)x - borderX);
            int cropY = Math.Max(0, (int)y - borderY);
            int cropW = Math.Min(original.Width - cropX, (int)width + borderX * 2);
            int cropH = Math.Min(original.Height - cropY, (int)height + borderY * 2);

            original.Mutate(ctx => ctx.Crop(new SixLabors.ImageSharp.Rectangle(cropX, cropY, cropW, cropH)));
            original.Mutate(ctx => ctx.Resize(new ResizeOptions
            {
                Size = new SixLabors.ImageSharp.Size(outputSize, outputSize),
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

    [RelayCommand]
    private void ShowKnownFaces()
    {
        var window = new Views.KnownFacesWindow(_vectorStore);
        window.Show();
    }

    [RelayCommand]
    private void ClearCache()
    {
        _embeddingCache.Clear();
        StatusMessage = "Embedding cache cleared.";
    }

    [RelayCommand]
    private void CleanupDatabase()
    {
        var orphaned = _maintenanceService.GetOrphanedEntries();
        if (orphaned.Count == 0)
        {
            StatusMessage = "No orphaned entries found.";
            return;
        }

        var result = MessageBox.Show(
            $"Found {orphaned.Count} entries with missing files.\nRemove them?",
            "Confirm Cleanup",
            MessageBoxButton.YesNo,
            MessageBoxImage.Question);

        if (result != MessageBoxResult.Yes) return;

        int removed = _maintenanceService.RemoveOrphanedEntries();
        KnownFaceCount = _vectorStore.Count;
        RefreshKnownFaces();
        StatusMessage = $"Removed {removed} orphaned entries.";
    }

    public void Dispose()
    {
        SaveConfig();
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
