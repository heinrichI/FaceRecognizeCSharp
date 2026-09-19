using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Windows;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using FaceRecognize.Abstractions;
using FaceRecognize.Data;

namespace FaceRecognize.ViewModels;

public partial class MainViewModel : ObservableObject, IDisposable
{
    private readonly IFaceRecognizer _recognizer;
    private readonly IVectorStore _vectorStore;
    private readonly IConfiguration _configuration;
    private readonly IDatabaseMaintenanceService _maintenanceService;
    private readonly IEmbeddingCache _embeddingCache;
    private readonly IModelDownloader _modelDownloader;
    private readonly IFaceScanService _faceScanService;
    private readonly IFaceEnrollmentService _enrollmentService;
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
        IConfiguration configuration,
        IDatabaseMaintenanceService maintenanceService,
        IEmbeddingCache embeddingCache,
        IModelDownloader modelDownloader,
        IFaceScanService faceScanService,
        IFaceEnrollmentService enrollmentService)
    {
        _recognizer = recognizer;
        _vectorStore = vectorStore;
        _configuration = configuration;
        _maintenanceService = maintenanceService;
        _embeddingCache = embeddingCache;
        _modelDownloader = modelDownloader;
        _faceScanService = faceScanService;
        _enrollmentService = enrollmentService;

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
        => ModelPathResolver.Resolve(ModelsDirectory, AppDomain.CurrentDomain.BaseDirectory);

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
            var modelPath = GetModelPath();
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
    private async Task AddKnownFace()
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
            var paths = dialog.FileNames.ToList();
            int added = await Task.Run(() =>
                _enrollmentService.EnrollByName(NewFaceName, paths, new Progress<int>(v => ProgressValue = v)));

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

        Debug.Assert(Directory.Exists(ScanDirectoryPath), "Scan directory must exist");
        Debug.Assert(ThreadCount >= 1, "ThreadCount must be >= 1");

        _scanCts = new CancellationTokenSource();
        var token = _scanCts.Token;
        IsProcessing = true;
        ProgressVisible = Visibility.Visible;
        ProgressValue = 0;
        StatusMessage = "Scanning directory...";

        // Создаётся в UI-контексте: Progress<T> сам постит обновления в UI-поток.
        var progress = new Progress<ScanProgress>(p =>
        {
            ProgressMaximum = p.Total;
            ProgressValue = p.Completed;
            ProgressText = p.FileName;
        });

        try
        {
            var threshold = (float)DistanceThreshold;
            var scanResult = await Task.Run(
                () => _faceScanService.ScanDirectory(ScanDirectoryPath, threshold, ThreadCount, progress, token),
                token);

            StatusMessage = (scanResult.Cancelled ? "Stopped" : "Done")
                + $": {scanResult.KnownFaces.Count} known, "
                + $"{scanResult.UnknownClusters.Sum(c => c.Faces.Count)} unknown "
                + $"in {scanResult.TotalFacesFound} faces.";

            var resultsWindow = new Views.ResultsWindow(scanResult, _enrollmentService, threshold);
            resultsWindow.Show();
        }
        catch (OperationCanceledException)
        {
            StatusMessage = "Scan stopped.";
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
        Debug.Assert(_vectorStore.Count == 0, "Vector store not empty after ClearAll");
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

        Debug.Assert(KnownFaces.Count == _vectorStore.Count, "Known faces view out of sync with store");
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

    /// <summary>
    /// Конфигурация сохраняется; синглтоны (recognizer, store, cache)
    /// разворачивает контейнер в App.OnExit — не разворачиваем их здесь.
    /// </summary>
    public void Dispose()
    {
        SaveConfig();
    }
}

public class KnownFaceViewModel
{
    public string Name { get; set; } = string.Empty;
    public string ImagePath { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
}