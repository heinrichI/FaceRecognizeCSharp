using System.Collections.ObjectModel;
using CommunityToolkit.Mvvm.ComponentModel;
using CommunityToolkit.Mvvm.Input;
using FaceRecognize.Abstractions;

namespace FaceRecognize.ViewModels;

public partial class KnownFacesViewModel : ObservableObject
{
    private readonly IVectorStore _vectorStore;

    public ObservableCollection<KnownPersonViewModel> People { get; } = [];

    public KnownFacesViewModel(IVectorStore vectorStore)
    {
        _vectorStore = vectorStore;
        Refresh();
    }

    [RelayCommand]
    private void Refresh()
    {
        var allFaces = _vectorStore.GetAll();
        var grouped = allFaces
            .GroupBy(f => f.Name)
            .Select(g => new KnownPersonViewModel
            {
                Name = g.Key,
                DescriptorCount = g.Count(),
                Faces = g.Select(f => new KnownFaceItemViewModel
                {
                    ImagePath = f.ImagePath,
                    CreatedAt = f.CreatedAt,
                    Thumbnail = f.Thumbnail
                }).ToList()
            })
            .OrderBy(p => p.Name)
            .ToList();

        People.Clear();
        foreach (var person in grouped)
            People.Add(person);
    }
}

public class KnownPersonViewModel
{
    public string Name { get; set; } = string.Empty;
    public int DescriptorCount { get; set; }
    public List<KnownFaceItemViewModel> Faces { get; set; } = [];
}

public class KnownFaceItemViewModel
{
    public string ImagePath { get; set; } = string.Empty;
    public DateTime CreatedAt { get; set; }
    public byte[]? Thumbnail { get; set; }
}
