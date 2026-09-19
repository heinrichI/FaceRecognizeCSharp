using System.Collections.ObjectModel;
using System.Windows;

namespace FaceRecognize.Views;

public partial class InputDialog : Window
{
    public string? Result { get; private set; }

    public InputDialog(string title, string prompt, IEnumerable<string> existingNames, string defaultValue = "")
    {
        InitializeComponent();
        Title = title;
        PromptText.Text = prompt;

        var sorted = existingNames.OrderBy(n => n).ToList();
        NameComboBox.ItemsSource = sorted;

        if (!string.IsNullOrEmpty(defaultValue))
        {
            var index = sorted.IndexOf(defaultValue);
            if (index >= 0)
                NameComboBox.SelectedIndex = index;
            else
                NameComboBox.Text = defaultValue;
        }
        else if (sorted.Count > 0)
            NameComboBox.SelectedIndex = 0;

        NameComboBox.Focus();
    }

    public static string? ShowDialog(string title, string prompt, IEnumerable<string> existingNames, string defaultValue = "")
    {
        var dialog = new InputDialog(title, prompt, existingNames, defaultValue);
        return dialog.ShowDialog() == true ? dialog.Result : null;
    }

    private void Ok_Click(object sender, RoutedEventArgs e)
    {
        Result = NameComboBox.Text;
        DialogResult = true;
    }

    private void Cancel_Click(object sender, RoutedEventArgs e)
    {
        DialogResult = false;
    }
}
