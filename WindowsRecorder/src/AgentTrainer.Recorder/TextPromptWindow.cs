using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;

namespace AgentTrainer.Recorder;

internal sealed class TextPromptWindow : Window
{
    private readonly TextBox _textBox;

    private TextPromptWindow(Window owner, string title, string prompt, string initialValue)
    {
        Owner = owner;
        Title = title;
        Width = 420;
        Height = 190;
        ResizeMode = ResizeMode.NoResize;
        WindowStartupLocation = WindowStartupLocation.CenterOwner;
        ShowInTaskbar = false;
        var root = new Grid { Margin = new Thickness(20) };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.Children.Add(new TextBlock { Text = prompt, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 0, 0, 10) });
        _textBox = new TextBox { Text = initialValue };
        Grid.SetRow(_textBox, 1);
        root.Children.Add(_textBox);
        var buttons = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Bottom };
        var cancel = new Button { Content = "Cancel", Margin = new Thickness(0, 0, 8, 0), IsCancel = true };
        var save = new Button { Content = "Save", IsDefault = true };
        save.Click += (_, _) => { if (!string.IsNullOrWhiteSpace(_textBox.Text)) DialogResult = true; };
        buttons.Children.Add(cancel);
        buttons.Children.Add(save);
        Grid.SetRow(buttons, 2);
        root.Children.Add(buttons);
        Content = root;
        Loaded += (_, _) => { _textBox.Focus(); _textBox.SelectAll(); };
        PreviewKeyDown += (_, args) =>
        {
            if (args.Key == Key.Enter && !string.IsNullOrWhiteSpace(_textBox.Text)) DialogResult = true;
        };
    }

    internal static string? Show(Window owner, string title, string prompt, string initialValue = "")
    {
        var window = new TextPromptWindow(owner, title, prompt, initialValue);
        return window.ShowDialog() == true ? window._textBox.Text.Trim() : null;
    }
}
