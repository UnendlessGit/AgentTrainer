using System.Text;
using AgentTrainer.Recorder.Core;

namespace AgentTrainer.Recorder.Core.Tests;

public sealed class CompatibilityTests
{
    [Theory]
    [InlineData(0x1E, false, 0, 0)]
    [InlineData(0x13, false, 0, 15)]
    [InlineData(0x1D, true, 0, 62)]
    [InlineData(0x5B, true, 0, 55)]
    [InlineData(0x48, true, 0, 126)]
    [InlineData(0, false, MacKeyMap.VkSnapshot, 105)]
    public void WindowsPhysicalKeysUseTheMacPolicyKeySpace(ushort scanCode, bool extended, ushort virtualKey, ushort expected) =>
        Assert.Equal(expected, MacKeyMap.Translate(scanCode, extended, virtualKey));

    [Fact]
    public void ManifestJsonMatchesFoundationDateAndFieldConventions()
    {
        var manifest = RecordingTestData.Manifest(Guid.Parse("A33D4762-3F54-4F86-9214-CA19F2845388"), null);
        var json = Encoding.UTF8.GetString(RecordingJson.Serialize(manifest));
        Assert.Contains("\"createdAt\": \"2026-07-19T12:34:56Z\"", json, StringComparison.Ordinal);
        Assert.Contains("\"schemaVersion\": 2", json, StringComparison.Ordinal);
        Assert.DoesNotContain(".000Z", json, StringComparison.Ordinal);
        var decoded = RecordingJson.Deserialize<RecordingManifest>(Encoding.UTF8.GetBytes(json));
        Assert.Equal(manifest.Id, decoded.Id);
        Assert.Equal(manifest.CreatedAt, decoded.CreatedAt);
        Assert.Equal(manifest.Capture, decoded.Capture);
        Assert.Equal(manifest.ExcludedKeyCodes, decoded.ExcludedKeyCodes);
        Assert.True(decoded.IsStructurallyValid);
    }
}
