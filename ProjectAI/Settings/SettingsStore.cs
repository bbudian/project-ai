using System.Text.Json;

// Server-side app settings, persisted at config/settings.json — deliberately OUTSIDE the models directory so a
// `git add` of a models/checkpoints tree can never sweep configuration (or, worse, secrets) into a repo. Writes
// are atomic (temp + rename); reads are a volatile snapshot so request threads never see a half-applied update.
// v1 scope (docs/CLIENT_DESIGN.md §7): the memory injection budgets only — client-local preferences stay in the
// client, and per-model sidecars come later.
internal sealed record MemorySettings(
    int BridgeCards = 24, int BridgeBudget = 400, int RecallHits = 3, int RecallBudget = 400);

internal sealed record AppSettings(MemorySettings Memory)
{
    public static AppSettings Defaults => new(new MemorySettings());
}

internal static class SettingsStore
{
    private const string FilePath = "config/settings.json";
    private static readonly JsonSerializerOptions JsonOpts = new()
    { PropertyNameCaseInsensitive = true, WriteIndented = true };

    private static volatile AppSettings _current = Load();

    public static AppSettings Current => _current;

    private static AppSettings Load()
    {
        try
        {
            if (!File.Exists(FilePath)) return AppSettings.Defaults;
            var parsed = JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(FilePath), JsonOpts);
            return parsed?.Memory is null ? AppSettings.Defaults : parsed;
        }
        catch (Exception) { return AppSettings.Defaults; } // unreadable file → defaults; the next save overwrites it
    }

    /// <summary>Validates and applies new settings; on success they take effect immediately (volatile swap) and
    /// persist atomically. Returns the problems instead of throwing so PUT /config can 400 with all of them.</summary>
    public static IReadOnlyList<string> Update(AppSettings next)
    {
        var problems = new List<string>();
        var m = next.Memory;
        if (m.BridgeCards is < 0 or > 200) problems.Add("memory.bridgeCards must be in [0, 200]");
        if (m.BridgeBudget is < 0 or > 100_000) problems.Add("memory.bridgeBudget must be in [0, 100000]");
        if (m.RecallHits is < 0 or > 64) problems.Add("memory.recallHits must be in [0, 64]");
        if (m.RecallBudget is < 0 or > 100_000) problems.Add("memory.recallBudget must be in [0, 100000]");
        if (problems.Count > 0) return problems;

        Directory.CreateDirectory(Path.GetDirectoryName(FilePath)!);
        string tmp = FilePath + ".tmp";
        File.WriteAllText(tmp, JsonSerializer.Serialize(next, JsonOpts));
        File.Move(tmp, FilePath, overwrite: true);
        _current = next;
        return problems;
    }
}
