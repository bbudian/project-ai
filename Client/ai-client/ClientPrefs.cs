using System.Text.Json;

// Client-local preferences persisted to user://settings.json (docs/CLIENT_DESIGN.md 1a). These are THIS machine's
// choices — server URL, picker defaults, decoding settings, font size — never server state. Loaded once in
// Main._Ready before the first /health call; saved debounced whenever AppState raises PrefsChanged.
public sealed class ClientPrefs
{
    public string ServerUrl { get; set; } = "http://localhost:8080";
    public int FontSize { get; set; } = Palette.DefaultFontSize;
    public string DefaultModel { get; set; } = "";    // last model the user chose; "" = follow the server default
    public string DefaultBackend { get; set; } = "";  // last backend the user chose; "" = follow the server default
    public bool Sample { get; set; }
    public float Temperature { get; set; } = 0.8f;
    public int TopK { get; set; } = 40;
    public float TopP { get; set; } = 0.9f;
    public int MaxTokens { get; set; }                // 0 = dynamic (until the model stops / context fills)
    public bool Research { get; set; }
    public bool Memory { get; set; }                  // memory opt-in rides the chat start frame (P1b)
    public ulong Seed { get; set; }
    public string LastView { get; set; } = ViewIds.Chat;

    // ---- local server lifecycle (the app can spawn `projectai serve` itself) ----
    public string ServerExePath { get; set; } = "";   // "" = auto-discover from the repo/exported layout
    public string ServerModelsDir { get; set; } = ""; // "" = auto-discover <repo>/checkpoints
    public string ServerExtraArgs { get; set; } = "--backend torch --device cuda";
    public bool AutoStartServer { get; set; }         // spawn on launch when the server is unreachable
    public bool StopServerOnExit { get; set; } = true; // kill a server WE started when the app closes
}

public static class PrefsStore
{
    private const string Path = "user://settings.json";
    private static readonly JsonSerializerOptions Options = new() { WriteIndented = true };

    public static ClientPrefs Load()
    {
        if (!Godot.FileAccess.FileExists(Path)) return new ClientPrefs();
        using var file = Godot.FileAccess.Open(Path, Godot.FileAccess.ModeFlags.Read);
        if (file is null) return new ClientPrefs();
        try { return JsonSerializer.Deserialize<ClientPrefs>(file.GetAsText()) ?? new ClientPrefs(); }
        catch (JsonException) { return new ClientPrefs(); } // corrupted file → defaults; the next save overwrites it
    }

    public static void Save(ClientPrefs prefs)
    {
        using var file = Godot.FileAccess.Open(Path, Godot.FileAccess.ModeFlags.Write);
        file?.StoreString(JsonSerializer.Serialize(prefs, Options));
    }
}
