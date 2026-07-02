using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Godot;

// Launches and supervises a local `projectai serve` as a child process — the "run models from this app" story:
// Start spawns the server (its console stays visible so the logs are readable), then the controller polls /health
// until the server answers and reports every lifecycle step through Changed. The controller only ever manages a
// process IT started: an externally-launched server is simply "connected" and is never killed from here. The
// child's working directory is the repo root, so `checkpoints/`, `memory/`, and `config/` resolve exactly as they
// do when the server is run from the CLI.
public partial class ServerController : Node
{
    public event Action Changed;

    private readonly AppState _state;
    private readonly IApiClient _api;
    private Process _process; // non-null only for a server WE spawned
    private Godot.Timer _pollTimer;

    public bool Starting { get; private set; }
    public bool OwnsRunningServer => _process is { HasExited: false };
    public string Error { get; private set; } = "";

    public ServerController(AppState state, IApiClient api)
    {
        _state = state;
        _api = api;
    }

    public override void _Ready()
    {
        // While starting: poll /health (self-deduping on the ApiClient side) and watch for an early exit.
        _pollTimer = new Godot.Timer { WaitTime = 1.5, OneShot = false, Autostart = false };
        _pollTimer.Timeout += () =>
        {
            if (Starting && _process is { HasExited: true })
            {
                Fail($"the server exited (code {_process.ExitCode}) — check its console output");
                return;
            }
            _api.CheckHealth();
        };
        AddChild(_pollTimer);

        _state.HealthChanged += () =>
        {
            if (!Starting || _state.Health is not { Ok: true }) return;
            Starting = false; // online — the regular connected flow takes over from here
            _pollTimer.Stop();
            Changed?.Invoke();
        };
    }

    public void Start()
    {
        if (Starting || OwnsRunningServer) return;

        string exe = ResolveExePath();
        if (exe is null)
        {
            Fail("projectai.exe not found — build the server (`dotnet build ProjectAI`) or set its path in Settings");
            return;
        }
        string modelsDir = ResolveModelsDir();
        if (modelsDir is null)
        {
            Fail("no models directory found — set one in Settings (it needs at least one .ckpt)");
            return;
        }

        var psi = new ProcessStartInfo
        {
            FileName = exe,
            WorkingDirectory = DiscoverRepoRoot() ?? Path.GetDirectoryName(exe) ?? ".",
            UseShellExecute = false,
            CreateNoWindow = false, // a console app with no parent console gets its own window → live server logs
        };
        psi.ArgumentList.Add("serve");
        psi.ArgumentList.Add("--models");
        psi.ArgumentList.Add(modelsDir);
        psi.ArgumentList.Add("--port");
        psi.ArgumentList.Add(PortFromUrl(_state.ServerUrl).ToString());
        foreach (string arg in (_state.Prefs.ServerExtraArgs ?? "").Split(' ', StringSplitOptions.RemoveEmptyEntries))
            psi.ArgumentList.Add(arg);

        try { _process = Process.Start(psi); }
        catch (Exception e)
        {
            Fail($"could not start the server: {e.Message}");
            return;
        }

        Error = "";
        Starting = true;
        _pollTimer.Start();
        Changed?.Invoke();
        _api.CheckHealth(); // first probe right away — model load takes a while, but why wait to start asking
    }

    public void Stop()
    {
        if (_process is null) return;
        try { if (!_process.HasExited) _process.Kill(entireProcessTree: true); }
        catch (Exception) { /* already gone */ }
        _process.Dispose();
        _process = null;
        Starting = false;
        _pollTimer.Stop();
        Changed?.Invoke();
        _api.CheckHealth(); // reflect the disconnect in the status line
    }

    /// <summary>Called by Main on window close: a server we spawned dies with the app unless the pref says keep it.</summary>
    public void OnAppClosing()
    {
        if (_state.Prefs.StopServerOnExit) Stop();
    }

    private void Fail(string message)
    {
        Starting = false;
        Error = message;
        _pollTimer.Stop();
        Changed?.Invoke();
    }

    // ---- discovery ---------------------------------------------------------------------------------------------

    /// <summary>The repo root (identified by ProjectAI.sln): two up from res:// in dev, else walking up from the
    /// executable (exported build living somewhere in/near the repo). Null when nothing matches.</summary>
    public static string DiscoverRepoRoot()
    {
        var candidates = new List<string>();
        string res = ProjectSettings.GlobalizePath("res://");
        if (!string.IsNullOrEmpty(res)) candidates.Add(Path.GetFullPath(Path.Combine(res, "..", "..")));
        string dir = Path.GetDirectoryName(OS.GetExecutablePath());
        for (int up = 0; up < 5 && !string.IsNullOrEmpty(dir); up++, dir = Path.GetDirectoryName(dir))
            candidates.Add(dir);
        foreach (string c in candidates)
            if (File.Exists(Path.Combine(c, "ProjectAI.sln"))) return c;
        return null;
    }

    /// <summary>The server executable: the pref if set, else the repo's Debug/Release build output.</summary>
    public string ResolveExePath()
    {
        if (_state.Prefs.ServerExePath is { Length: > 0 } configured)
            return File.Exists(configured) ? configured : null;
        if (DiscoverRepoRoot() is not { } repo) return null;
        foreach (string flavor in new[] { "Debug", "Release" })
        {
            string candidate = Path.Combine(repo, "ProjectAI", "bin", flavor, "net10.0", "projectai.exe");
            if (File.Exists(candidate)) return candidate;
        }
        return null;
    }

    /// <summary>The models directory: the pref if set, else &lt;repo&gt;/checkpoints.</summary>
    public string ResolveModelsDir()
    {
        if (_state.Prefs.ServerModelsDir is { Length: > 0 } configured)
            return Directory.Exists(configured) ? configured : null;
        if (DiscoverRepoRoot() is not { } repo) return null;
        string candidate = Path.Combine(repo, "checkpoints");
        return Directory.Exists(candidate) ? candidate : null;
    }

    private static int PortFromUrl(string url) =>
        Uri.TryCreate((url ?? "").Trim(), UriKind.Absolute, out var uri) && uri.Port > 0 ? uri.Port : 8080;
}
