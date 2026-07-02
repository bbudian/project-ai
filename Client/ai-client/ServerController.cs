using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Text.Json;
using System.Threading.Tasks;
using Godot;

/// <summary>A server found by discovery: from the machine registry (%LOCALAPPDATA%/ProjectAI/servers) and/or a
/// port probe. <see cref="Alive"/> is false for a stale registry entry whose /health no longer answers.</summary>
public sealed record DiscoveredServer(string Url, int Port, int Pid, int Models, string Backend, string Source, bool Alive);

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

    // ---- multi-server discovery + remote stop -------------------------------------------------------------------

    private static readonly System.Net.Http.HttpClient Probe = new() { Timeout = TimeSpan.FromMilliseconds(1200) }; // Godot has its own HttpClient type — qualify

    private static string RegistryDir => Path.Combine(
        System.Environment.GetFolderPath(System.Environment.SpecialFolder.LocalApplicationData), "ProjectAI", "servers");

    /// <summary>
    /// Finds running servers: every port in the machine registry (servers self-register on startup) plus a probe
    /// of the common ports 8080-8089 and the configured URL's port, all in parallel. A registry entry whose
    /// /health doesn't answer is reported as stale rather than hidden, so leftovers from a crash are visible.
    /// Await-safe from UI handlers — Godot's SynchronizationContext resumes on the main thread.
    /// </summary>
    public async Task<IReadOnlyList<DiscoveredServer>> DiscoverAsync()
    {
        var candidates = new Dictionary<int, (int Pid, string Source)>();
        if (Directory.Exists(RegistryDir))
            foreach (string file in Directory.GetFiles(RegistryDir, "*.json"))
                try
                {
                    using var doc = JsonDocument.Parse(File.ReadAllText(file));
                    int port = doc.RootElement.GetProperty("port").GetInt32();
                    int pid = doc.RootElement.TryGetProperty("pid", out var p) ? p.GetInt32() : 0;
                    candidates[port] = (pid, "registry");
                }
                catch (Exception) { /* unreadable entry — the port scan may still find it */ }
        for (int port = 8080; port <= 8089; port++) candidates.TryAdd(port, (0, "scan"));
        if (Uri.TryCreate(_state.ServerUrl, UriKind.Absolute, out var current) && current.Port > 0)
            candidates.TryAdd(current.Port, (0, "scan"));

        var probes = candidates.Select(async kv =>
        {
            string url = $"http://localhost:{kv.Key}";
            try
            {
                string json = await Probe.GetStringAsync($"{url}/health");
                using var doc = JsonDocument.Parse(json);
                var root = doc.RootElement;
                return new DiscoveredServer(url, kv.Key,
                    root.TryGetProperty("pid", out var pid) ? pid.GetInt32() : kv.Value.Pid,
                    root.TryGetProperty("models", out var models) ? models.GetArrayLength() : 0,
                    root.TryGetProperty("defaultBackend", out var backend) ? backend.GetString() ?? "" : "",
                    kv.Value.Source, Alive: true);
            }
            catch (Exception)
            {
                // Only surface silent ports that CLAIMED to exist (registry) — dead scan ports are just noise.
                return kv.Value.Source == "registry"
                    ? new DiscoveredServer(url, kv.Key, kv.Value.Pid, 0, "", "registry — stale (not answering)", false)
                    : null;
            }
        });
        return (await Task.WhenAll(probes)).Where(s => s is not null).Select(s => s!).OrderBy(s => s.Port).ToList();
    }

    /// <summary>Asks any server to stop via PUT /shutdown (graceful: it deregisters and exits its accept loop).
    /// PUT is deliberate — the server's CORS preflight blocks cross-origin PUTs, so only real clients can do this.</summary>
    public async Task<bool> RequestShutdownAsync(string url)
    {
        try
        {
            // An explicit empty JSON body: http.sys rejects a body-less PUT with 411 before the handler runs.
            using var body = new StringContent("{}", System.Text.Encoding.UTF8, "application/json");
            using var response = await Probe.PutAsync(url.TrimEnd('/') + "/shutdown", body);
            return response.IsSuccessStatusCode;
        }
        catch (Exception) { return false; }
        finally { ReleaseIfExited(); }
    }

    /// <summary>Clears the bookkeeping for a spawned server that has since exited (e.g. stopped gracefully).</summary>
    public void ReleaseIfExited()
    {
        if (_process is not { HasExited: true }) return;
        _process.Dispose();
        _process = null;
        Starting = false;
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
