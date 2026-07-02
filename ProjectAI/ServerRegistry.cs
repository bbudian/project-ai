using System.Text.Json;

// Machine-local registry of running servers at %LOCALAPPDATA%/ProjectAI/servers/<port>.json — how a client finds
// "what's already running" without port-guessing. A server registers after its listener starts and removes itself
// when the accept loop ends (Ctrl+C and PUT /shutdown both flow through that path); a crash leaves a stale file,
// which clients detect by probing /health and can report as such. Best-effort by design: registration failing must
// never stop a server from serving.
internal static class ServerRegistry
{
    private static string Dir => Path.Combine(
        Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "ProjectAI", "servers");

    public static void Register(int port, string modelsDirectory, string backendId, string defaultModel)
    {
        try
        {
            Directory.CreateDirectory(Dir);
            File.WriteAllText(Path.Combine(Dir, port + ".json"), JsonSerializer.Serialize(new
            {
                port,
                pid = Environment.ProcessId,
                modelsDir = Path.GetFullPath(modelsDirectory),
                backend = backendId,
                defaultModel,
                startedUtc = DateTime.UtcNow.ToString("O"),
            }, new JsonSerializerOptions { WriteIndented = true }));
        }
        catch (Exception e) { Console.Error.WriteLine($"warning: could not register the server for discovery: {e.Message}"); }
    }

    public static void Remove(int port)
    {
        try { File.Delete(Path.Combine(Dir, port + ".json")); }
        catch (Exception) { /* best effort */ }
    }
}
