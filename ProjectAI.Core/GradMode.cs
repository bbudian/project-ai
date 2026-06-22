namespace ProjectAI.Core;

/// <summary>
/// Ambient on/off switch for autograd graph construction. Forward ops in <see cref="Autograd"/> only
/// record a <see cref="GradNode"/> while grad mode is enabled. Wrap evaluation or optimizer updates in
/// <see cref="NoGrad"/> so they don't build a tape. Thread-local; scopes nest and restore.
/// </summary>
public static class GradMode
{
    [ThreadStatic] private static bool _disabled;

    /// <summary>True unless inside a <see cref="NoGrad"/> scope.</summary>
    public static bool IsEnabled => !_disabled;

    /// <summary>Disables grad mode until the returned scope is disposed.</summary>
    public static IDisposable NoGrad() => new Scope();

    private sealed class Scope : IDisposable
    {
        private readonly bool _previous;
        public Scope()
        {
            _previous = _disabled;
            _disabled = true;
        }
        public void Dispose() => _disabled = _previous;
    }
}
