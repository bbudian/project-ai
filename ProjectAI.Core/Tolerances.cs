namespace ProjectAI.Core;

/// <summary>
/// Centralized numeric tolerances and epsilons (ticket S0-7). The single home for the comparison
/// constants tests and ops share, so they aren't re-invented inline.
/// </summary>
public static class Tolerances
{
    /// <summary>Default RMSNorm epsilon (matches <c>ModelConfig.NormEpsilon</c>).</summary>
    public const float RmsNormEps = 1e-5f;
    /// <summary>Default AdamW epsilon.</summary>
    public const float AdamEps = 1e-8f;

    /// <summary>Central-difference step for finite-difference gradient checks.</summary>
    public const float FdStep = 1e-2f;
    /// <summary>Relative / absolute tolerance for gradient checks of polynomial ops.</summary>
    public const float GradRtol = 2e-2f;
    public const float GradAtol = 2e-3f;
    /// <summary>Looser tolerance for transcendental ops (softmax/silu/rmsnorm) whose central differences carry O(eps²) truncation.</summary>
    public const float LooseRtol = 1e-2f;
    public const float LooseAtol = 1e-3f;

    /// <summary>Op-vs-reference equality tolerance.</summary>
    public const float RefMatch = 1e-5f;
    /// <summary>Model-vs-reference (logits) tolerance for Stage 1.</summary>
    public const float ModelMatch = 1e-4f;
}
