using ProjectAI.Core;

namespace ProjectAI.Models;

/// <summary>Loss functions for language-model training (ticket S1-3).</summary>
public static class Loss
{
    /// <summary>
    /// Mean token-level cross-entropy. <paramref name="logits"/> is [.., vocab] and <paramref name="targets"/>
    /// holds the gold token ids (as floats) with the same leading shape. Positions whose target equals
    /// <paramref name="ignoreIndex"/> (e.g. padding) are excluded from the mean and receive no gradient.
    /// </summary>
    public static Tensor CrossEntropy(Autograd ag, Tensor logits, Tensor targets, int ignoreIndex = -100)
    {
        int vocab = logits.Shape[logits.Shape.Rank - 1];
        int n = (int)(logits.ElementCount / vocab);

        var flatLogits = logits.Shape.Rank == 2 ? logits : ag.Reshape(ag.Contiguous(logits), n, vocab);
        var ids = ToIds(ag.Backend, targets);
        if (ids.Length != n)
            throw new ArgumentException($"targets ({ids.Length}) must have one entry per logits row ({n}).");

        return ag.CrossEntropy(flatLogits, ids, ignoreIndex);
    }

    private static int[] ToIds(IComputeBackend backend, Tensor targets)
    {
        var host = new float[targets.ElementCount];
        backend.ToHost(targets, host);
        var ids = new int[host.Length];
        for (int i = 0; i < host.Length; i++) ids[i] = (int)MathF.Round(host[i]);
        return ids;
    }
}
