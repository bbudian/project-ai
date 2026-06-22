using ProjectAI.Core;

namespace ProjectAI.Models;

/// <summary>Selects the next token id from a 1-D logits vector over the vocabulary.</summary>
public interface ISampler
{
    int Sample(ReadOnlySpan<float> logits);
}

/// <summary>Shared decoding helpers.</summary>
internal static class Decoding
{
    /// <summary>
    /// Index of the maximum logit, skipping NaN (a NaN must never win an argmax). Ties resolve to the lowest
    /// index. An all-NaN vector has no real maximum and returns index 0 as a defined fallback.
    /// </summary>
    public static int Argmax(ReadOnlySpan<float> logits)
    {
        if (logits.Length == 0) throw new ArgumentException("logits must be non-empty.", nameof(logits));
        int best = -1;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < logits.Length; i++)
        {
            float v = logits[i];
            if (float.IsNaN(v)) continue;
            if (best < 0 || v > bestVal) { best = i; bestVal = v; }
        }
        return best < 0 ? 0 : best;
    }
}

/// <summary>Deterministic argmax decoding (NaN ignored; ties resolve to the lowest index).</summary>
public sealed class GreedySampler : ISampler
{
    public int Sample(ReadOnlySpan<float> logits) => Decoding.Argmax(logits);
}

/// <summary>
/// Temperature + top-k + top-p (nucleus) sampling, ticket S1-9. The pipeline is: scale by temperature →
/// softmax → keep the top-<see cref="TopK"/> tokens → keep the smallest set whose cumulative probability
/// reaches <see cref="TopP"/> → sample from the renormalized remainder using a seeded <see cref="IRng"/>
/// (so a fixed seed reproduces a sequence). Temperature ≤ 0 falls back to greedy (the T→0 limit). If the
/// logits are non-finite (NaN/±inf) the softmax cannot be normalized, so the sampler falls back to argmax
/// rather than silently sampling a poisoned distribution.
/// </summary>
public sealed class TopKTopPSampler(IRng rng, float temperature = 1.0f, int topK = 0, float topP = 1.0f) : ISampler
{
    public float Temperature { get; } = temperature;
    public int TopK { get; } = topK;
    public float TopP { get; } = topP;

    public int Sample(ReadOnlySpan<float> logits)
    {
        int vocab = logits.Length;
        if (vocab == 0) throw new ArgumentException("logits must be non-empty.", nameof(logits));
        if (Temperature <= 0f) return Decoding.Argmax(logits); // temperature → 0 limit is greedy

        // Numerically-stable softmax with temperature.
        float max = float.NegativeInfinity;
        for (int i = 0; i < vocab; i++) if (logits[i] > max) max = logits[i];
        var probs = new float[vocab];
        float sum = 0f;
        for (int i = 0; i < vocab; i++)
        {
            float e = MathF.Exp((logits[i] - max) / Temperature);
            probs[i] = e;
            sum += e;
        }
        // A NaN/±inf logit poisons the sum to NaN (a +inf or all-(-inf) input makes (logit-max) indeterminate),
        // so the distribution is undefined — fall back to argmax instead of normalizing into NaN and sampling the
        // wrong support. (The sum <= 0 clause is an extra guard; a finite max always gives its own term 1, so sum ≥ 1.)
        if (!float.IsFinite(sum) || sum <= 0f) return Decoding.Argmax(logits);
        for (int i = 0; i < vocab; i++) probs[i] /= sum;

        // Indices by descending probability (ties broken by index for determinism).
        var order = new int[vocab];
        for (int i = 0; i < vocab; i++) order[i] = i;
        Array.Sort(order, (a, b) =>
        {
            int c = probs[b].CompareTo(probs[a]);
            return c != 0 ? c : a.CompareTo(b);
        });

        int keep = vocab;
        if (TopK > 0 && TopK < keep) keep = TopK; // top-k filter

        if (TopP < 1f) // top-p (nucleus) filter: smallest prefix reaching TopP, at least one token
        {
            float cumulative = 0f;
            int kept = 0;
            for (int i = 0; i < keep; i++)
            {
                cumulative += probs[order[i]];
                kept++;
                if (cumulative >= TopP) break;
            }
            keep = Math.Max(1, kept);
        }

        // Sample from the kept set, renormalized.
        float total = 0f;
        for (int i = 0; i < keep; i++) total += probs[order[i]];
        float r = rng.NextFloat() * total;
        float acc = 0f;
        for (int i = 0; i < keep; i++)
        {
            acc += probs[order[i]];
            if (r < acc) return order[i];
        }
        return order[keep - 1]; // floating-point guard
    }
}
