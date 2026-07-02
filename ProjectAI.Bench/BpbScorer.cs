using ProjectAI.Core;
using ProjectAI.Models;
using ProjectAI.Tokenizers;

namespace ProjectAI.Bench;

/// <summary>
/// Bits-per-byte over a held-out corpus — the primary cross-model quality signal, because normalizing the summed
/// token NLL by the UTF-8 byte count makes models with different tokenizers comparable (per-token perplexity is
/// not: a coarser tokenizer gets fewer, harder predictions). Computed with the shipped CrossEntropy forward under
/// no_grad — no backward, no optimizer.
/// </summary>
public static class BpbScorer
{
    /// <summary>
    /// Sum of token NLL (nats) over non-overlapping blocks of <paramref name="blockLength"/> tokens, divided by
    /// ln(2) × UTF-8 bytes of the corpus. The first token of each block is unscored (nothing predicts it) — the
    /// small resulting optimism is identical for every model scored on the same corpus, so rankings are unaffected.
    /// </summary>
    public static double Score(
        IComputeBackend be, LlamaModel model, ITokenizer tokenizer, ModelConfig config, string corpus,
        int blockLength = 512)
    {
        if (string.IsNullOrEmpty(corpus)) throw new ArgumentException("empty eval corpus", nameof(corpus));
        blockLength = Math.Min(blockLength, config.MaxSequenceLength);

        var ids = tokenizer.Encode(corpus);
        if (ids.Count < 2) throw new ArgumentException("eval corpus tokenizes to fewer than 2 tokens", nameof(corpus));

        var ag = new Autograd(be);
        double totalNll = 0;
        long scoredTokens = 0;

        using (GradMode.NoGrad())
        {
            for (int start = 0; start + 1 < ids.Count; start += blockLength)
            {
                int n = Math.Min(blockLength, ids.Count - start - 1); // inputs [start..start+n), targets shifted by 1
                var inputs = new float[n];
                var targets = new float[n];
                for (int i = 0; i < n; i++)
                {
                    inputs[i] = ids[start + i];
                    targets[i] = ids[start + i + 1];
                }

                using (be.BeginScope()) // free this block's activations before the next one allocates
                {
                    var logits = model.Forward(be.FromHost(inputs, new Shape(1, n), DType.F32), ForwardContext.Inference());
                    var loss = Loss.CrossEntropy(ag, logits, be.FromHost(targets, new Shape(1, n), DType.F32));
                    var host = new float[1];
                    be.ToHost(loss, host);
                    totalNll += host[0] * n; // CrossEntropy is the MEAN over the block's n predictions
                    scoredTokens += n;
                }
            }
        }

        long bytes = System.Text.Encoding.UTF8.GetByteCount(corpus);
        if (scoredTokens == 0 || bytes == 0) throw new InvalidOperationException("nothing was scored");
        return totalNll / (Math.Log(2) * bytes);
    }
}
