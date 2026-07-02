// Server-side tuning for how much memory is injected per turn — kept in one place so the /generate path and the warm
// chat session agree on the budgets. Bridge = the always-pinned map + core facts (fixed cost every turn); Recall =
// the top trusted memories matched to the current message (Stage-0 preemptive recall). Budgets are token estimates.
internal static class MemoryPolicy
{
    public const int BridgeCards = 24;   // map digest entries (titles/keys only)
    public const int BridgeBudget = 400; // ~tokens for the pinned bridge
    public const int RecallHits = 3;     // top matched memories inlined per turn
    public const int RecallBudget = 400; // ~tokens for inlined recalled bodies
}
