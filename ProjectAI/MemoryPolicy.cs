// Server-side tuning for how much memory is injected per turn — kept in one place so the /generate path and the warm
// chat session agree on the budgets. Bridge = the always-pinned map + core facts (fixed cost every turn); Recall =
// the top trusted memories matched to the current message (Stage-0 preemptive recall). Budgets are token estimates.
// Now a facade over SettingsStore (GET/PUT /config edits them live); the defaults live on MemorySettings.
internal static class MemoryPolicy
{
    public static int BridgeCards => SettingsStore.Current.Memory.BridgeCards;   // map digest entries (titles/keys only)
    public static int BridgeBudget => SettingsStore.Current.Memory.BridgeBudget; // ~tokens for the pinned bridge
    public static int RecallHits => SettingsStore.Current.Memory.RecallHits;     // top matched memories inlined per turn
    public static int RecallBudget => SettingsStore.Current.Memory.RecallBudget; // ~tokens for inlined recalled bodies
}
