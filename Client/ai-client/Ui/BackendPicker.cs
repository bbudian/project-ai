using Godot;

// The compute-backend dropdown, shared by the chat composer and the train panel (DRY). Both need the exact same
// behavior: populate from BackendOption[], carry each backend's id as item metadata (so labels stay friendly while
// the request sends the id), disable the unavailable ones with their reason as a tooltip, and keep the current
// pick across refreshes. Centralizing it here means a change to how backends are presented — or a third place that
// needs the picker — touches one file. Callers set their own layout (size flags / min size) after constructing.
public partial class BackendPicker : OptionButton
{
    public BackendPicker()
    {
        TooltipText = "Compute backend";
    }

    /// <summary>The chosen backend's id (carried as item metadata), or "" if nothing is selected.</summary>
    public string SelectedId => Selected >= 0 ? GetItemMetadata(Selected).AsString() : "";

    /// <summary>
    /// Repopulates the picker. Keeps the current choice if it's still available, else the server default, else the
    /// first available backend.
    /// </summary>
    public void SetBackends(BackendOption[] backends, string defaultId)
    {
        string previous = SelectedId;
        Clear();
        int keep = -1, fallback = -1, firstAvailable = -1;
        for (int i = 0; i < backends.Length; i++)
        {
            var b = backends[i];
            AddItem(b.Label);
            SetItemMetadata(i, b.Id);
            SetItemDisabled(i, !b.Available);
            if (!b.Available && !string.IsNullOrEmpty(b.Reason)) SetItemTooltip(i, b.Reason);
            if (b.Available)
            {
                if (b.Id == previous) keep = i;
                if (b.Id == defaultId) fallback = i;
                if (firstAvailable < 0) firstAvailable = i;
            }
        }
        int select = Selection.FirstValid(keep, fallback, firstAvailable);
        if (select >= 0) Selected = select;
    }
}
