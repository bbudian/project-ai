// Shared selection rule for the OptionButton-backed pickers (model / backend / size). After repopulating, each picks
// "the previously-selected item if it survived, else a preferred default, else the first valid one." That index rule
// is the one thing they share — they differ in how they identify items and which are selectable — so it lives here
// once. Pass candidate indices in priority order; returns the first real one (>= 0), or -1 if none qualify.
public static class Selection
{
    public static int FirstValid(params int[] candidatesInPriorityOrder)
    {
        foreach (int i in candidatesInPriorityOrder) if (i >= 0) return i;
        return -1;
    }
}
