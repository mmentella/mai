namespace mai.v1;

internal static class Identity
{
    private static long now = DateTime.UtcNow.Ticks;

    public static long ReadLong() => Interlocked.Read(ref now);
    public static long NextLong() => Interlocked.Increment(ref now);

    public static string Read() => ReadLong().ToString("X");
    public static string Next() => NextLong().ToString("X");
}