namespace mai.v1;

public static class ValidationUtils
{
    public static string? Null(this string s) => string.IsNullOrWhiteSpace(s) ? null : s;
}
