using System.Runtime.CompilerServices;

namespace mai.v1.tensor
{
    public static class ValidationExtensions
    {
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool LessThanOrEqualsCheck(this int[] left, int[] right) =>
            left.LessThanOrEquals(right) ? true :
                throw new ArgumentOutOfRangeException(nameof(left));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool LessThanCheck(this int[] left, int[] right) =>
            left.LessThan(right) ? true :
                throw new ArgumentOutOfRangeException(nameof(left));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool GreatherThanOrEqualsCheck(this int[] left, int[] right) =>
            left.GreatherThanOrEquals(right) ? true :
                throw new ArgumentOutOfRangeException(nameof(left));
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool GreatherThanCheck(this int[] left, int[] right) =>
            left.GreatherThanCheck(right) ? true :
                throw new ArgumentOutOfRangeException(nameof(left));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool AreEqualsCheck(this int[] left, int[] right) =>
            left.AreEquals(right) ? true :
                throw new ArgumentOutOfRangeException(nameof(left));

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static bool SameCardinalityCheck(this int[] left, int[] right) =>
            left.SameCardinality(right) ? true :
                throw new ArgumentOutOfRangeException(nameof(left));

        public static void IfNotThrow<TException>(this bool check)
            where TException : Exception, new()
        {
            if (check) { return; }
            throw new TException();
        }
    }
}
