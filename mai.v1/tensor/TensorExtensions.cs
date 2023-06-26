﻿//using System.Runtime.CompilerServices;

//namespace mai.v1.Matrix
//{
//    public static class MatrixExtensions
//    {
//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static string Print(this int[] shapeOrSize) => $"({string.Join(",", shapeOrSize)})";

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static Matrix AsMatrix(this double[] data, params int[] shape) => new(data, shape);

//        public static Matrix AsMatrix(this double scalar) => new double[] { scalar }.AsMatrix();

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static Matrix AsMatrix(this double[] data, int[] shape, int[] stride) => new(shape, data, stride);

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static bool LessThanOrEquals(this int[] left, int[] right) =>
//            left.Length == right.Length && left.Zip(right, (i, s) => i <= s).All(b => b);

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static bool LessThan(this int[] left, int[] right) =>
//            left.Length == right.Length && left.Zip(right, (i, s) => i < s).All(b => b);

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static bool GreatherThanOrEquals(this int[] left, int[] right) =>
//            left.Length == right.Length && left.Zip(right, (i, s) => i >= s).All(b => b);

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static bool GreatherThan(this int[] left, int[] right) =>
//            left.Length == right.Length && left.Zip(right, (i, s) => i > s).All(b => b);

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static bool AreEquals(this int[] left, int[] right) =>
//            left.Length == right.Length && left.Zip(right, (i, s) => i == s).All(b => b);

//        [MethodImpl(MethodImplOptions.AggressiveInlining)]
//        public static bool SameCardinality(this int[] left, int[] right) =>
//            left.Aggregate((a, b) => a * b) == right.Aggregate((a, b) => a * b);
//    }
//}