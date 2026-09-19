using System.Diagnostics;

namespace FaceRecognize.Abstractions;

/// <summary>
/// Нейтральная векторная арифметика домена распознавания.
/// Все значения сходства/дистанции в проекте определяются для L2-нормированных
/// векторов (пайплайн распознавания нормализует эмбеддинги; контракт проверяется
/// в <see cref="AssertNormalized"/>).
/// </summary>
public static class EmbeddingMath
{
    public static float DotProduct(float[] a, float[] b)
    {
        Debug.Assert(a.Length == b.Length, "Embedding dimensions must match");
        float sum = 0f;
        for (int i = 0; i < a.Length; i++)
            sum += a[i] * b[i];
        return sum;
    }

    /// <summary>1 - косинусное сходство; корректна как дистанция только для L2-нормированных векторов.</summary>
    public static float CosineDistance(float[] a, float[] b)
    {
        return 1f - DotProduct(a, b);
    }

    public static bool IsNormalized(float[] v, float tolerance = 0.01f)
    {
        float sum = 0f;
        for (int i = 0; i < v.Length; i++)
            sum += v[i] * v[i];
        return MathF.Abs(MathF.Sqrt(sum) - 1f) <= tolerance;
    }

    public static void AssertNormalized(float[] v)
    {
        Debug.Assert(IsNormalized(v),
            "Embedding must be L2-normalized (contract of the recognition pipeline)");
    }
}