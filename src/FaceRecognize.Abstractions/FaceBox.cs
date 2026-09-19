namespace FaceRecognize.Abstractions;

/// <summary>Граничный прямоугольник лица в пиксельных координатах изображения.</summary>
public readonly record struct FaceBox(float X, float Y, float Width, float Height);