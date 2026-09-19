using System.Diagnostics;
using System.IO;
using FaceRecognize.Abstractions;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

namespace FaceRecognize.Core.Services;

public class ThreeDAlignmentService : IDisposable
{
    private readonly InferenceSession _landmarkSession;

    private const int NumLandmarks = 66;
    private const int InputSize = 224;
    private const int OutputGrid = 28;
    private const int OutputChannels = 198; // 66 * 3

    private static readonly float[] Mean = { 0.485f / 0.229f, 0.456f / 0.224f, 0.406f / 0.225f };
    private static readonly float[] StdInv = { 1.0f / (0.229f * 255.0f), 1.0f / (0.224f * 255.0f), 1.0f / (0.225f * 255.0f) };

    private static readonly float[][] FaceModel3D = {
        new float[] {  0.455177f,  0.300896f, -0.764429f }, new float[] {  0.448999f,  0.166996f, -0.765143f },
        new float[] {  0.437432f,  0.022655f, -0.739267f }, new float[] {  0.415033f, -0.088941f, -0.747947f },
        new float[] {  0.389124f, -0.232380f, -0.704788f }, new float[] {  0.334630f, -0.361265f, -0.615588f },
        new float[] {  0.263725f, -0.460010f, -0.491479f }, new float[] {  0.162416f, -0.558037f, -0.339445f },
        new float[] {  0.000000f, -0.621079f, -0.287295f }, new float[] { -0.162416f, -0.558037f, -0.339445f },
        new float[] { -0.263725f, -0.460010f, -0.491479f }, new float[] { -0.334630f, -0.361265f, -0.615588f },
        new float[] { -0.389124f, -0.232380f, -0.704788f }, new float[] { -0.415033f, -0.088941f, -0.747947f },
        new float[] { -0.437432f,  0.022655f, -0.739267f }, new float[] { -0.448999f,  0.166996f, -0.765143f },
        new float[] { -0.455177f,  0.300896f, -0.764429f }, new float[] {  0.385530f,  0.402801f, -0.310031f },
        new float[] {  0.322197f,  0.464439f, -0.250558f }, new float[] {  0.254098f,  0.464204f, -0.208178f },
        new float[] {  0.186875f,  0.447061f, -0.145300f }, new float[] {  0.120881f,  0.423566f, -0.110757f },
        new float[] { -0.120881f,  0.423566f, -0.110757f }, new float[] { -0.186875f,  0.447061f, -0.145300f },
        new float[] { -0.254098f,  0.464204f, -0.208178f }, new float[] { -0.322197f,  0.464439f, -0.250558f },
        new float[] { -0.385530f,  0.402801f, -0.310031f }, new float[] {  0.000000f,  0.293333f, -0.137582f },
        new float[] {  0.000000f,  0.194829f, -0.069158f }, new float[] {  0.000000f,  0.103844f, -0.009152f },
        new float[] {  0.000000f,  0.000000f,  0.000000f }, new float[] {  0.080626f, -0.041276f, -0.134161f },
        new float[] {  0.046439f, -0.057675f, -0.102991f }, new float[] {  0.000000f, -0.068753f, -0.090545f },
        new float[] { -0.046439f, -0.057675f, -0.102991f }, new float[] { -0.080626f, -0.041276f, -0.134161f },
        new float[] {  0.315905f,  0.298338f, -0.285107f }, new float[] {  0.275252f,  0.312722f, -0.244558f },
        new float[] {  0.176395f,  0.311907f, -0.219205f }, new float[] {  0.131230f,  0.284447f, -0.234239f },
        new float[] {  0.184125f,  0.260180f, -0.226591f }, new float[] {  0.279434f,  0.267363f, -0.248441f },
        new float[] { -0.131230f,  0.284447f, -0.234239f }, new float[] { -0.176395f,  0.311907f, -0.219205f },
        new float[] { -0.275252f,  0.312722f, -0.244558f }, new float[] { -0.315905f,  0.298338f, -0.285107f },
        new float[] { -0.279434f,  0.267363f, -0.248441f }, new float[] { -0.184125f,  0.260180f, -0.226591f },
        new float[] {  0.121155f, -0.208989f, -0.160606f }, new float[] {  0.041356f, -0.194484f, -0.096160f },
        new float[] {  0.000000f, -0.205180f, -0.083299f }, new float[] { -0.041356f, -0.194484f, -0.096160f },
        new float[] { -0.121155f, -0.208989f, -0.160606f }, new float[] { -0.132325f, -0.290858f, -0.187068f },
        new float[] { -0.064138f, -0.325378f, -0.158924f }, new float[] {  0.000000f, -0.343743f, -0.113926f },
        new float[] {  0.064138f, -0.325378f, -0.158924f }, new float[] {  0.132325f, -0.290858f, -0.187068f },
        new float[] {  0.181482f, -0.243239f, -0.231285f }, new float[] {  0.084000f, -0.239718f, -0.155256f },
        new float[] {  0.000000f, -0.256058f, -0.095062f }, new float[] { -0.084000f, -0.239718f, -0.155256f },
        new float[] { -0.181482f, -0.243239f, -0.231285f }, new float[] { -0.074036f, -0.250690f, -0.177346f },
        new float[] {  0.000000f, -0.264946f, -0.112350f }, new float[] {  0.074036f, -0.250690f, -0.177346f },
        new float[] {  0.257990f,  0.276080f, -0.219999f }, new float[] { -0.257990f,  0.276080f, -0.219999f },
        new float[] {  0.257990f,  0.276080f, -0.324571f }, new float[] { -0.257990f,  0.276080f, -0.324571f }
    };

    private static readonly int[] ContourPoints = { 0, 1, 8, 15, 16, 27, 28, 29, 30, 31, 32, 33, 34, 35 };

    static ThreeDAlignmentService()
    {
        Debug.Assert(FaceModel3D.Length == NumLandmarks, "FaceModel3D must contain NumLandmarks points");
        Debug.Assert(ContourPoints.All(p => p is >= 0 and < NumLandmarks), "ContourPoints index out of landmark range");
    }

    public ThreeDAlignmentService(string onnxModelPath, SessionOptions? sessionOptions = null)
    {
        var opts = sessionOptions ?? new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
        };
        _landmarkSession = new InferenceSession(onnxModelPath, opts);
    }

    public Mat AlignFace3D(Mat source, Rect bbox)
    {
        var landmarks = DetectLandmarks(source, bbox);
        if (landmarks == null)
            return new Mat();

        var cameraMatrix = BuildCameraMatrix(source.Width, source.Height);
        var distCoeffs = new Mat();
        var rvec = new Mat();
        var tvec = new Mat();

        var modelPts = new Point3f[ContourPoints.Length];
        var imagePts = new Point2f[ContourPoints.Length];
        for (int i = 0; i < ContourPoints.Length; i++)
        {
            int idx = ContourPoints[i];
            modelPts[i] = new Point3f(
                FaceModel3D[idx][0] * 100,
                FaceModel3D[idx][1] * 100,
                FaceModel3D[idx][2] * 100);
            imagePts[i] = new Point2f(landmarks[idx, 0], landmarks[idx, 1]);
        }

        Cv2.SolvePnP(
            InputArray.Create(modelPts),
            InputArray.Create(imagePts),
            cameraMatrix, distCoeffs,
            OutputArray.Create(rvec),
            OutputArray.Create(tvec),
            false,
            SolvePnPMethod.Iterative);

        var rmat = new Mat();
        Cv2.Rodrigues(rvec, OutputArray.Create(rmat));

        var rmat3x4 = new Mat(3, 4, MatType.CV_64FC1);
        rmat.CopyTo(rmat3x4[new Rect(0, 0, 3, 3)]);
        tvec.CopyTo(rmat3x4[new Rect(3, 0, 1, 3)]);

        var warpMat = (cameraMatrix * rmat3x4).ToMat();
        var frontalized = new Mat();
        Cv2.WarpPerspective(source, frontalized, warpMat, new Size(112, 112));

        return frontalized;
    }

    public HeadPose EstimateHeadPose(Mat source, Rect bbox)
    {
        var landmarks = DetectLandmarks(source, bbox);
        if (landmarks == null)
            return new HeadPose();

        var cameraMatrix = BuildCameraMatrix(source.Width, source.Height);
        var distCoeffs = new Mat();
        var rvec = new Mat();
        var tvec = new Mat();

        var modelPts = new Point3f[ContourPoints.Length];
        var imagePts = new Point2f[ContourPoints.Length];
        for (int i = 0; i < ContourPoints.Length; i++)
        {
            int idx = ContourPoints[i];
            modelPts[i] = new Point3f(
                FaceModel3D[idx][0] * 100,
                FaceModel3D[idx][1] * 100,
                FaceModel3D[idx][2] * 100);
            imagePts[i] = new Point2f(landmarks[idx, 0], landmarks[idx, 1]);
        }

        Cv2.SolvePnP(
            InputArray.Create(modelPts),
            InputArray.Create(imagePts),
            cameraMatrix, distCoeffs,
            OutputArray.Create(rvec),
            OutputArray.Create(tvec),
            false,
            SolvePnPMethod.Iterative);

        var rmat = new Mat();
        Cv2.Rodrigues(rvec, OutputArray.Create(rmat));

        double pitch = Math.Atan2(-rmat.At<double>(2, 0),
            Math.Sqrt(rmat.At<double>(2, 1) * rmat.At<double>(2, 1) +
                      rmat.At<double>(2, 2) * rmat.At<double>(2, 2)));
        double yaw = Math.Atan2(rmat.At<double>(1, 0), rmat.At<double>(0, 0));
        double roll = Math.Atan2(rmat.At<double>(2, 1), rmat.At<double>(2, 2));

        return new HeadPose
        {
            Yaw = yaw * 180.0 / Math.PI,
            Pitch = pitch * 180.0 / Math.PI,
            Roll = roll * 180.0 / Math.PI
        };
    }

    private float[,]? DetectLandmarks(Mat source, Rect bbox)
    {
        var cropX1 = bbox.X - (int)(bbox.Width * 0.1);
        var cropY1 = bbox.Y - (int)(bbox.Height * 0.125);
        var cropX2 = bbox.X + bbox.Width + (int)(bbox.Width * 0.1);
        var cropY2 = bbox.Y + bbox.Height + (int)(bbox.Height * 0.125);

        cropX1 = Math.Max(0, cropX1);
        cropY1 = Math.Max(0, cropY1);
        cropX2 = Math.Min(source.Width, cropX2);
        cropY2 = Math.Min(source.Height, cropY2);

        var scaleX = (float)(cropX2 - cropX1) / InputSize;
        var scaleY = (float)(cropY2 - cropY1) / InputSize;

        var crop = new Mat(source, new Rect(cropX1, cropY1, cropX2 - cropX1, cropY2 - cropY1));
        var resized = new Mat();
        Cv2.Resize(crop, resized, new Size(InputSize, InputSize));

        var rgb = new Mat();
        Cv2.CvtColor(resized, rgb, ColorConversionCodes.BGR2RGB);

        var tensor = new DenseTensor<float>(new[] { 1, 3, InputSize, InputSize });
        var bytes = new byte[rgb.Rows * rgb.Cols * rgb.Channels()];
        System.Runtime.InteropServices.Marshal.Copy(rgb.Data, bytes, 0, bytes.Length);

        for (int y = 0; y < InputSize; y++)
        {
            for (int x = 0; x < InputSize; x++)
            {
                int srcIdx = (y * InputSize + x) * 3;
                tensor[0, 0, y, x] = (bytes[srcIdx + 0] / 255.0f - Mean[0]) * StdInv[0];
                tensor[0, 1, y, x] = (bytes[srcIdx + 1] / 255.0f - Mean[1]) * StdInv[1];
                tensor[0, 2, y, x] = (bytes[srcIdx + 2] / 255.0f - Mean[2]) * StdInv[2];
            }
        }

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", tensor)
        };

        using var results = _landmarkSession.Run(inputs);
        var output = results.First().AsTensor<float>().ToArray();

        var landmarks = new float[NumLandmarks, 3];
        int gridSize = OutputGrid + 1;
        Debug.Assert(output.Length == 3 * NumLandmarks * gridSize * gridSize,
            "Unexpected landmark model output size");

        for (int i = 0; i < NumLandmarks; i++)
        {
            float maxVal = float.MinValue;
            int maxIdx = 0;
            for (int j = 0; j < gridSize * gridSize; j++)
            {
                float val = output[i * gridSize * gridSize + j];
                if (val > maxVal)
                {
                    maxVal = val;
                    maxIdx = j;
                }
            }

            float conf = maxVal;
            Debug.Assert(maxVal > 0 && maxVal <= 1, "Landmark confidence outside (0, 1]");
            int gridY = maxIdx / gridSize;
            int gridX = maxIdx % gridSize;

            float offX = output[NumLandmarks * gridSize * gridSize + i * gridSize * gridSize + maxIdx];
            float offY = output[2 * NumLandmarks * gridSize * gridSize + i * gridSize * gridSize + maxIdx];

            offX = Logit(offX);
            offY = Logit(offY);

            float x = (InputSize - 1) * ((float)gridX / OutputGrid) + offX * (InputSize - 1);
            float y = (InputSize - 1) * ((float)gridY / OutputGrid) + offY * (InputSize - 1);

            landmarks[i, 0] = cropX1 + scaleX * x;
            landmarks[i, 1] = cropY1 + scaleY * y;
            landmarks[i, 2] = conf;
        }

        crop.Dispose();
        resized.Dispose();
        rgb.Dispose();

        return landmarks;
    }

    private static float Logit(float p)
    {
        p = Math.Clamp(p, 0.0000001f, 0.9999999f);
        return (float)Math.Log(p / (1 - p)) / 16.0f;
    }

    private static Mat BuildCameraMatrix(int width, int height)
    {
        var data = new double[,]
        {
            { width, 0,     width / 2.0 },
            { 0,     width, height / 2.0 },
            { 0,     0,     1 }
        };
        return Mat.FromArray(data);
    }

    public void Dispose()
    {
        _landmarkSession?.Dispose();
    }
}
