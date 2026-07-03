using FaceRecognize.Models;

namespace FaceRecognize.Services;

public class ClusteringService
{
    public static List<PersonCluster> ClusterUnknownFaces(
        List<(string imagePath, float[] embedding, byte[]? thumbnail)> unknownFaces,
        float eps = 0.42f,
        int minSamples = 2)
    {
        if (unknownFaces.Count == 0)
            return [];

        var labels = Dbscan(unknownFaces.Select(f => f.embedding).ToList(), eps, minSamples);

        var clusters = new Dictionary<int, PersonCluster>();
        var noise = new PersonCluster { ClusterId = -1 };

        for (int i = 0; i < unknownFaces.Count; i++)
        {
            var label = labels[i];
            if (label == -1)
            {
                noise.Faces.Add(new ClusterFace
                {
                    ImagePath = unknownFaces[i].imagePath,
                    Embedding = unknownFaces[i].embedding,
                    Thumbnail = unknownFaces[i].thumbnail
                });
                continue;
            }

            if (!clusters.ContainsKey(label))
                clusters[label] = new PersonCluster { ClusterId = label };

            clusters[label].Faces.Add(new ClusterFace
            {
                ImagePath = unknownFaces[i].imagePath,
                Embedding = unknownFaces[i].embedding,
                Thumbnail = unknownFaces[i].thumbnail
            });
        }

        foreach (var cluster in clusters.Values)
        {
            cluster.Centroid = ComputeCentroid(cluster.Faces.Select(f => f.Embedding).ToList());
        }

        var result = clusters.Values.ToList();
        if (noise.Faces.Count > 0)
            result.Add(noise);

        return result;
    }

    private static int[] Dbscan(List<float[]> embeddings, float eps, int minSamples)
    {
        int n = embeddings.Count;
        var labels = Enumerable.Repeat(-1, n).ToArray();
        int clusterId = 0;

        for (int i = 0; i < n; i++)
        {
            if (labels[i] != -1) continue;

            var neighbors = RangeQuery(embeddings, i, eps);
            if (neighbors.Count < minSamples) continue;

            labels[i] = clusterId;
            var seeds = new Queue<int>(neighbors.Where(j => j != i));

            while (seeds.Count > 0)
            {
                int j = seeds.Dequeue();
                if (labels[j] != -1) continue;

                labels[j] = clusterId;
                var jNeighbors = RangeQuery(embeddings, j, eps);
                if (jNeighbors.Count >= minSamples)
                {
                    foreach (var k in jNeighbors)
                    {
                        if (labels[k] == -1)
                            seeds.Enqueue(k);
                    }
                }
            }

            clusterId++;
        }

        return labels;
    }

    private static List<int> RangeQuery(List<float[]> embeddings, int queryIdx, float eps)
    {
        var result = new List<int>();
        var query = embeddings[queryIdx];

        for (int i = 0; i < embeddings.Count; i++)
        {
            var dist = FaceRecognizer.CosineDistance(query, embeddings[i]);
            if (dist <= eps)
                result.Add(i);
        }

        return result;
    }

    private static float[] ComputeCentroid(List<float[]> embeddings)
    {
        if (embeddings.Count == 0) return [];

        int dim = embeddings[0].Length;
        var centroid = new float[dim];

        foreach (var emb in embeddings)
        {
            for (int i = 0; i < dim; i++)
                centroid[i] += emb[i];
        }

        float count = embeddings.Count;
        for (int i = 0; i < dim; i++)
            centroid[i] /= count;

        return centroid;
    }
}
