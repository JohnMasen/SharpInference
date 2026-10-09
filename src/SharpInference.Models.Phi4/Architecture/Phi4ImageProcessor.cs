namespace SharpInference.Architectures.Phi4;

public enum Phi4PixelFormat
{
    Rgb24,
    Bgr24,
    Rgba32,
    Bgra32,
}

public sealed record Phi4ImageFeatures(
    float[] PixelValues,
    float[] AttentionMask,
    int CropCount,
    int TargetHeight,
    int TargetWidth,
    int ImageTokenCount)
{
    public const int CropSize = 448;
    public const int MaskSize = 32;
}

public sealed class Phi4ImageProcessor(int maximumTiles = 36)
{
    private const int CropSize = Phi4ImageFeatures.CropSize;
    private const int MaskSize = Phi4ImageFeatures.MaskSize;

    public int MaximumTiles { get; } = maximumTiles is > 0 and <= 36
        ? maximumTiles
        : throw new ArgumentOutOfRangeException(nameof(maximumTiles));

    public Phi4ImageFeatures Process(
        ReadOnlySpan<byte> pixels,
        int width,
        int height,
        int stride,
        Phi4PixelFormat format)
    {
        if (width <= 0 || height <= 0)
            throw new ArgumentOutOfRangeException(nameof(width));
        var bytesPerPixel = format is Phi4PixelFormat.Rgb24 or Phi4PixelFormat.Bgr24 ? 3 : 4;
        if (stride < checked(width * bytesPerPixel) || pixels.Length < checked(stride * height))
            throw new ArgumentException("Pixel buffer is smaller than its dimensions and stride.", nameof(pixels));

        var (horizontalTiles, verticalTiles) = SelectTileGrid(width, height);
        var targetWidth = checked(horizontalTiles * CropSize);
        var targetHeight = checked(verticalTiles * CropSize);
        var widthRatio = targetWidth / (double)width;
        var heightRatio = targetHeight / (double)height;
        int resizedWidth;
        int resizedHeight;
        if (widthRatio < heightRatio)
        {
            resizedWidth = targetWidth;
            resizedHeight = (int)(height * widthRatio);
        }
        else
        {
            resizedWidth = (int)(width * heightRatio);
            resizedHeight = targetHeight;
        }
        if (resizedWidth < 10 || resizedHeight < 10)
            throw new ArgumentException("The image aspect ratio is too extreme.", nameof(pixels));

        var normalized = new float[checked(3 * targetWidth * targetHeight)];
        normalized.AsSpan().Fill(1f);
        ResizeToPlanar(
            pixels, width, height, stride, format,
            normalized, targetWidth, targetHeight, resizedWidth, resizedHeight);

        var tileCount = checked(horizontalTiles * verticalTiles);
        var cropCount = checked(tileCount + 1);
        var output = new float[checked(cropCount * 3 * CropSize * CropSize)];
        ResizeBicubic(normalized, targetWidth, targetHeight, output.AsSpan(0, 3 * CropSize * CropSize));
        CopyTiles(normalized, targetWidth, horizontalTiles, verticalTiles, output);

        var mask = CreateMask(
            horizontalTiles, verticalTiles,
            targetWidth - resizedWidth, targetHeight - resizedHeight);
        var tokenCount = CalculateTokenCount(mask, horizontalTiles, verticalTiles);
        var outputMask = new float[checked(cropCount * MaskSize * MaskSize)];
        outputMask.AsSpan(0, MaskSize * MaskSize).Fill(1f);
        CopyMaskTiles(mask, horizontalTiles, verticalTiles, outputMask);

        return new Phi4ImageFeatures(
            output, outputMask, cropCount, targetHeight, targetWidth, tokenCount);
    }

    private (int Width, int Height) SelectTileGrid(int width, int height)
    {
        var horizontal = DivideRoundUp(width, CropSize);
        var vertical = DivideRoundUp(height, CropSize);
        if (horizontal * vertical <= MaximumTiles)
            return (horizontal, vertical);

        var aspect = width / (double)height;
        var area = (long)width * height;
        var bestDifference = double.PositiveInfinity;
        var best = (Width: 1, Height: 1);
        for (var count = 1; count <= MaximumTiles; count++)
        for (var candidateWidth = 1; candidateWidth <= count; candidateWidth++)
        for (var candidateHeight = 1; candidateHeight <= count; candidateHeight++)
        {
            var tiles = candidateWidth * candidateHeight;
            if (tiles > MaximumTiles || tiles < 1)
                continue;
            var difference = Math.Abs(aspect - candidateWidth / (double)candidateHeight);
            if (difference < bestDifference ||
                difference == bestDifference &&
                area > (long)CropSize * CropSize * candidateWidth * candidateHeight / 2)
            {
                bestDifference = difference;
                best = (candidateWidth, candidateHeight);
            }
        }
        return best;
    }

    private static void ResizeToPlanar(
        ReadOnlySpan<byte> source,
        int sourceWidth,
        int sourceHeight,
        int sourceStride,
        Phi4PixelFormat format,
        Span<float> target,
        int targetWidth,
        int targetHeight,
        int resizedWidth,
        int resizedHeight)
    {
        var bytesPerPixel = format is Phi4PixelFormat.Rgb24 or Phi4PixelFormat.Bgr24 ? 3 : 4;
        var redOffset = format is Phi4PixelFormat.Rgb24 or Phi4PixelFormat.Rgba32 ? 0 : 2;
        var blueOffset = 2 - redOffset;
        var plane = targetWidth * targetHeight;
        for (var y = 0; y < resizedHeight; y++)
        {
            var sourceY = Math.Max(0, Math.Min(sourceHeight - 1, (y + 0.5) * sourceHeight / resizedHeight - 0.5));
            var y0 = (int)Math.Floor(sourceY);
            var y1 = Math.Min(sourceHeight - 1, y0 + 1);
            var fy = (float)(sourceY - y0);
            for (var x = 0; x < resizedWidth; x++)
            {
                var sourceX = Math.Max(0, Math.Min(sourceWidth - 1, (x + 0.5) * sourceWidth / resizedWidth - 0.5));
                var x0 = (int)Math.Floor(sourceX);
                var x1 = Math.Min(sourceWidth - 1, x0 + 1);
                var fx = (float)(sourceX - x0);
                var outputIndex = y * targetWidth + x;
                target[outputIndex] = Normalize(BilinearSample(
                    source, sourceStride, bytesPerPixel, redOffset, x0, x1, y0, y1, fx, fy));
                target[plane + outputIndex] = Normalize(BilinearSample(
                    source, sourceStride, bytesPerPixel, 1, x0, x1, y0, y1, fx, fy));
                target[2 * plane + outputIndex] = Normalize(BilinearSample(
                    source, sourceStride, bytesPerPixel, blueOffset, x0, x1, y0, y1, fx, fy));
            }
        }
        static float Normalize(float value) => value * (2f / 255f) - 1f;
    }

    private static float BilinearSample(
        ReadOnlySpan<byte> source,
        int stride,
        int bytesPerPixel,
        int channel,
        int x0,
        int x1,
        int y0,
        int y1,
        float fx,
        float fy)
    {
        var topLeft = source[y0 * stride + x0 * bytesPerPixel + channel];
        var topRight = source[y0 * stride + x1 * bytesPerPixel + channel];
        var bottomLeft = source[y1 * stride + x0 * bytesPerPixel + channel];
        var bottomRight = source[y1 * stride + x1 * bytesPerPixel + channel];
        var top = topLeft + (topRight - topLeft) * fx;
        var bottom = bottomLeft + (bottomRight - bottomLeft) * fx;
        return top + (bottom - top) * fy;
    }

    private static void ResizeBicubic(
        ReadOnlySpan<float> source,
        int sourceWidth,
        int sourceHeight,
        Span<float> target)
    {
        var sourcePlane = sourceWidth * sourceHeight;
        var targetPlane = CropSize * CropSize;
        for (var channel = 0; channel < 3; channel++)
        for (var y = 0; y < CropSize; y++)
        {
            var sourceY = (y + 0.5) * sourceHeight / CropSize - 0.5;
            var yBase = (int)Math.Floor(sourceY);
            for (var x = 0; x < CropSize; x++)
            {
                var sourceX = (x + 0.5) * sourceWidth / CropSize - 0.5;
                var xBase = (int)Math.Floor(sourceX);
                var value = 0f;
                var weightSum = 0f;
                for (var dy = -1; dy <= 2; dy++)
                {
                    var sy = Math.Clamp(yBase + dy, 0, sourceHeight - 1);
                    var wy = Cubic((float)(sourceY - (yBase + dy)));
                    for (var dx = -1; dx <= 2; dx++)
                    {
                        var sx = Math.Clamp(xBase + dx, 0, sourceWidth - 1);
                        var weight = wy * Cubic((float)(sourceX - (xBase + dx)));
                        value += source[channel * sourcePlane + sy * sourceWidth + sx] * weight;
                        weightSum += weight;
                    }
                }
                target[channel * targetPlane + y * CropSize + x] = value / weightSum;
            }
        }

        static float Cubic(float value)
        {
            const float a = -0.75f;
            value = MathF.Abs(value);
            if (value <= 1)
                return (a + 2) * value * value * value - (a + 3) * value * value + 1;
            return value < 2
                ? a * value * value * value - 5 * a * value * value + 8 * a * value - 4 * a
                : 0;
        }
    }

    private static void CopyTiles(
        ReadOnlySpan<float> source,
        int sourceWidth,
        int horizontalTiles,
        int verticalTiles,
        Span<float> target)
    {
        var sourceHeight = verticalTiles * CropSize;
        var sourcePlane = sourceWidth * sourceHeight;
        var cropPlane = CropSize * CropSize;
        var cropStride = 3 * cropPlane;
        var crop = 1;
        for (var tileY = 0; tileY < verticalTiles; tileY++)
        for (var tileX = 0; tileX < horizontalTiles; tileX++, crop++)
        for (var channel = 0; channel < 3; channel++)
        for (var y = 0; y < CropSize; y++)
        {
            var sourceOffset = channel * sourcePlane +
                (tileY * CropSize + y) * sourceWidth + tileX * CropSize;
            var targetOffset = crop * cropStride + channel * cropPlane + y * CropSize;
            source.Slice(sourceOffset, CropSize).CopyTo(target[targetOffset..]);
        }
    }

    private static float[] CreateMask(
        int horizontalTiles,
        int verticalTiles,
        int paddingWidth,
        int paddingHeight)
    {
        var width = horizontalTiles * MaskSize;
        var height = verticalTiles * MaskSize;
        var result = new float[width * height];
        result.AsSpan().Fill(1f);
        var maskedColumns = paddingWidth >= 14 ? paddingWidth / 14 : 0;
        var maskedRows = paddingHeight >= 14 ? paddingHeight / 14 : 0;
        if (maskedColumns > 0)
            for (var y = 0; y < height; y++)
                result.AsSpan(y * width + width - maskedColumns, maskedColumns).Clear();
        if (maskedRows > 0)
            result.AsSpan((height - maskedRows) * width).Clear();
        return result;
    }

    private static int CalculateTokenCount(
        ReadOnlySpan<float> mask,
        int horizontalTiles,
        int verticalTiles)
    {
        var maskWidth = horizontalTiles * MaskSize;
        var sum = 0;
        var firstColumn = 0;
        for (var tileY = 0; tileY < verticalTiles; tileY++)
        for (var localY = 0; localY < MaskSize; localY += 2)
        for (var tileX = 0; tileX < horizontalTiles; tileX++)
        for (var localX = 0; localX < MaskSize; localX += 2)
        {
            var value = mask[(tileY * MaskSize + localY) * maskWidth + tileX * MaskSize + localX] != 0;
            if (value) sum++;
            if (tileX == 0 && localX == 0 && value) firstColumn++;
        }
        return checked(256 + 1 + sum + firstColumn + 16);
    }

    private static void CopyMaskTiles(
        ReadOnlySpan<float> source,
        int horizontalTiles,
        int verticalTiles,
        Span<float> target)
    {
        var sourceWidth = horizontalTiles * MaskSize;
        var cropStride = MaskSize * MaskSize;
        var crop = 1;
        for (var tileY = 0; tileY < verticalTiles; tileY++)
        for (var tileX = 0; tileX < horizontalTiles; tileX++, crop++)
        for (var y = 0; y < MaskSize; y++)
        {
            var sourceOffset = (tileY * MaskSize + y) * sourceWidth + tileX * MaskSize;
            source.Slice(sourceOffset, MaskSize)
                .CopyTo(target[(crop * cropStride + y * MaskSize)..]);
        }
    }

    private static int DivideRoundUp(int value, int divisor) =>
        checked((value + divisor - 1) / divisor);
}
