using System.Globalization;
using System.Text.Json;
using System.Xml.Linq;
using SharpInference.Graphs;

namespace SharpInference;

public sealed class RwkvLegacyGraphSignatureReader : IGraphModelSignatureReader
{
    public GraphModelSignature ReadJson(JsonElement model)
    {
        string[] allowed = ["VocabularySize", "EmbeddingSize", "LayerCount", "HeadCount", "HeadSize", "StateAbiId"];
        if (model.ValueKind != JsonValueKind.Object ||
            !model.EnumerateObject().Select(property => property.Name).Order(StringComparer.Ordinal)
                .SequenceEqual(allowed.Order(StringComparer.Ordinal)))
            throw new InvalidDataException("Invalid legacy RWKV JSON signature fields.");
        int Number(string key)
        {
            var field = model.GetProperty(key);
            return field.ValueKind == JsonValueKind.Number && field.TryGetInt32(out var value) && value > 0
                ? value : throw new InvalidDataException($"Legacy RWKV dimension '{key}' must be a positive integer.");
        }
        var abi = model.GetProperty("StateAbiId");
        if (abi.ValueKind != JsonValueKind.String || string.IsNullOrWhiteSpace(abi.GetString()))
            throw new InvalidDataException("Legacy RWKV signatures require a state ABI.");
        return RwkvGraphSignatures.Create(Number("VocabularySize"), Number("EmbeddingSize"),
            Number("LayerCount"), Number("HeadCount"), Number("HeadSize"), abi.GetString()!);
    }

    public GraphModelSignature ReadXml(XElement model)
    {
        string[] allowed = ["vocabularySize", "embeddingSize", "layerCount", "headCount", "headSize", "stateAbiId"];
        if (model.Name != "Model" || model.HasElements ||
            !model.Attributes().Select(attribute => attribute.Name.ToString()).Order(StringComparer.Ordinal)
                .SequenceEqual(allowed.Order(StringComparer.Ordinal)) ||
            model.Nodes().OfType<XText>().Any(text => !string.IsNullOrWhiteSpace(text.Value)))
            throw new InvalidDataException("Invalid legacy RWKV XML signature fields.");
        int Number(string key) => int.TryParse((string?)model.Attribute(key), NumberStyles.None,
            CultureInfo.InvariantCulture, out var value) && value > 0
            ? value : throw new InvalidDataException($"Legacy RWKV dimension '{key}' must be a positive integer.");
        var abi = (string?)model.Attribute("stateAbiId");
        if (string.IsNullOrWhiteSpace(abi))
            throw new InvalidDataException("Legacy RWKV signatures require a state ABI.");
        return RwkvGraphSignatures.Create(Number("vocabularySize"), Number("embeddingSize"),
            Number("layerCount"), Number("headCount"), Number("headSize"), abi);
    }
}
