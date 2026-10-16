using System.Collections.ObjectModel;
using System.Globalization;
using System.Xml;
using System.Xml.Linq;

namespace SharpInference.Graphs.Internal;

/// <summary>Shares XML primitives, validation, and limits between logical and execution serializers.</summary>
internal static class GraphXmlSupport
{
    /// <summary>The maximum serialized string length and XML reader document character count.</summary>
    private const int MaxCharacters = 16 * 1024 * 1024;
    /// <summary>The maximum number of elements, including the root, accepted on read and write.</summary>
    private const int MaxElements = 100_000;
    /// <summary>The maximum ancestor count of any element accepted on read; the root has depth zero.</summary>
    private const int MaxDepth = 64;

    /// <summary>Wraps name/value entries in an Attributes element in dictionary enumeration order.</summary>
    /// <param name="attributes">The entries to serialize without sorting.</param>
    /// <returns>An Attributes container with one Attribute element per entry.</returns>
    internal static XElement WriteAttributes(IReadOnlyDictionary<string, string> attributes) =>
        new(
            "Attributes",
            attributes.Select(attribute => new XElement(
                "Attribute",
                new XAttribute("name", attribute.Key),
                new XAttribute("value", attribute.Value))));

    /// <summary>Reads only Attribute children into an ordinal, read-only name/value dictionary.</summary>
    /// <param name="element">The container whose children supply the attributes.</param>
    /// <returns>The materialized attribute dictionary.</returns>
    /// <exception cref="InvalidDataException">A child or attribute is unexpected, or a name or value is missing or blank.</exception>
    /// <exception cref="ArgumentException">An attribute name occurs more than once.</exception>
    internal static IReadOnlyDictionary<string, string> ReadAttributes(XElement element)
    {
        var attributes = new Dictionary<string, string>(StringComparer.Ordinal);
        foreach (var attribute in Children(element, "Attribute"))
        {
            Check(attribute, "Attribute", "name", "value");
            var name = Required(attribute, "name");
            if (!attributes.TryAdd(name, Required(attribute, "value")))
                throw new InvalidDataException($"Duplicate attribute '{name}'.");
        }
        return new ReadOnlyDictionary<string, string>(attributes);
    }

    /// <summary>Writes arithmetic and accumulator representations as attributes of a Precision element.</summary>
    /// <param name="precision">The precision requirements to serialize.</param>
    /// <returns>The precision element.</returns>
    internal static XElement WritePrecision(PrecisionRequirement precision) =>
        new(
            "Precision",
            new XAttribute("arithmetic", precision.MinimumArithmeticType),
            new XAttribute("accumulator", precision.MinimumAccumulatorType));

    /// <summary>Reads a Precision element and validates both representations as floating-point.</summary>
    /// <param name="element">The precision element to decode.</param>
    /// <returns>The independently validated arithmetic and accumulator requirements.</returns>
    /// <exception cref="InvalidDataException">The element name or attributes are invalid, or a required enum value is undefined.</exception>
    /// <exception cref="ArgumentOutOfRangeException">A defined representation is not floating-point.</exception>
    internal static PrecisionRequirement ReadPrecision(XElement element)
    {
        Check(element, "Precision", "arithmetic", "accumulator");
        return new(
            DefinitionEnumValue<GraphElementType>(element, "arithmetic"),
            DefinitionEnumValue<GraphElementType>(element, "accumulator"));
    }

    /// <summary>Writes a tensor's representation, layout, and dimensions in axis order.</summary>
    /// <param name="tensor">The descriptor to serialize.</param>
    /// <returns>A Tensor element with one Dimension child per axis.</returns>
    internal static XElement WriteTensor(TensorDescriptor tensor) =>
        new(
            "Tensor",
            new XAttribute("elementType", tensor.ElementType),
            new XAttribute("layout", tensor.Layout),
            tensor.Dimensions.Select(dimension =>
                new XElement("Dimension", new XAttribute("value", dimension))));

    /// <summary>Reads a Tensor element and constructs a descriptor with a positive, nonempty shape.</summary>
    /// <param name="element">The tensor element to decode.</param>
    /// <returns>A descriptor preserving the declared dimension order and layout text.</returns>
    /// <exception cref="InvalidDataException">The XML shape, required attributes, enum, or dimension number is invalid.</exception>
    /// <exception cref="ArgumentOutOfRangeException">The dimension list is empty or contains zero.</exception>
    internal static TensorDescriptor ReadTensor(XElement element)
    {
        Check(element, "Tensor", "elementType", "layout");
        return new(
            DefinitionEnumValue<GraphElementType>(element, "elementType"),
            Children(element, "Dimension").Select(dimension =>
            {
                Check(dimension, "Dimension", "value");
                return Number(dimension, "value");
            }),
            Required(element, "layout"));
    }

    /// <summary>Parses a required, case-sensitive enum value and rejects values not defined by the enum.</summary>
    /// <typeparam name="T">The enum representation to parse.</typeparam>
    /// <param name="element">The element containing the attribute.</param>
    /// <param name="attribute">The required attribute name.</param>
    /// <returns>A defined enum value; numeric strings are accepted when they identify a defined value.</returns>
    /// <exception cref="InvalidDataException">The attribute is missing, blank, unparsable, or undefined.</exception>
    internal static T DefinitionEnumValue<T>(XElement element, string attribute)
        where T : struct, Enum
    {
        var value = Required(element, attribute);
        if (!Enum.TryParse<T>(value, false, out var result) || !Enum.IsDefined(result))
        {
            throw new InvalidDataException($"Invalid {attribute} '{value}'.");
        }

        return result;
    }

    /// <summary>Requires exactly one direct child with the requested name without rejecting other child names.</summary>
    /// <param name="parent">The container to search.</param>
    /// <param name="name">The exact child name to select.</param>
    /// <returns>The single matching child.</returns>
    /// <exception cref="InvalidDataException">Zero or multiple children match the requested name.</exception>
    internal static XElement RequiredChild(XElement parent, string name)
    {
        var values = parent.Elements(name).ToArray();
        if (values.Length != 1)
        {
            throw new InvalidDataException(
                $"Element <{parent.Name}> requires exactly one <{name}> child.");
        }

        return values[0];
    }

    /// <summary>Validates direct children against unqualified allowed names and returns them in document order.</summary>
    /// <param name="parent">The container whose element children are validated.</param>
    /// <param name="names">The case-sensitive allowed local names.</param>
    /// <returns>The direct element children, possibly empty.</returns>
    /// <exception cref="InvalidDataException">A child has a namespace or a name outside the allowed set.</exception>
    internal static IEnumerable<XElement> DefinitionChildren(
        XElement parent,
        params string[] names)
    {
        if (parent.Elements().Any(element =>
            !names.Contains(element.Name.LocalName, StringComparer.Ordinal) ||
            element.Name.Namespace != XNamespace.None))
        {
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        }

        return parent.Elements();
    }

    /// <summary>Serializes an element without added formatting and enforces character and element-count limits.</summary>
    /// <param name="root">The root element to serialize.</param>
    /// <returns>The XML string without an added declaration.</returns>
    /// <remarks>This helper does not validate graph schema or impose the read-side depth limit.</remarks>
    /// <exception cref="InvalidDataException">The serialized length or element count exceeds its limit.</exception>
    internal static string Write(XElement root)
    {
        var xml = root.ToString(SaveOptions.DisableFormatting);
        if (xml.Length > MaxCharacters ||
            root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements)
        {
            throw new InvalidDataException("Graph XML exceeds the size limit.");
        }

        return xml;
    }

    /// <summary>Parses bounded graph XML with DTDs prohibited and validates its root kind and version.</summary>
    /// <param name="xml">The complete XML document text.</param>
    /// <param name="kind">The exact, case-sensitive graph kind required on the root.</param>
    /// <param name="version">The required root format version.</param>
    /// <returns>The Graph root after size, depth, root-name, and root-attribute validation.</returns>
    /// <remarks>Child schema is left to the logical or execution decoder. External XML resolution is disabled.</remarks>
    /// <exception cref="ArgumentNullException">The document text is null.</exception>
    /// <exception cref="InvalidDataException">The XML is malformed, exceeds a limit, or has an invalid root, kind, or version.</exception>
    internal static XElement Read(string xml, string kind, int version)
    {
        ArgumentNullException.ThrowIfNull(xml);
        if (xml.Length > MaxCharacters)
        {
            throw new InvalidDataException("Graph XML exceeds the size limit.");
        }

        try
        {
            var settings = new XmlReaderSettings
            {
                DtdProcessing = DtdProcessing.Prohibit,
                XmlResolver = null,
                MaxCharactersInDocument = MaxCharacters,
                MaxCharactersFromEntities = 0,
            };
            using var reader = XmlReader.Create(new StringReader(xml), settings);
            var document = XDocument.Load(reader, LoadOptions.None);
            var root = document.Root
                ?? throw new InvalidDataException("Graph XML has no root.");
            if (root.DescendantsAndSelf().Take(MaxElements + 1).Count() > MaxElements ||
                root.DescendantsAndSelf().Any(element =>
                    element.Ancestors().Take(MaxDepth + 1).Count() > MaxDepth))
            {
                throw new InvalidDataException(
                    "Graph XML exceeds the structural limit.");
            }

            Check(root, "Graph", "kind", "version");
            if (Required(root, "kind") != kind)
            {
                throw new InvalidDataException($"Graph XML kind must be '{kind}'.");
            }

            if (Number(root, "version") != version)
            {
                throw new InvalidDataException(
                    $"Unsupported graph XML version '{Required(root, "version")}'.");
            }

            return root;
        }
        catch (InvalidDataException)
        {
            throw;
        }
        catch (Exception exception) when (
            exception is XmlException or InvalidOperationException or FormatException)
        {
            throw new InvalidDataException("Invalid graph XML.", exception);
        }
    }

    /// <summary>Validates an unqualified element name and rejects attributes outside an unqualified whitelist.</summary>
    /// <param name="element">The element to validate.</param>
    /// <param name="name">The exact expected element name.</param>
    /// <param name="allowedAttributes">The case-sensitive allowed attribute names.</param>
    /// <remarks>Required attributes and children are validated separately.</remarks>
    /// <exception cref="InvalidDataException">The element name, namespace, or an attribute is unexpected.</exception>
    internal static void Check(
        XElement element,
        string name,
        params string[] allowedAttributes)
    {
        if (element.Name != name || element.Name.Namespace != XNamespace.None)
        {
            throw new InvalidDataException($"Expected <{name}>.");
        }

        if (element.Attributes().Any(attribute =>
            attribute.IsNamespaceDeclaration ||
            !allowedAttributes.Contains(
                attribute.Name.LocalName, StringComparer.Ordinal) ||
            attribute.Name.Namespace != XNamespace.None))
        {
            throw new InvalidDataException($"Unexpected attribute on <{name}>.");
        }
    }

    /// <summary>Requires all direct element children to have one unqualified name.</summary>
    /// <param name="parent">The container whose children are validated.</param>
    /// <param name="name">The exact name allowed for every child.</param>
    /// <returns>The matching children in document order, possibly empty.</returns>
    /// <exception cref="InvalidDataException">A child has a different name or a namespace.</exception>
    internal static IEnumerable<XElement> Children(XElement parent, string name)
    {
        if (parent.Elements().Any(element =>
            element.Name != name || element.Name.Namespace != XNamespace.None))
        {
            throw new InvalidDataException($"Unexpected children in <{parent.Name}>.");
        }

        return parent.Elements(name);
    }

    /// <summary>Reads a required nonblank attribute without trimming its value.</summary>
    /// <param name="element">The element containing the attribute.</param>
    /// <param name="attribute">The exact attribute name.</param>
    /// <returns>The original attribute text.</returns>
    /// <exception cref="InvalidDataException">The attribute is absent, empty, or whitespace.</exception>
    internal static string Required(XElement element, string attribute) =>
        string.IsNullOrWhiteSpace((string?)element.Attribute(attribute))
            ? throw new InvalidDataException(
                $"Element <{element.Name}> requires attribute '{attribute}'.")
            : (string)element.Attribute(attribute)!;

    /// <summary>Reads invariant decimal digits as a nonnegative 32-bit integer without signs or surrounding whitespace.</summary>
    /// <param name="element">The element containing the numeric attribute.</param>
    /// <param name="attribute">The required attribute name.</param>
    /// <returns>The parsed integer, including zero when declared.</returns>
    /// <exception cref="InvalidDataException">The attribute is absent, blank, not decimal digits, or outside the supported range.</exception>
    internal static int Number(XElement element, string attribute) =>
        int.TryParse(
            Required(element, attribute),
            NumberStyles.None,
            CultureInfo.InvariantCulture,
            out var value)
            ? value
            : throw new InvalidDataException(
                $"Attribute '{attribute}' on <{element.Name}> must be an integer.");

    /// <summary>Creates an attribute only when its value is non-null; empty values are retained.</summary>
    /// <param name="name">The attribute name.</param>
    /// <param name="value">The value to write, or null to omit the attribute.</param>
    /// <returns>The attribute, or null for an omitted value.</returns>
    internal static XAttribute? Optional(string name, string? value) =>
        value is null ? null : new XAttribute(name, value);
}
