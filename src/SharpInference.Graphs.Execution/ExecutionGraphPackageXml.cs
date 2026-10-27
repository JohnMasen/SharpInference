using System.Xml.Linq;

namespace SharpInference.Graphs.Execution;

/// <summary>The version-one XML entry point for execution graph packages.</summary>
public static class ExecutionGraphPackageXml
{
    private static readonly Serializer Instance = new();

    public static string Serialize(ExecutionGraphPackage package) => Instance.Serialize(package);
    public static ExecutionGraphPackage Deserialize(string xml) => Instance.Deserialize(xml);

    private sealed class Serializer : GraphPackageXmlSerializer<ExecutionGraph, ExecutionGraphPackage>
    {
        protected override string Stage => "execution";
        protected override bool IncludeMetadata(string key) => key != ExecutionGraphPackage.BackendMetadataKey;

        protected override IEnumerable<string> AdditionalChildren(string elementName) => elementName switch
        {
            "GraphPackage" => ["Target"],
            "Resource" => ["Storage"],
            "Operation" => ["Implementation", "Sources"],
            "Call" => ["Sources"],
            _ => [],
        };

        protected override void WritePackageFields(ExecutionGraphPackage package, XElement element)
        {
            if (package.GetTarget() is { } backend)
            {
                element.Add(new XElement("Target", new XAttribute("backend", backend)));
            }
        }

        protected override void ReadPackageFields(ExecutionGraphPackage package, XElement element)
        {
            if (OptionalChild(element, "Target") is not { } target)
            {
                return;
            }

            Check(target, "Target", "backend");
            DefinitionChildren(target);
            if (package.Metadata.ContainsKey(ExecutionGraphPackage.BackendMetadataKey))
            {
                throw new InvalidDataException("Duplicate execution target.");
            }

            package.SetTarget(Required(target, "backend"));
        }

        protected override void WriteResourceFields(ExecutionGraph graph, Resource resource, XElement element)
        {
            if (graph.Storage.TryGetValue(resource.Id, out var storage))
            {
                element.Add(new XElement("Storage", new XAttribute("kind", storage.Kind), Optional("slot", storage.Slot)));
            }
        }

        protected override void ReadResourceFields(ExecutionGraph graph, Resource resource, XElement element)
        {
            if (OptionalChild(element, "Storage") is not { } storage)
            {
                return;
            }

            Check(storage, "Storage", "kind", "slot");
            DefinitionChildren(storage);
            if (!graph.Storage.TryAdd(resource.Id, new(
                DefinitionEnumValue<ResourceStorageKind>(storage, "kind"), (string?)storage.Attribute("slot"))))
            {
                throw new InvalidDataException($"Duplicate storage for '{resource.Id}'.");
            }
        }

        protected override void WriteActionFields(GraphAction action, XElement element)
        {
            if (action is ExecutionOperation { SelectedImplementation: { } implementation })
            {
                element.Add(new XElement("Implementation",
                    new XAttribute("collection", implementation.CollectionId),
                    new XAttribute("instruction", implementation.InstructionName),
                    new XAttribute("target", implementation.Target),
                    new XAttribute("arithmetic", implementation.Precision.ArithmeticType),
                    new XAttribute("accumulator", implementation.Precision.AccumulatorType),
                    implementation.Dispatch is { } dispatch
                        ? new XElement("Dispatch", new XAttribute("threadsX", dispatch.ThreadsX),
                            new XAttribute("threadsY", dispatch.ThreadsY), new XAttribute("threadsZ", dispatch.ThreadsZ),
                            new XAttribute("groupsX", dispatch.GroupsX), new XAttribute("groupsY", dispatch.GroupsY),
                            new XAttribute("groupsZ", dispatch.GroupsZ)) : null));
            }

            if (Sources(action) is { Count: > 0 } sources)
            {
                element.Add(new XElement("Sources", sources.Select(source => new XElement("Source",
                    new XAttribute("document", source.Document), new XAttribute("graph", source.Graph),
                    new XAttribute("node", source.Node)))));
            }
        }

        protected override void ReadActionFields(GraphAction action, XElement element)
        {
            if (OptionalChild(element, "Implementation") is { } implementation)
            {
                if (action is not ExecutionOperation operation)
                {
                    throw new InvalidDataException("An implementation belongs to an execution operation.");
                }

                Check(implementation, "Implementation", "collection", "instruction", "target", "arithmetic", "accumulator");
                DefinitionChildren(implementation, "Dispatch");
                if (!Guid.TryParse(Required(implementation, "collection"), out var collection))
                {
                    throw new InvalidDataException("Invalid instruction collection identifier.");
                }

                operation.SelectedImplementation = new(collection, Required(implementation, "instruction"),
                    new ExecutionTarget(Required(implementation, "target")),
                    new ExecutionPrecision(DefinitionEnumValue<ElementType>(implementation, "arithmetic"),
                        DefinitionEnumValue<ElementType>(implementation, "accumulator")),
                    ReadDispatch(OptionalChild(implementation, "Dispatch")));
            }

            if (OptionalChild(element, "Sources") is { } sources)
            {
                var destination = Sources(action)
                    ?? throw new InvalidDataException("Sources belong to execution actions.");
                Check(sources, "Sources");
                foreach (var source in Children(sources, "Source"))
                {
                    Check(source, "Source", "document", "graph", "node");
                    DefinitionChildren(source);
                    destination.Add(new(Required(source, "document"), Required(source, "graph"), Required(source, "node")));
                }
            }
        }

        private static List<SourceOrigin>? Sources(GraphAction action) => action switch
        {
            ExecutionOperation operation => operation.Sources,
            ExecutionCall call => call.Sources,
            _ => null,
        };

        private static ExecutionDispatch? ReadDispatch(XElement? element)
        {
            if (element is null)
            {
                return null;
            }

            Check(element, "Dispatch", "threadsX", "threadsY", "threadsZ", "groupsX", "groupsY", "groupsZ");
            DefinitionChildren(element);
            return new((uint)Number(element, "threadsX"), (uint)Number(element, "threadsY"),
                (uint)Number(element, "threadsZ"), (uint)Number(element, "groupsX"),
                (uint)Number(element, "groupsY"), (uint)Number(element, "groupsZ"));
        }
    }
}
