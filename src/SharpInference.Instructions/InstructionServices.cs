using System.Reflection;
using System.Runtime.Loader;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;
using Microsoft.Extensions.DependencyModel;

namespace SharpInference.Instructions;

public static class InstructionServices
{
    /// <summary>Scans explicitly trusted assemblies; constructs no devices and compiles no kernels.</summary>
    public static IServiceCollection AddInstructions(this IServiceCollection services, params Assembly[] assemblies)
    {
        ArgumentNullException.ThrowIfNull(services);
        ArgumentNullException.ThrowIfNull(assemblies);
        if (assemblies.Any(assembly => assembly is null))
        {
            throw new ArgumentException("Discovery assemblies cannot contain null.", nameof(assemblies));
        }
        services.Scan(scan => scan.FromAssemblies(assemblies.Distinct())
            .AddClasses(classes => classes.AssignableTo<IInstructionProvider>())
            .As<IInstructionProvider>()
            .WithSingletonLifetime());
        var seen = new HashSet<Type>();
        foreach (var descriptor in services.Where(descriptor =>
                     descriptor.ServiceType == typeof(IInstructionProvider) &&
                     descriptor.ImplementationType is not null).ToArray())
        {
            if (!seen.Add(descriptor.ImplementationType!))
            {
                services.Remove(descriptor);
            }
        }
        services.TryAddSingleton<InstructionRegistry>(provider =>
        {
            var registry = new InstructionRegistry(provider.GetServices<IInstructionProvider>());
            _ = registry.QueryGraphInstructionBindings();
            _ = registry.QueryOptimizationCapabilities();
            return registry;
        });
        services.TryAddSingleton<IInstructionCatalog>(provider => provider.GetRequiredService<InstructionRegistry>());
        services.TryAddSingleton<IInstructionCollectionProvider>(provider => provider.GetRequiredService<InstructionRegistry>());
        return services;
    }

    /// <summary>Uses the application's runtime dependency manifest, including not-yet-loaded managed dependencies.</summary>
    public static IServiceCollection AddApplicationInstructions(this IServiceCollection services)
    {
        var context = DependencyContext.Default ?? throw new InvalidOperationException(
            "The application runtime dependency manifest is unavailable. Use AddInstructions with explicit assemblies.");
        var assemblies = context.RuntimeLibraries.SelectMany(library => library.GetDefaultAssemblyNames(context))
            .DistinctBy(name => name.FullName)
            .OrderBy(name => name.FullName, StringComparer.Ordinal)
            .Select(AssemblyLoadContext.Default.LoadFromAssemblyName).ToArray();
        return services.AddInstructions(assemblies);
    }
}
