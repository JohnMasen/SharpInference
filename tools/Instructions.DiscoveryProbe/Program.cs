using Microsoft.Extensions.DependencyInjection;
using SharpInference.Instructions;

const string implementationAssembly = "SharpInference.Instructions.Tier0.CPU";
if (AppDomain.CurrentDomain.GetAssemblies().Any(assembly => assembly.GetName().Name == implementationAssembly))
{
    throw new InvalidOperationException("The probe must start without the CPU implementation loaded.");
}
var services = new ServiceCollection();
services.AddApplicationInstructions();
using var container = services.BuildServiceProvider(new ServiceProviderOptions { ValidateOnBuild = true, ValidateScopes = true });
var catalog = container.GetRequiredService<IInstructionCatalog>();
if (!catalog.Rows.Any(row => row.Instruction == "core.add" && row.Category == InstructionCategory.Tier0) ||
    !catalog.Rows.Any(row => row.Instruction == "core.affine" && row.Category == InstructionCategory.Tier1) ||
    !AppDomain.CurrentDomain.GetAssemblies().Any(assembly => assembly.GetName().Name == implementationAssembly))
{
    throw new InvalidOperationException("Runtime dependency discovery did not install the CPU instruction sets.");
}
Console.WriteLine($"discovery-ok:{catalog.Rows.Count}");
