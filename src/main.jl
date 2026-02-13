using JuMP
using Gurobi

include("Gen_instances.jl")
include("models.jl")

n = 10
C, f, x, y = generate_instance(n, seed=42)

model = build_PLS(C, f)
optimize!(model)
println("Statut de la solution : ", termination_status(model))
println("Valeur optimale : ", objective_value(model))

modelrelax = build_PLSR(C, f)
optimize!(modelrelax)
println("Statut de la solution : ", termination_status(modelrelax))
println("Valeur optimale : ", objective_value(modelrelax))