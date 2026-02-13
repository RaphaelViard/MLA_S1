using JuMP
using Gurobi

include("Gen_instances.jl")
include("models.jl")

println("\n" * "="^40)
println("   RÉSULTATS DU PROBLÈME DE LOCALISATION")
println("="^40)

# Modèle PLNE
model = build_PLS(C, f)
optimize!(model)
z_plne = objective_value(model)
s_plne = termination_status(model)

# Modèle Relaxé
modelrelax = build_PLSR(C, f)
optimize!(modelrelax)
z_relax = objective_value(modelrelax)
s_relax = termination_status(modelrelax)

# --- Affichage "Propre" sans printf ---

println("\n" * "-"^50)
println("RÉSULTATS DU PROBLÈME DE LOCALISATION")
println("-"^50)

println("PLNE (Entier)  | Statut: $s_plne")
println("               | Objectif: $(round(z_plne, digits=2))")

println("-"^30)

println("RELAXATION     | Statut: $s_relax")
println("               | Objectif: $(round(z_relax, digits=2))")

println("-"^50)

# Calcul du Gap
gap = (z_plne - z_relax) / z_plne * 100
println("GAP RELAXATION : $(round(gap, digits=4)) %")
println("-"^50)