using JuMP
using Gurobi
using Plots
using DelimitedFiles

include("Gen_instances.jl")
include("models.jl")
include("Plots.jl")

println("\n" * "="^40)
println("   RÉSULTATS DU PROBLÈME DE LOCALISATION")
println("="^40)


# 1. Générer les données
#C, f, clients, sites = generate_instance2(50, 50)


C = readdlm("Instances_with_gap/C_instance_gap.txt")
f = vec(readdlm("Instances_with_gap/f_instance_gap.txt")) # vec() pour transformer la matrice 1 col en vecteur

model = build_PLS2(C, f)

optimize!(model)


optimize!(model)
z_plne = objective_value(model)
s_plne = termination_status(model)

# Modèle Relaxé
modelrelax = build_PLSR2(C, f)
optimize!(modelrelax)
z_relax = objective_value(modelrelax)
s_relax = termination_status(modelrelax)


# 4. Heuristique Duale
y_h, x_h = Heuristic1(C, f)
z_heur = compute_cost(C, f, x_h, y_h)

println("\n" * "-"^50)
println("RÉSULTATS DE L'INSTANCHE")
println("-"^50)
println("PLNE (Optimal)  | Objectif : $(round(z_plne, digits=2))")
println("RELAXATION (LB) | Objectif : $(round(z_relax, digits=2))")
println("HEURISTIQUE (UB)| Objectif : $(round(z_heur, digits=2))")
println("-"^50)

# Gaps
gap_rel = (z_plne - z_relax) / z_plne * 100
gap_heur = (z_heur - z_plne) / z_plne * 100

println("GAP RELAXATION : $(round(gap_rel, digits=4)) %")
println("GAP HEURISTIQUE: $(round(gap_heur, digits=4)) % (vs Optimal)")
println("-"^50)

# --- Visualisation ---
p1 = plot_solution(C, f, clients, sites, model, title="Solution Optimale (Entière)")
p2 = plot_solution(C, f, clients, sites, modelrelax, title="Solution Relaxée (Fractionnaire)")
p3 = plot_raw_solution(C, clients, sites, x_h, y_h, title="Heuristique (Dual Cluster)")
plot(p1, p2, p3, layout=(1, 3), size=(1500, 500))