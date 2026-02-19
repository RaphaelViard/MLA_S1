using JuMP
using Gurobi

"""
    build_PLSR(C::Matrix, f::Vector)

Construit un modèle JuMP pour la relaxation linéaire du UFLP.
- C : Matrice des coûts de transport (n x n)
- f : Vecteur des coûts d'ouverture (n)
- Retourne : Un objet `Model` JuMP prêt à être résolu (mais non résolu).
"""
function build_PLSR(C::Matrix, f::Vector)
    n = length(f)
    
    model = Model(Gurobi.Optimizer)


    @variable(model, 0 <= y[1:n] <= 1) 
    @variable(model, 0 <= x[1:n, 1:n] <= 1)


    @objective(model, Min, 
        sum(f[j] * y[j] for j in 1:n) + 
        sum(C[i,j] * x[i,j] for i in 1:n, j in 1:n)
    )

    @constraint(model, demande[i=1:n], sum(x[i,j] for j in 1:n) == 1)
    @constraint(model, liaison[i=1:n, j=1:n], x[i,j] <= y[j])

    return model
end


function build_PLSR2(C::Matrix{Float64}, f::Vector{Float64})
    # n : nombre de clients (lignes de C)
    # p : nombre de sites potentiels (colonnes de C et taille de f)
    n, p = size(C) 
    
    model = Model(Gurobi.Optimizer)

    @variable(model, 0 <= y[j=1:p] <= 1) 
    @variable(model, 0 <= x[i=1:n, j=1:p] <= 1)

    @objective(model, Min, 
        sum(f[j] * y[j] for j in 1:p) + 
        sum(C[i,j] * x[i,j] for i in 1:n, j in 1:p)
    )

    @constraint(model, demande[i=1:n], sum(x[i,j] for j in 1:p) == 1)
    @constraint(model, liaison[i=1:n, j=1:p], x[i,j] <= y[j])

    return model
end


"""
    construire_mip_splp_gurobi(C::Matrix, f::Vector; verbose::Bool=true)

Construit le modèle de programmation linéaire en nombres entiers (MIP) 
pour le problème de localisation simple avec Gurobi.
"""
function build_PLS(C::Matrix, f::Vector; verbose::Bool=true)
    n = length(f)
    
    model = Model(Gurobi.Optimizer)

    if !verbose
        set_optimizer_attribute(model, "OutputFlag", 0)
    end
    @variable(model, y[1:n], Bin) 
    @variable(model, x[1:n, 1:n], Bin)

    # --- Fonction Objectif ---
    # Minimiser Coûts fixes + Coûts de transport
    @objective(model, Min, 
        sum(f[j] * y[j] for j in 1:n) + 
        sum(C[i,j] * x[i,j] for i in 1:n, j in 1:n)
    )

    @constraint(model, demande[i=1:n], sum(x[i,j] for j in 1:n) == 1)
    @constraint(model, liaison[i=1:n, j=1:n], x[i,j] <= y[j])

    return model
end


function build_PLS2(C::Matrix{Float64}, f::Vector{Float64})
    # n : nombre de clients (lignes de C)
    # p : nombre de sites potentiels (colonnes de C et taille de f)
    n, p = size(C) 
    
    model = Model(Gurobi.Optimizer)

    @variable(model, 0 <= y[j=1:p] <= 1, Bin) 
    @variable(model, 0 <= x[i=1:n, j=1:p] <= 1, Bin)

    @objective(model, Min, 
        sum(f[j] * y[j] for j in 1:p) + 
        sum(C[i,j] * x[i,j] for i in 1:n, j in 1:p)
    )

    @constraint(model, demande[i=1:n], sum(x[i,j] for j in 1:p) == 1)
    @constraint(model, liaison[i=1:n, j=1:p], x[i,j] <= y[j])

    return model
end

function compute_cost(C::Matrix{Float64}, f::Vector{Float64}, x::Matrix{Int}, y::Vector{Int})
    # Coût fixe total : somme des f[j] pour tous les sites où y[j] == 1
    fix_cost = sum(f[j] * y[j] for j in 1:length(f))
    
    # Coût de transport total : somme des C[i,j] pour toutes les liaisons x[i,j] == 1
    transport_cost = sum(C[i,j] * x[i,j] for i in 1:size(C,1), j in 1:size(C,2))
    
    total_cost = fix_cost + transport_cost
    
    return total_cost
end