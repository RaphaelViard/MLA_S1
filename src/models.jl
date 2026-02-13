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
    
    model = Model(Gurobi.optimizer)


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