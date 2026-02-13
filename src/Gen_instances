using Random
using LinearAlgebra

"""
    genererate_instance(n::Int; seed::Union{Int, Nothing}=nothing)

Génère une instance du problème de localisation simple (UFLP).
- n : Nombre de points (clients et sites potentiels).
- Retourne : 
    - C : Matrice n x n des coûts de transport (distances).
    - f : Vecteur de taille n des coûts d'ouverture.
    - x_coords, y_coords : Les coordonnées pour visualisation éventuelle.
"""
function generate_instance(n::Int; seed=nothing)
    # Fixer la graine aléatoire pour la reproductibilité (optionnel)
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # 1. Génération des coordonnées dans une grille 100x100
    # rand(n) génère des nombres entre 0 et 1, donc on multiplie par 100.
    x_coords = rand(n) .* 100
    y_coords = rand(n) .* 100

    # 2. Calcul de la matrice des coûts de transport C (Distances Euclidiennes)
    C = zeros(Float64, n, n)
    
    for i in 1:n        # Pour chaque client i
        for j in 1:n    # Pour chaque site potentiel j
            # Calcul de la distance euclidienne : sqrt((x1-x2)^2 + (y1-y2)^2)
            dist = sqrt((x_coords[i] - x_coords[j])^2 + (y_coords[i] - y_coords[j])^2)
            C[i, j] = dist
        end
    end

    # 3. Génération des coûts d'ouverture fixes (f)
    # On génère des coûts aléatoires. 
    # Pour que le problème soit intéressant, il faut que ces coûts soient comparables aux distances.
    # Ici, je génère des coûts entre 0 et 100, comme la taille de la grille.
    f = rand(n) .* 100

    return C, f, x_coords, y_coords
end

# --- Exemple d'utilisation ---

n = 10
C, f, x, y = generate_instance(n, seed=42)

println("Matrice des distances (C) - 3 premières lignes/colonnes :")
display(C[1:3, 1:3])

println("\nCoûts d'ouverture (f) :")
println(f)