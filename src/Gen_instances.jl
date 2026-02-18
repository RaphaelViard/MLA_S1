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
            C[i, j] = round(dist)  # Arrondir à l'entier le plus proche
        end
    end

    # 3. Génération des coûts d'ouverture fixes (f)
    # On génère des coûts aléatoires. 
    # Pour que le problème soit intéressant, il faut que ces coûts soient comparables aux distances.
    # Ici, je génère des coûts entre 0 et 100, comme la taille de la grille.
    f = rand(n) .* 10#0

    return C, f, x_coords, y_coords
end

function generate_instance2(n::Int, p::Int; seed=nothing)
    # Fixer la graine aléatoire pour la reproductibilité
    if !isnothing(seed)
        Random.seed!(seed)
    end

    # 1. Génération des coordonnées (Grille 100x100)
    # Coordonnées des n clients
    x_clients = rand(n) .* 100
    y_clients = rand(n) .* 100

    # Coordonnées des p sites potentiels
    x_sites = rand(p) .* 100
    y_sites = rand(p) .* 100

    # 2. Calcul de la matrice des coûts C (n clients x p sites)
    # C[i, j] est le coût pour servir le client i depuis le site j
    C = zeros(Float64, n, p)
    
    for i in 1:n        # Pour chaque client
        for j in 1:p    # Pour chaque site potentiel
            dist = sqrt((x_clients[i] - x_sites[j])^2 + (y_clients[i] - y_sites[j])^2)
            C[i, j] = round(dist) 
        end
    end

    # 3. Génération des coûts d'ouverture fixes (f) pour les p sites
    # On génère p coûts, un pour chaque emplacement potentiel
    f = rand(p) .* 100

    return C, f, (x_clients, y_clients), (x_sites, y_sites)
end
