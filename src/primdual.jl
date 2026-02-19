"""
    Algorithme Primal-Dual de Jain-Vazirani pour le problème de localisation simple (UFLP)
    
    Référence: Jain, K., & Vazirani, V. V. (2001). Approximation algorithms for metric 
    facility location and k-median problems using the primal-dual schema and Lagrangian relaxation.
    
    Garantie d'approximation : 3
"""

"""
    Event

Structure représentant un événement dans l'algorithme primal-dual.
- type : :contribute (client i commence à contribuer au site j) ou :open (site j s'ouvre)
- time : temps auquel l'événement se produit
- client : indice du client concerné (ou 0 si pas pertinent)
- facility : indice du site concerné
"""
struct Event
    type::Symbol      # :contribute ou :open
    time::Float64     # temps de l'événement
    client::Int       # client concerné
    facility::Int     # site concerné
end

"""
    jain_vazirani(C::Matrix, f::Vector)

Implémente l'algorithme primal-dual de Jain-Vazirani pour le UFLP.

# Arguments
- `C::Matrix` : Matrice des coûts de connexion (n_clients × n_facilities)
- `f::Vector` : Vecteur des coûts d'ouverture des sites

# Retourne
- `opened_facilities::Vector{Int}` : Indices des sites ouverts
- `assignment::Vector{Int}` : assignment[i] = j signifie que le client i est assigné au site j
- `total_cost::Float64` : Coût total de la solution
- `dual_value::Float64` : Valeur de la solution duale (borne inférieure)

# Description de l'algorithme
L'algorithme fonctionne en deux phases :

**Phase 1 : Croissance des variables duales**
- Tous les clients "non connectés" augmentent leur variable duale α_i uniformément
- Quand α_i atteint c_ij, le client i commence à contribuer au site j (β_ij croît)
- Quand la somme des contributions atteint f_j, le site j s'ouvre (temporairement)
- Les clients connectés à ce site sont gelés

**Phase 2 : Élagage**
- On sélectionne un sous-ensemble indépendant de sites pour éviter la redondance
- Garantit le ratio d'approximation de 3
"""
function jain_vazirani(C::Matrix, f::Vector; verbose::Bool=false)
    n_clients, n_facilities = size(C)
    
    # --- Variables duales ---
    α = zeros(Float64, n_clients)           # Variable duale par client
    β = zeros(Float64, n_clients, n_facilities)  # Contribution du client i au site j
    
    # --- État des clients et sites ---
    connected = falses(n_clients)           # Client connecté ?
    frozen = falses(n_clients)              # Client gelé (ne contribue plus) ?
    temp_opened = falses(n_facilities)      # Site temporairement ouvert ?
    contribution_sum = zeros(Float64, n_facilities)  # Somme des contributions par site
    
    # --- Ordre d'ouverture (pour la phase d'élagage) ---
    opening_order = Int[]
    
    # --- Ensemble des clients contribuant à chaque site ---
    contributors = [Set{Int}() for _ in 1:n_facilities]
    
    # --- Phase 1 : Croissance des variables duales ---
    if verbose
        println("\n=== PHASE 1 : Croissance des variables duales ===")
    end
    
    current_time = 0.0
    
    while !all(connected)
        # Trouver les clients actifs (non connectés et non gelés)
        active_clients = findall(.!connected .& .!frozen)
        
        if isempty(active_clients)
            # Tous les clients actifs sont gelés, on dégèle ceux non connectés
            frozen .= false
            continue
        end
        
        # Calculer le prochain événement pour chaque client actif
        next_event_time = Inf
        next_event = nothing
        
        for i in active_clients
            # Temps actuel pour ce client
            α_i = α[i]
            
            for j in 1:n_facilities
                if temp_opened[j]
                    continue  # Site déjà ouvert
                end
                
                # Temps pour commencer à contribuer au site j
                if α_i < C[i, j] && i ∉ contributors[j]
                    t_contribute = C[i, j]
                    if t_contribute < next_event_time
                        next_event_time = t_contribute
                        next_event = Event(:contribute, t_contribute, i, j)
                    end
                end
            end
        end
        
        # Vérifier quand un site s'ouvre
        for j in 1:n_facilities
            if temp_opened[j]
                continue
            end
            
            # Calculer le temps restant pour ouvrir ce site
            current_contrib = contribution_sum[j]
            remaining = f[j] - current_contrib
            
            if remaining <= 0
                # Le site devrait déjà être ouvert
                next_event_time = current_time
                next_event = Event(:open, current_time, 0, j)
                break
            end
            
            # Nombre de clients actifs contribuant à ce site
            active_contributors = [i for i in contributors[j] if !connected[i] && !frozen[i]]
            n_active = length(active_contributors)
            
            if n_active > 0
                # Temps pour que les contributions atteignent f_j
                t_open = current_time + remaining / n_active
                if t_open < next_event_time
                    next_event_time = t_open
                    next_event = Event(:open, t_open, 0, j)
                end
            end
        end
        
        if next_event === nothing
            # Pas d'événement trouvé, avancer le temps pour les clients actifs
            # vers le prochain coût de connexion
            min_cost = Inf
            for i in active_clients
                for j in 1:n_facilities
                    if !temp_opened[j] && C[i, j] > α[i]
                        min_cost = min(min_cost, C[i, j])
                    end
                end
            end
            
            if min_cost == Inf
                break
            end
            
            # Avancer α pour tous les clients actifs
            for i in active_clients
                α[i] = min_cost
            end
            current_time = min_cost
            continue
        end
        
        # Avancer le temps et mettre à jour les contributions
        Δt = next_event_time - current_time
        
        for j in 1:n_facilities
            if temp_opened[j]
                continue
            end
            active_contributors = [i for i in contributors[j] if !connected[i] && !frozen[i]]
            contribution_sum[j] += length(active_contributors) * Δt
            for i in active_contributors
                β[i, j] += Δt
            end
        end
        
        # Avancer α pour tous les clients actifs
        for i in active_clients
            α[i] = next_event_time
        end
        
        current_time = next_event_time
        
        # Traiter l'événement
        if next_event.type == :contribute
            i, j = next_event.client, next_event.facility
            push!(contributors[j], i)
            if verbose
                println("  t=$(round(current_time, digits=2)): Client $i commence à contribuer au site $j")
            end
            
        elseif next_event.type == :open
            j = next_event.facility
            temp_opened[j] = true
            push!(opening_order, j)
            
            if verbose
                println("  t=$(round(current_time, digits=2)): Site $j s'ouvre (temporairement)")
            end
            
            # Connecter tous les clients qui contribuent à ce site
            for i in contributors[j]
                if !connected[i]
                    connected[i] = true
                    if verbose
                        println("    -> Client $i connecté au site $j")
                    end
                end
            end
            
            # Geler les clients voisins (ceux qui pourraient contribuer indirectement)
            for i in 1:n_clients
                if !connected[i] && α[i] >= C[i, j]
                    frozen[i] = true
                    connected[i] = true  # Ils seront connectés à ce site ou un voisin
                    if verbose
                        println("    -> Client $i gelé (peut se connecter au site $j)")
                    end
                end
            end
        end
    end
    
    # --- Phase 2 : Élagage (Pruning) ---
    if verbose
        println("\n=== PHASE 2 : Élagage ===")
        println("Sites temporairement ouverts : ", findall(temp_opened))
    end
    
    # Construire un ensemble indépendant maximal de sites
    final_opened = Int[]
    client_covered = falses(n_clients)
    
    # Trier les sites par ordre d'ouverture
    for j in opening_order
        # Vérifier si ce site partage des contributeurs avec un site déjà sélectionné
        has_conflict = false
        for j_selected in final_opened
            if !isempty(intersect(contributors[j], contributors[j_selected]))
                has_conflict = true
                break
            end
        end
        
        if !has_conflict
            push!(final_opened, j)
            for i in contributors[j]
                client_covered[i] = true
            end
            if verbose
                println("  Site $j ajouté à la solution finale")
            end
        end
    end
    
    # --- Assignation finale des clients ---
    assignment = zeros(Int, n_clients)
    
    for i in 1:n_clients
        # Si le client contribuait à un site ouvert, l'assigner à ce site
        best_facility = 0
        best_cost = Inf
        
        for j in final_opened
            if C[i, j] < best_cost
                best_cost = C[i, j]
                best_facility = j
            end
        end
        
        # Si pas de site direct, chercher via un voisin
        if best_facility == 0
            for j in findall(temp_opened)
                if C[i, j] < best_cost
                    best_cost = C[i, j]
                    best_facility = j
                end
            end
        end
        
        assignment[i] = best_facility
    end
    
    # Pour les clients non couverts directement, les assigner au site ouvert le plus proche
    for i in 1:n_clients
        if assignment[i] == 0 || assignment[i] ∉ final_opened
            best_j = argmin([j ∈ final_opened ? C[i, j] : Inf for j in 1:n_facilities])
            assignment[i] = best_j
        end
    end
    
    # --- Calcul du coût total ---
    total_cost = sum(f[j] for j in final_opened)
    total_cost += sum(C[i, assignment[i]] for i in 1:n_clients)
    
    # Valeur duale (borne inférieure)
    dual_value = sum(α)
    
    if verbose
        println("\n=== RÉSULTAT ===")
        println("Sites ouverts : ", final_opened)
        println("Coût total : ", round(total_cost, digits=2))
        println("Valeur duale : ", round(dual_value, digits=2))
        println("Ratio d'approximation : ", round(total_cost / dual_value, digits=3))
    end
    
    return final_opened, assignment, total_cost, dual_value
end


"""
    jain_vazirani_simple(C::Matrix, f::Vector)

Version simplifiée et plus robuste de l'algorithme de Jain-Vazirani.
Utilise une approche événementielle discrète.
"""
function jain_vazirani_simple(C::Matrix, f::Vector; verbose::Bool=false)
    n = size(C, 1)  # Nombre de clients = nombre de sites
    
    # Créer une liste d'événements triés par temps
    # Événement = (temps, type, client, site)
    events = Tuple{Float64, Symbol, Int, Int}[]
    
    # Pour chaque paire (client, site), créer un événement "contribute"
    for i in 1:n
        for j in 1:n
            push!(events, (C[i, j], :contribute, i, j))
        end
    end
    
    # Trier les événements par temps
    sort!(events, by = e -> e[1])
    
    # Variables
    α = zeros(Float64, n)                    # Variable duale par client
    connected = falses(n)                    # Client connecté ?
    temp_opened = falses(n)                  # Site temporairement ouvert ?
    contribution = zeros(Float64, n)          # Contribution totale à chaque site
    contributors = [Set{Int}() for _ in 1:n]  # Clients contribuant à chaque site
    opening_order = Int[]                     # Ordre d'ouverture des sites
    witness = zeros(Int, n)                   # Site témoin pour chaque client
    
    if verbose
        println("\n=== ALGORITHME PRIMAL-DUAL DE JAIN-VAZIRANI ===")
    end
    
    current_time = 0.0
    event_idx = 1
    
    while !all(connected) && event_idx <= length(events)
        # Trouver le prochain temps d'événement pertinent
        while event_idx <= length(events)
            t, type, i, j = events[event_idx]
            
            if type == :contribute && !connected[i] && !temp_opened[j]
                break
            end
            event_idx += 1
        end
        
        if event_idx > length(events)
            break
        end
        
        next_time = events[event_idx][1]
        
        # Avant d'avancer au prochain événement, vérifier si un site s'ouvre
        Δt = next_time - current_time
        
        # Mettre à jour les contributions
        for j in 1:n
            if !temp_opened[j]
                n_contributors = count(i -> i ∈ contributors[j] && !connected[i], 1:n)
                contribution[j] += n_contributors * Δt
                
                # Vérifier si le site s'ouvre
                if contribution[j] >= f[j] - 1e-9
                    temp_opened[j] = true
                    push!(opening_order, j)
                    
                    if verbose
                        println("  t=$(round(current_time + (f[j] - (contribution[j] - n_contributors * Δt)) / max(n_contributors, 1), digits=2)): Site $j s'ouvre")
                    end
                    
                    # Connecter tous les contributeurs
                    for i in contributors[j]
                        if !connected[i]
                            connected[i] = true
                            witness[i] = j
                            α[i] = current_time + (f[j] - (contribution[j] - n_contributors * Δt)) / max(n_contributors, 1)
                            if verbose
                                println("    -> Client $i connecté")
                            end
                        end
                    end
                end
            end
        end
        
        if all(connected)
            break
        end
        
        current_time = next_time
        
        # Traiter l'événement de contribution
        t, type, i, j = events[event_idx]
        if type == :contribute && !connected[i] && !temp_opened[j]
            push!(contributors[j], i)
            if verbose
                println("  t=$(round(t, digits=2)): Client $i commence à contribuer au site $j")
            end
        end
        
        event_idx += 1
        
        # Mettre à jour α pour les clients non connectés
        for i in 1:n
            if !connected[i]
                α[i] = current_time
            end
        end
    end
    
    # Finaliser α pour les clients restants
    for i in 1:n
        if !connected[i]
            α[i] = current_time
            # Trouver le site ouvert le plus proche
            for j in 1:n
                if temp_opened[j]
                    connected[i] = true
                    witness[i] = j
                    break
                end
            end
        end
    end
    
    # --- Phase 2 : Élagage ---
    if verbose
        println("\n=== PHASE 2 : Élagage ===")
    end
    
    final_opened = Int[]
    
    for j in opening_order
        # Vérifier s'il y a un conflit avec un site déjà sélectionné
        has_conflict = false
        for j2 in final_opened
            if !isempty(intersect(contributors[j], contributors[j2]))
                has_conflict = true
                break
            end
        end
        
        if !has_conflict
            push!(final_opened, j)
            if verbose
                println("  Site $j sélectionné")
            end
        end
    end
    
    # Si aucun site n'est ouvert, ouvrir le meilleur site
    if isempty(final_opened)
        # Trouver le site avec le meilleur ratio coût/couverture
        best_j = argmin(j -> f[j] + sum(C[:, j]), 1:n)
        push!(final_opened, best_j)
    end
    
    # --- Assignation finale ---
    assignment = zeros(Int, n)
    for i in 1:n
        # Assigner au site ouvert le plus proche
        best_j = 0
        best_cost = Inf
        for j in final_opened
            if C[i, j] < best_cost
                best_cost = C[i, j]
                best_j = j
            end
        end
        assignment[i] = best_j
    end
    
    # --- Calcul du coût ---
    facility_cost = sum(f[j] for j in final_opened)
    connection_cost = sum(C[i, assignment[i]] for i in 1:n)
    total_cost = facility_cost + connection_cost
    dual_value = sum(α)
    
    if verbose
        println("\n=== RÉSULTAT FINAL ===")
        println("Sites ouverts : ", final_opened)
        println("Coût des sites : ", round(facility_cost, digits=2))
        println("Coût de connexion : ", round(connection_cost, digits=2))
        println("Coût total : ", round(total_cost, digits=2))
        println("Valeur duale : ", round(dual_value, digits=2))
        if dual_value > 0
            println("Ratio empirique : ", round(total_cost / dual_value, digits=3))
        end
    end
    
    return final_opened, assignment, total_cost, dual_value
end


"""
    evaluate_solution(C::Matrix, f::Vector, opened::Vector{Int}, assignment::Vector{Int})

Évalue le coût d'une solution pour le UFLP.
"""
function evaluate_solution(C::Matrix, f::Vector, opened::Vector{Int}, assignment::Vector{Int})
    facility_cost = sum(f[j] for j in opened)
    connection_cost = sum(C[i, assignment[i]] for i in 1:length(assignment))
    return facility_cost + connection_cost
end


"""
    test_primdual()

Fonction de test pour l'algorithme primal-dual.
"""
function test_primdual()
    println("\n" * "="^60)
    println("   TEST DE L'ALGORITHME PRIMAL-DUAL DE JAIN-VAZIRANI")
    println("="^60)
    
    # Inclure le générateur d'instances si nécessaire
    include("Gen_instances.jl")
    
    # Test sur une petite instance
    n = 10
    C, f, _, _ = generate_instance(n, seed=42)
    
    println("\nInstance générée avec $n clients/sites")
    println("Coûts d'ouverture : ", round.(f, digits=2))
    
    # Exécuter l'algorithme
    opened, assignment, cost, dual = jain_vazirani_simple(C, f, verbose=true)
    
    println("\n" * "-"^40)
    println("COMPARAISON AVEC LA SOLUTION OPTIMALE")
    println("-"^40)
    
    # Comparer avec la solution optimale (nécessite Gurobi)
    include("models.jl")
    
    model = build_PLS(C, f, verbose=false)
    set_optimizer_attribute(model, "OutputFlag", 0)
    optimize!(model)
    opt_value = objective_value(model)
    
    println("Valeur optimale (PLNE) : ", round(opt_value, digits=2))
    println("Valeur primal-dual : ", round(cost, digits=2))
    println("Borne duale : ", round(dual, digits=2))
    println("Ratio approximation : ", round(cost / opt_value, digits=3))
    println("Garantie théorique : 3.0")
    
    return opened, assignment, cost, dual
end
