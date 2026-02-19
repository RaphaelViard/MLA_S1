"""
    Benchmark des algorithmes d'approximation pour le UFLP
    
    Compare :
    1. Heuristique d'arrondi (Heuristic1) - basée sur les variables duales
    2. Algorithme Primal-Dual de Jain-Vazirani (jain_vazirani_simple)
    
    Référence : Solution optimale via PLNE (Gurobi)
"""

using JuMP
using Gurobi
using Statistics
using Printf
using Random

# Inclure les fichiers sources
include("src/Gen_instances.jl")
include("src/models.jl")

# ============================================================================
# ALGORITHME 1 : Heuristique d'arrondi (adapté de arrondi.jl)
# ============================================================================

"""
    heuristic_arrondi(C::Matrix{Float64}, f::Vector{Float64})

Heuristique d'arrondi basée sur les variables duales de la relaxation LP.
Crée des clusters de clients et ouvre le site de coût minimal par cluster.

Retourne : (sites_ouverts, coût_total)
"""
function heuristic_arrondi(C::Matrix{Float64}, f::Vector{Float64})
    n, p = size(C)
    
    # Résoudre la relaxation
    model_relax = build_PLSR2(C, f)
    set_silent(model_relax)
    optimize!(model_relax)
    
    if termination_status(model_relax) != OPTIMAL
        error("Relaxation non résolue")
    end
    
    # Variables duales
    v = [dual(model_relax[:demande][i]) for i in 1:n]
    x_relax = value.(model_relax[:x])
    
    # Construire la solution
    assigned_clients = fill(false, n)
    y_heur = zeros(Int, p)
    x_heur = zeros(Int, n, p)
    
    while sum(assigned_clients) < n
        # Trouver le client non affecté avec v_i minimal
        ik = -1
        v_min = Inf
        for i in 1:n
            if !assigned_clients[i] && v[i] < v_min
                v_min = v[i]
                ik = i
            end
        end
        
        if ik == -1 break end
        
        # Créer le cluster
        cluster_k = [ik]
        sites_lies_a_ik = [j for j in 1:p if x_relax[ik, j] > 1e-6]
        
        for i in 1:n
            if !assigned_clients[i] && i != ik
                partage = false
                for j in sites_lies_a_ik
                    if x_relax[i, j] > 1e-6
                        partage = true
                        break
                    end
                end
                if partage
                    push!(cluster_k, i)
                end
            end
        end
        
        # Choisir le site avec coût minimal
        jk = -1
        f_min = Inf
        for j in sites_lies_a_ik
            if f[j] < f_min
                f_min = f[j]
                jk = j
            end
        end
        
        # Affectation
        if jk > 0
            y_heur[jk] = 1
            for i in cluster_k
                x_heur[i, jk] = 1
                assigned_clients[i] = true
            end
        else
            # Fallback: assigner au site le plus proche
            for i in cluster_k
                best_j = argmin([C[i, j] + f[j] for j in 1:p])
                y_heur[best_j] = 1
                x_heur[i, best_j] = 1
                assigned_clients[i] = true
            end
        end
    end
    
    # Calculer le coût
    sites_ouverts = findall(y_heur .== 1)
    cost = sum(f[j] * y_heur[j] for j in 1:p) + sum(C[i, j] * x_heur[i, j] for i in 1:n, j in 1:p)
    
    return sites_ouverts, cost
end


# ============================================================================
# ALGORITHME 2 : Primal-Dual Jain-Vazirani (copié de primdual.jl)
# ============================================================================

"""
    jain_vazirani_simple(C::Matrix, f::Vector)

Algorithme primal-dual de Jain-Vazirani pour le UFLP.
Garantie d'approximation : 3

Retourne : (sites_ouverts, assignment, coût_total, valeur_duale)
"""
function jain_vazirani_simple(C::Matrix, f::Vector; verbose::Bool=false)
    n, p = size(C)  # n clients, p sites
    
    # Créer une liste d'événements triés par temps
    events = Tuple{Float64, Symbol, Int, Int}[]
    
    for i in 1:n
        for j in 1:p
            push!(events, (C[i, j], :contribute, i, j))
        end
    end
    
    sort!(events, by = e -> e[1])
    
    # Variables
    α = zeros(Float64, n)
    connected = falses(n)
    temp_opened = falses(p)
    contribution = zeros(Float64, p)
    contributors = [Set{Int}() for _ in 1:p]
    opening_order = Int[]
    witness = zeros(Int, n)
    
    current_time = 0.0
    event_idx = 1
    
    while !all(connected) && event_idx <= length(events)
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
        Δt = next_time - current_time
        
        # Mettre à jour les contributions
        for j in 1:p
            if !temp_opened[j]
                n_contributors = count(i -> i ∈ contributors[j] && !connected[i], 1:n)
                contribution[j] += n_contributors * Δt
                
                if contribution[j] >= f[j] - 1e-9
                    temp_opened[j] = true
                    push!(opening_order, j)
                    
                    for i in contributors[j]
                        if !connected[i]
                            connected[i] = true
                            witness[i] = j
                            α[i] = current_time + (f[j] - (contribution[j] - n_contributors * Δt)) / max(n_contributors, 1)
                        end
                    end
                end
            end
        end
        
        if all(connected)
            break
        end
        
        current_time = next_time
        
        t, type, i, j = events[event_idx]
        if type == :contribute && !connected[i] && !temp_opened[j]
            push!(contributors[j], i)
        end
        
        event_idx += 1
        
        for i in 1:n
            if !connected[i]
                α[i] = current_time
            end
        end
    end
    
    for i in 1:n
        if !connected[i]
            α[i] = current_time
            for j in 1:p
                if temp_opened[j]
                    connected[i] = true
                    witness[i] = j
                    break
                end
            end
        end
    end
    
    # Phase 2 : Élagage
    final_opened = Int[]
    
    for j in opening_order
        has_conflict = false
        for j2 in final_opened
            if !isempty(intersect(contributors[j], contributors[j2]))
                has_conflict = true
                break
            end
        end
        
        if !has_conflict
            push!(final_opened, j)
        end
    end
    
    if isempty(final_opened)
        best_j = argmin(j -> f[j] + sum(C[:, j]), 1:p)
        push!(final_opened, best_j)
    end
    
    # Assignation finale
    assignment = zeros(Int, n)
    for i in 1:n
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
    
    # Calculer le coût
    total_cost = sum(f[j] for j in final_opened)
    total_cost += sum(C[i, assignment[i]] for i in 1:n)
    
    dual_value = sum(α)
    
    return final_opened, assignment, total_cost, dual_value
end


# ============================================================================
# BENCHMARK
# ============================================================================

"""
    run_benchmark(sizes::Vector{Int}; n_instances::Int=5)

Exécute le benchmark comparant les deux algorithmes.

# Arguments
- `sizes` : Vecteur des tailles d'instances (nombre de clients = nombre de sites)
- `n_instances` : Nombre d'instances par taille
"""
function run_benchmark(sizes::Vector{Int}; n_instances::Int=5)
    println("\n" * "="^90)
    println("   BENCHMARK : HEURISTIQUE D'ARRONDI vs PRIMAL-DUAL JAIN-VAZIRANI")
    println("="^90)
    println("Tailles testées : ", sizes)
    println("Instances par taille : ", n_instances)
    println("="^90)
    
    # Structure pour stocker les résultats
    results = Dict{Int, Dict{String, Vector{Float64}}}()
    
    for n in sizes
        results[n] = Dict(
            "opt_val" => Float64[],
            "opt_time" => Float64[],
            "relax_val" => Float64[],
            "arrondi_ratio" => Float64[],
            "arrondi_time" => Float64[],
            "arrondi_sites" => Float64[],
            "jv_ratio" => Float64[],
            "jv_time" => Float64[],
            "jv_sites" => Float64[],
            "gap_lp" => Float64[]
        )
    end
    
    for n in sizes
        println("\n" * "-"^70)
        @printf("Taille n = %d\n", n)
        println("-"^70)
        @printf("%-8s | %-10s | %-8s | %-12s %-8s | %-12s %-8s\n", 
                "Inst", "OPT", "Gap LP", "Arrondi", "Sites", "Jain-Vaz", "Sites")
        println("-"^70)
        
        for inst in 1:n_instances
            seed = 1000 * n + inst
            C, f, _, _ = generate_instance(n, seed=seed)
            C = Float64.(C)
            f = Float64.(f)
            
            # 1. Solution optimale (PLNE)
            t_opt = @elapsed begin
                model = build_PLS2(C, f)
                set_silent(model)
                optimize!(model)
                opt_val = objective_value(model)
            end
            
            # 2. Relaxation LP
            model_relax = build_PLSR2(C, f)
            set_silent(model_relax)
            optimize!(model_relax)
            relax_val = objective_value(model_relax)
            gap_lp = (opt_val - relax_val) / opt_val * 100
            
            # 3. Heuristique d'arrondi
            t_arrondi = @elapsed begin
                sites_arr, cost_arr = heuristic_arrondi(C, f)
            end
            ratio_arr = cost_arr / opt_val
            n_sites_arr = length(sites_arr)
            
            # 4. Primal-Dual Jain-Vazirani
            t_jv = @elapsed begin
                sites_jv, _, cost_jv, _ = jain_vazirani_simple(C, f)
            end
            ratio_jv = cost_jv / opt_val
            n_sites_jv = length(sites_jv)
            
            # Stocker les résultats
            push!(results[n]["opt_val"], opt_val)
            push!(results[n]["opt_time"], t_opt)
            push!(results[n]["relax_val"], relax_val)
            push!(results[n]["arrondi_ratio"], ratio_arr)
            push!(results[n]["arrondi_time"], t_arrondi)
            push!(results[n]["arrondi_sites"], n_sites_arr)
            push!(results[n]["jv_ratio"], ratio_jv)
            push!(results[n]["jv_time"], t_jv)
            push!(results[n]["jv_sites"], n_sites_jv)
            push!(results[n]["gap_lp"], gap_lp)
            
            @printf("%-8d | %-10.2f | %-7.2f%% | %-12.4f %-8d | %-12.4f %-8d\n",
                    inst, opt_val, gap_lp, ratio_arr, n_sites_arr, ratio_jv, n_sites_jv)
        end
    end
    
    # Afficher le résumé
    print_summary(results, sizes)
    
    return results
end


"""
    print_summary(results, sizes)

Affiche un résumé des résultats du benchmark.
"""
function print_summary(results, sizes)
    println("\n" * "="^90)
    println("   RÉSUMÉ DES RÉSULTATS")
    println("="^90)
    
    # Tableau récapitulatif
    println("\n┌─────────┬──────────────────────────────────┬──────────────────────────────────┐")
    println("│    n    │        ARRONDI (LP-based)        │      PRIMAL-DUAL (Jain-Vaz)      │")
    println("│         │  Ratio (moy±std)   Temps   Sites │  Ratio (moy±std)   Temps   Sites │")
    println("├─────────┼──────────────────────────────────┼──────────────────────────────────┤")
    
    for n in sizes
        r = results[n]
        
        arr_mean = mean(r["arrondi_ratio"])
        arr_std = std(r["arrondi_ratio"])
        arr_time = mean(r["arrondi_time"]) * 1000
        arr_sites = mean(r["arrondi_sites"])
        
        jv_mean = mean(r["jv_ratio"])
        jv_std = std(r["jv_ratio"])
        jv_time = mean(r["jv_time"]) * 1000
        jv_sites = mean(r["jv_sites"])
        
        @printf("│  %4d   │   %.3f ± %.3f   %6.1fms  %4.1f  │   %.3f ± %.3f   %6.1fms  %4.1f  │\n",
                n, arr_mean, arr_std, arr_time, arr_sites, jv_mean, jv_std, jv_time, jv_sites)
    end
    
    println("└─────────┴──────────────────────────────────┴──────────────────────────────────┘")
    
    # Statistiques globales
    println("\n" * "-"^70)
    println("STATISTIQUES GLOBALES")
    println("-"^70)
    
    all_arr = vcat([results[n]["arrondi_ratio"] for n in sizes]...)
    all_jv = vcat([results[n]["jv_ratio"] for n in sizes]...)
    all_gap = vcat([results[n]["gap_lp"] for n in sizes]...)
    
    @printf("Gap LP moyen           : %.2f%%\n\n", mean(all_gap))
    
    @printf("ARRONDI (LP-based)     : ratio moyen = %.4f, max = %.4f, min = %.4f\n",
            mean(all_arr), maximum(all_arr), minimum(all_arr))
    @printf("PRIMAL-DUAL (Jain-Vaz) : ratio moyen = %.4f, max = %.4f, min = %.4f\n",
            mean(all_jv), maximum(all_jv), minimum(all_jv))
    
    # Meilleur algorithme
    println("\n" * "-"^70)
    println("COMPARAISON DIRECTE")
    println("-"^70)
    
    n_total = length(all_arr)
    wins_arr = sum(all_arr .< all_jv)
    wins_jv = sum(all_jv .< all_arr)
    ties = sum(abs.(all_arr .- all_jv) .< 1e-6)
    
    @printf("Arrondi meilleur       : %d/%d (%.1f%%)\n", wins_arr, n_total, 100*wins_arr/n_total)
    @printf("Jain-Vazirani meilleur : %d/%d (%.1f%%)\n", wins_jv, n_total, 100*wins_jv/n_total)
    @printf("Égalité                : %d/%d (%.1f%%)\n", ties, n_total, 100*ties/n_total)
    
    # Garanties théoriques
    println("\n" * "-"^70)
    println("RAPPEL DES GARANTIES THÉORIQUES")
    println("-"^70)
    println("Arrondi (LP-based)     : dépend de l'heuristique (typiquement O(log n) ou 4)")
    println("Primal-Dual Jain-Vaz   : 3-approximation")
    println("-"^70)
end


# ============================================================================
# BENCHMARK SUR INSTANCES AVEC GAP
# ============================================================================

using DelimitedFiles

"""
    load_instance(c_file::String, f_file::String)

Charge une instance depuis des fichiers texte.
"""
function load_instance(c_file::String, f_file::String)
    C = readdlm(c_file, Float64)
    f = vec(readdlm(f_file, Float64))
    return C, f
end


"""
    run_benchmark_gap_instances()

Exécute le benchmark sur les instances pré-définies avec gap significatif.
Ces instances sont dans le dossier Instances_with_gap/
"""
function run_benchmark_gap_instances()
    println("\n" * "="^90)
    println("   BENCHMARK SUR INSTANCES AVEC GAP LP SIGNIFICATIF")
    println("="^90)
    
    # Liste des instances disponibles (10 instances)
    instances = [
        ("Instance 1", "Instances_with_gap/C_instance_gap.txt", "Instances_with_gap/f_instance_gap.txt"),
        ("Instance 2", "Instances_with_gap/C_instance_gap2.txt", "Instances_with_gap/f_instance_gap2.txt"),
        ("Instance 3", "Instances_with_gap/C_instance_gap3.txt", "Instances_with_gap/f_instance_gap3.txt"),
        ("Instance 4", "Instances_with_gap/C_instance_gap4.txt", "Instances_with_gap/f_instance_gap4.txt"),
        ("Instance 5", "Instances_with_gap/C_instance_gap5.txt", "Instances_with_gap/f_instance_gap5.txt"),
        ("Instance 6", "Instances_with_gap/C_instance_gap6.txt", "Instances_with_gap/f_instance_gap6.txt"),
        ("Instance 7", "Instances_with_gap/C_instance_gap7.txt", "Instances_with_gap/f_instance_gap7.txt"),
        ("Instance 8", "Instances_with_gap/C_instance_gap8.txt", "Instances_with_gap/f_instance_gap8.txt"),
        ("Instance 9", "Instances_with_gap/C_instance_gap9.txt", "Instances_with_gap/f_instance_gap9.txt"),
        ("Instance 10", "Instances_with_gap/C_instance_gap10.txt", "Instances_with_gap/f_instance_gap10.txt")
    ]
    
    # Vérifier que les fichiers existent
    base_path = @__DIR__
    
    results_gap = Dict{String, Dict{String, Float64}}()
    
    println("\n" * "-"^90)
    @printf("%-12s | %-6s | %-10s | %-10s | %-8s | %-12s | %-12s\n", 
            "Instance", "Taille", "OPT", "Relax LP", "Gap LP", "Arrondi", "Jain-Vaz")
    println("-"^90)
    
    for (name, c_file, f_file) in instances
        c_path = joinpath(base_path, c_file)
        f_path = joinpath(base_path, f_file)
        
        if !isfile(c_path) || !isfile(f_path)
            println("  [!] Fichiers non trouvés pour $name")
            continue
        end
        
        # Charger l'instance
        C, f = load_instance(c_path, f_path)
        n, p = size(C)
        
        # 1. Solution optimale (PLNE)
        model = build_PLS2(C, f)
        set_silent(model)
        optimize!(model)
        opt_val = objective_value(model)
        
        # 2. Relaxation LP
        model_relax = build_PLSR2(C, f)
        set_silent(model_relax)
        optimize!(model_relax)
        relax_val = objective_value(model_relax)
        gap_lp = (opt_val - relax_val) / opt_val * 100
        
        # 3. Heuristique d'arrondi
        sites_arr, cost_arr = heuristic_arrondi(C, f)
        ratio_arr = cost_arr / opt_val
        
        # 4. Primal-Dual Jain-Vazirani
        sites_jv, _, cost_jv, _ = jain_vazirani_simple(C, f)
        ratio_jv = cost_jv / opt_val
        
        # Stocker les résultats
        results_gap[name] = Dict(
            "n" => n,
            "p" => p,
            "opt_val" => opt_val,
            "relax_val" => relax_val,
            "gap_lp" => gap_lp,
            "arrondi_ratio" => ratio_arr,
            "jv_ratio" => ratio_jv
        )
        
        @printf("%-12s | %3dx%-3d | %10.2f | %10.2f | %7.2f%% | %12.4f | %12.4f\n",
                name, n, p, opt_val, relax_val, gap_lp, ratio_arr, ratio_jv)
    end
    
    println("-"^90)
    
    # Résumé
    println("\n" * "="^60)
    println("ANALYSE DES RÉSULTATS")
    println("="^60)
    
    # Calcul des moyennes
    all_arrondi_surcout = Float64[]
    all_jv_surcout = Float64[]
    all_gap_lp = Float64[]
    
    for (name, r) in results_gap
        println("\n$name ($(Int(r["n"]))x$(Int(r["p"]))) :")
        @printf("  Gap LP           : %.2f%%\n", r["gap_lp"])
        @printf("  Ratio Arrondi    : %.4f (surcoût = %.2f%%)\n", r["arrondi_ratio"], (r["arrondi_ratio"]-1)*100)
        @printf("  Ratio Jain-Vaz   : %.4f (surcoût = %.2f%%)\n", r["jv_ratio"], (r["jv_ratio"]-1)*100)
        if r["arrondi_ratio"] < r["jv_ratio"]
            println("  → Arrondi meilleur")
        elseif r["jv_ratio"] < r["arrondi_ratio"]
            println("  → Jain-Vazirani meilleur")
        else
            println("  → Égalité")
        end
        
        push!(all_arrondi_surcout, (r["arrondi_ratio"]-1)*100)
        push!(all_jv_surcout, (r["jv_ratio"]-1)*100)
        push!(all_gap_lp, r["gap_lp"])
    end
    
    # Statistiques globales
    println("\n" * "="^60)
    println("STATISTIQUES GLOBALES")
    println("="^60)
    @printf("  Gap LP moyen              : %.2f%%\n", mean(all_gap_lp))
    @printf("  Surcoût moyen Arrondi     : %.2f%%\n", mean(all_arrondi_surcout))
    @printf("  Surcoût moyen Jain-Vaz    : %.2f%%\n", mean(all_jv_surcout))
    @printf("  Surcoût max Arrondi       : %.2f%%\n", maximum(all_arrondi_surcout))
    @printf("  Surcoût max Jain-Vaz      : %.2f%%\n", maximum(all_jv_surcout))
    
    return results_gap
end


# ============================================================================
# EXÉCUTION
# ============================================================================

# Lancer le benchmark si exécuté directement
println("Démarrage du benchmark...")

# Benchmark sur instances aléatoires
# results = run_benchmark([10, 15, 20, 30], n_instances=5)

# Benchmark sur instances avec gap
results_gap = run_benchmark_gap_instances()
