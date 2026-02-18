using JuMP
using Gurobi
using Plots
using DelimitedFiles

include("Gen_instances.jl")
include("models.jl")
include("Plots.jl")

C = readdlm("C_instance_gap.txt")
f = vec(readdlm("f_instance_gap.txt")) # vec() pour transformer la matrice 1 col en vecteur


function Heuristic1(C::Matrix{Float64}, f::Vector{Float64})
    n, p = size(C)
    

    model_relax = build_PLSR2(C, f) 
    
    set_silent(model_relax)
    optimize!(model_relax)
    if termination_status(model_relax) != OPTIMAL
        error("Problème lors de la résolution de la relaxation.")
    end


    v = [dual(model_relax[:demande][i]) for i in 1:n]
    

    w = [dual(model_relax[:liaison][i,j]) for i in 1:n, j in 1:p]

    y_relax = value.(model_relax[:y])
    x_relax = value.(model_relax[:x])

    assigned_clients = fill(false, n)
    
    y_heur = zeros(Int, p)
    x_heur = zeros(Int, n, p)

    while sum(assigned_clients) < n
        # Trouver ik : client non affecté avec vi minimal
        ik = -1
        v_min = Inf
        for i in 1:n
            if !assigned_clients[i] && v[i] < v_min
                v_min = v[i]
                ik = i
            end
        end
        
        if ik == -1 break end # Sécurité
        
        # Créer le cluster Ck
        # ik en fait partie, ainsi que les clients non affectés partageant un site avec ik
        cluster_k = [ik]
        
        # Identifier les sites auxquels ik est lié dans la relaxation (x_relax[ik, j] > 0)
        sites_lies_a_ik = [j for j in 1:p if x_relax[ik, j] > 1e-6]
        
        for i in 1:n
            if !assigned_clients[i] && i != ik
                # Vérifier si le client i partage un site avec ik
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
        
        # 2. Choisir le site jk pour ce cluster
        # Parmi les sites liés à ik, on prend celui avec le coût d'ouverture f minimal
        jk = -1
        f_min = Inf
        for j in sites_lies_a_ik
            if f[j] < f_min
                f_min = f[j]
                jk = j
            end
        end
        
        # 3. Affectation finale pour ce cluster
        y_heur[jk] = 1
        for i in cluster_k
            x_heur[i, jk] = 1
            assigned_clients[i] = true
        end
    end
    
    return y_heur, x_heur
end

y_heur, x_heur = Heuristic1(C, f)

#println(typeof(y_heur), size(y_heur))
#println(typeof(x_heur), size(x_heur))
compute_cost(C, f, x_heur, y_heur)

