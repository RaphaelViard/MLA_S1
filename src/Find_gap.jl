using JuMP, Gurobi, Plots, DelimitedFiles

# --- Paramètres de recherche ---
n_clients = 200
p_sites = 30
i = 11  
seed_depart = i*500
max_essais = 500
found = false
current_seed = seed_depart

println("🔍 Recherche d'une instance avec GAP > 0 (Relaxation < PLNE)...")

# Variables pour stocker l'instance gagnante
C_final, f_final, cl_final, st_final = nothing, nothing, nothing, nothing

while !found && current_seed < (seed_depart + max_essais)
    global C_final, f_final, cl_final, st_final, found, current_seed
    
    # 1. Génération
    C, f, clients, sites = generate_instance2(n_clients, p_sites, seed=current_seed)

    # 2. Résolution PLNE (Muet)
    m_entier = build_PLS2(C, f)
    set_silent(m_entier)
    optimize!(m_entier)
    
    # 3. Résolution Relaxée (Muet)
    m_relax = build_PLSR2(C, f)
    set_silent(m_relax)
    optimize!(m_relax)

    z_plne = objective_value(m_entier)
    z_relax = objective_value(m_relax)

    # 4. Test du Gap (seuil de 0.01 pour être sûr que ce n'est pas du bruit)
    if z_plne - z_relax > 0.01
        found = true
        C_final, f_final, cl_final, st_final = C, f, clients, sites
        println("✅ Instance trouvée à la Seed : $current_seed")
        println("   PLNE : $z_plne | Relax : $z_relax")
        println("   Gap  : $(round((z_plne-z_relax)/z_plne*100, digits=4)) %")
    else
        current_seed += 1
    end
end

if found
    # --- 1. Enregistrement des données ---
    # Sauvegarde de C et f en fichiers texte pour réutilisation future
    writedlm("Instances_with_gap/C_instance_gap$i.txt", C_final)
    writedlm("Instances_with_gap/f_instance_gap$i.txt", f_final)
    println("\n💾 Données sauvegardées dans 'C_instance_gap$i.txt' et 'f_instance_gap$i.txt'")

    # --- 2. Plot et Sauvegarde de l'image ---
    # On relance proprement pour avoir les objets de modèles à passer au plot
    m_entier = build_PLS2(C_final, f_final)
    optimize!(m_entier)
    m_relax = build_PLSR2(C_final, f_final)
    optimize!(m_relax)

    p1 = plot_solution(C_final, f_final, cl_final, st_final, m_entier, title="PLNE (Seed $current_seed)")
    p2 = plot_solution(C_final, f_final, cl_final, st_final, m_relax, title="Relaxation (Gap > 0)")
    
    final_plot = plot(p1, p2, layout=(1, 2), size=(1200, 500))
    
    # Sauvegarde du graphique en PNG
    savefig(final_plot, "Instances_with_gap/comparaison_gap$i.png")
    println("🖼️ Graphique sauvegardé sous 'Instances_with_gap/comparaison_gap$i.png'")
    
    # Affichage
    display(final_plot)
else
    println("❌ Aucune instance avec gap trouvée. Essaie d'augmenter n ou de réduire les coûts fixes f.")
end