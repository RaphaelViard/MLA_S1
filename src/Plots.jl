using Plots

function plot_solution(C, f, coords_clients, coords_sites, model; title="Solution")
    x_c, y_c = coords_clients
    x_s, y_s = coords_sites
    n, p = size(C)
    
    # Récupération des valeurs des variables
    # On utilise value.(variable) pour obtenir les résultats numériques
    x_val = value.(model[:x])
    y_val = value.(model[:y])
    
    plt = plot(title=title, size=(800, 600), legend=:outertopright, xlabel="X", ylabel="Y")
    
    # 1. Tracer les liaisons (traits)
    for i in 1:n
        for j in 1:p
            val = x_val[i, j]
            if val > 1e-4  # On ne trace que si la liaison existe
                # Pour le relaxé, l'opacité (alpha) dépend de la valeur fractionnaire
                plot!(plt, [x_c[i], x_s[j]], [y_c[i], y_s[j]], 
                      linecolor=:gray, 
                      linewidth=1.5 * val, 
                      alpha=val, 
                      label="")
            end
        end
    end
    
    # 2. Tracer les clients (Cercles)
    scatter!(plt, x_c, y_c, 
             markershape=:circle, 
             markercolor=:blue, 
             markersize=4, 
             label="Clients ($n)")
    
    # 3. Tracer les sites (Carrés)
    # On différencie les sites ouverts des sites fermés par la couleur/transparence
    for j in 1:p
        opacite = y_val[j]
        color = opacite > 1e-4 ? :red : :white
        
        scatter!(plt, [x_s[j]], [y_s[j]], 
                 markershape=:square, 
                 markersize=7, 
                 markercolor=color, 
                 markerstrokecolor=:red,
                 markerstrokewidth=2,
                 alpha=max(0.2, opacite), # Un minimum d'opacité pour voir les sites fermés
                 label=j == 1 ? "Sites Potentiels ($p)" : "")
    end
    
    return plt
end


function plot_raw_solution(C, coords_clients, coords_sites, x_val, y_val; title="Solution")
    x_c, y_c = coords_clients
    x_s, y_s = coords_sites
    n, p = size(C)
    
    plt = plot(title=title, size=(600, 500), legend=:none, xlabel="X", ylabel="Y")
    
    # 1. Tracer les liaisons
    for i in 1:n
        for j in 1:p
            val = x_val[i, j]
            if val > 1e-4
                plot!(plt, [x_c[i], x_s[j]], [y_c[i], y_s[j]], 
                      linecolor=:green, linewidth=1.5 * val, alpha=val)
            end
        end
    end
    
    # 2. Clients
    scatter!(plt, x_c, y_c, markershape=:circle, markercolor=:blue, markersize=3)
    
    # 3. Sites
    for j in 1:p
        if y_val[j] > 1e-4
            scatter!(plt, [x_s[j]], [y_s[j]], markershape=:square, 
                     markercolor=:green, markersize=6, markerstrokecolor=:black)
        else
            scatter!(plt, [x_s[j]], [y_s[j]], markershape=:square, 
                     markercolor=:white, markersize=4, markerstrokecolor=:red, alpha=0.3)
        end
    end
    return plt
end