import os
from graphviz import Digraph

os.makedirs("output", exist_ok=True)

dot = Digraph(comment="HBM Fisher-KPP Architecture", format="png")
dot.attr(rankdir="TB", size="8,10", fontname="Helvetica")
dot.attr('node', fontname="Helvetica", style="filled,rounded", margin="0.2")

dot.node("DataPrep", "1. Data Preparation & Indexing\n• Extract cohort DataFrame & unique Patient IDs\n• Build contiguous Patient_Idx mapping\n• Extract longitudinal arrays (dt, baseline, volume, active treatments)", fillcolor="#E1F5FE", shape="box")
dot.node("Priors", "2. Hierarchical Priors & Non-Centered Parameters\n• Define population hyper-priors (rho, D, K, treatment alphas)\n• Implement fully non-centered offsets (rho_off, D_off, K_off, alpha_offsets)\n• Compute deterministic patient parameters via exp and scaling", fillcolor="#FFF9C4", shape="box")
dot.node("Kinetics", "3. Fisher-KPP Step Kinetics & Scan\n• Define fisher_kpp_step inner kinetic function\n• Evaluate logistic capacity & treatment damping\n• Compute wavefront velocity v_fisher = 2*sqrt(D*rho_eff)\n• Execute pytensor.scan across longitudinal timepoints", fillcolor="#FFE0B2", shape="box")
dot.node("Likelihood", "4. Deterministic Volumes & Likelihood\n• Convert predicted radii to spherical volumes\n• Formulate Normal observation likelihood over log-volumes\n• Return compiled PyMC model graph", fillcolor="#C8E6C9", shape="box")

dot.edge("DataPrep", "Priors", label=" passes cohort arrays")
dot.edge("Priors", "Kinetics", label=" maps patient parameters")
dot.edge("Kinetics", "Likelihood", label=" evaluates scan outputs")

output_path = "output/hbm_fisher_kpp_structure"
dot.render(output_path, cleanup=True)
print(f"Successfully generated HBM Fisher-KPP structure diagram at: {output_path}.png")