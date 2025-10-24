import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- Chargement des données (décommentez si nécessaire) ---
# Assurez-vous que les fichiers CSV sont dans le dossier 'data/' si utilisés
# matches = pd.read_csv("data/matches_1930_2022.csv")
# international = pd.read_csv("data/international_matches.csv")
# fifa = pd.read_csv("data/fifa_ranking_2022-10-06.csv")
# worldcup = pd.read_csv("data/world_cup.csv")
# groups2022 = pd.read_csv("data/2022_world_cup_groups.csv")
# squads2022 = pd.read_csv("data/2022_world_cup_squads.csv", encoding="latin1")
# teamsinfo = pd.read_csv("data/world_cups_team_infos.csv")

# --- Configuration Pandas ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# --- Filtrage des matchs (optionnel, décommentez si nécessaire) ---
# matches["Date"] = pd.to_datetime(matches["Date"])
# matches = matches[matches["Date"] < "2022-11-20"]

# --- Fonction pour générer le bracket ---
def draw_bracket(final_winner):
    # Résultats hardcoded du notebook
    group_stage_qualified = [
        'senegal', 'netherlands', 'england', 'iran',
        'argentina', 'poland', 'france', 'denmark',
        'spain', 'germany', 'belgium', 'croatia',
        'brazil', 'serbia', 'uruguay', 'south korea'
    ]
    round16_winners = ['iran', 'netherlands', 'denmark', 'france', 'croatia', 'germany', 'south korea', 'serbia']
    quarter_winners = ['netherlands', 'france', 'germany', 'serbia']
    semi_winners = ['france', 'serbia']
    # final_winner est passé en paramètre pour permettre la sélection dynamique

    def make_pairs(team_list):
        return [(team_list[i], team_list[i+1]) for i in range(0, len(team_list), 2)]

    def split_half(lst):
        mid = len(lst) // 2
        return lst[:mid], lst[mid:]

    # Construction des tours
    left_r16, right_r16 = split_half(group_stage_qualified)
    left_r16_pairs = make_pairs(left_r16)
    right_r16_pairs = make_pairs(right_r16)

    left_qf, right_qf = split_half(round16_winners)
    left_qf_pairs = make_pairs(left_qf)
    right_qf_pairs = make_pairs(right_qf)

    left_sf, right_sf = split_half(quarter_winners)
    left_sf_pair = make_pairs(left_sf)
    right_sf_pair = make_pairs(right_sf)

    final_pair = tuple(semi_winners)

    # Configuration graphique
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_facecolor("white")
    plt.axis("off")
    plt.title("Coupe du Monde 2022 — Bracket complet (prédictions)", fontsize=16, weight='bold')

    x_left = [0, 2, 4, 6]
    x_right = [12, 10, 8, 6]
    y_start = 14
    step = 2
    gold = "#d4af37"

    def draw_side(matches, x, y_start, direction=1):
        y_positions = []
        for i, (t1, t2) in enumerate(matches):
            y1 = y_start - i * step * 2
            y2 = y1 - step
            mid = (y1 + y2) / 2
            ha = 'right' if direction == 1 else 'left'
            align_shift = -0.2 if direction == 1 else 0.2
            plt.text(x + align_shift, y1, t1.title(), ha=ha, va='center', fontsize=9)
            plt.text(x + align_shift, y2, t2.title(), ha=ha, va='center', fontsize=9)
            plt.plot([x, x + direction], [y1, mid], color=gold, lw=1)
            plt.plot([x, x + direction], [y2, mid], color=gold, lw=1)
            y_positions.append(mid)
        return y_positions

    # Dessin des côtés
    y_qf_left = draw_side(left_r16_pairs, x_left[0], y_start, direction=1)
    y_sf_left = draw_side(left_qf_pairs, x_left[1], y_qf_left[0], direction=1)
    y_final_left = draw_side(left_sf_pair, x_left[2], y_sf_left[0], direction=1)

    y_qf_right = draw_side(right_r16_pairs, x_right[0], y_start, direction=-1)
    y_sf_right = draw_side(right_qf_pairs, x_right[1], y_qf_right[0], direction=-1)
    y_final_right = draw_side(right_sf_pair, x_right[2], y_sf_right[0], direction=-1)

    # Relier à la finale
    plt.plot([x_left[3], x_right[3]], [y_final_left[0], y_final_right[0]], color=gold, lw=1)

    # Finale et vainqueur
    plt.text(6, y_final_left[0] + 0.5, f"{final_pair[0].title()}  vs  {final_pair[1].title()}", fontsize=10, ha='center', weight='bold')
    plt.text(6, y_final_left[0] - 1.5, f"Champion : {final_winner.title()}", fontsize=12, color="black", ha='center', weight='bold')

    return fig

# --- Interface Streamlit ---
st.title("Prédiction du Vainqueur de la Coupe du Monde FIFA 2022")
st.markdown("""
Cette application affiche le bracket complet des prédictions pour la Coupe du Monde 2022.
Les prédictions sont basées sur des données historiques et un modèle statistique (détails à préciser).
""")

# Menu déroulant pour choisir le vainqueur
finalists = ['france', 'serbia']
selected_winner = st.selectbox("Choisissez le champion prédit :", finalists, index=0)

# Afficher le bracket
fig = draw_bracket(selected_winner)
st.pyplot(fig)

# Bouton pour afficher les détails
if st.button("Voir les équipes qualifiées"):
    st.write("""
    **Équipes qualifiées pour les 8èmes de finale :**  
    Senegal, Netherlands, England, Iran, Argentina, Poland, France, Denmark,  
    Spain, Germany, Belgium, Croatia, Brazil, Serbia, Uruguay, South Korea  
    **Note** : Ces prédictions sont basées sur des données hardcoded pour cet exemple.
    """)