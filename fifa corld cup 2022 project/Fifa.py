import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CSS personnalisé pour un look moderne ---
st.markdown("""
<style>
    .main {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .title {
        font-size: 36px;
        color: #1e3a8a;
        text-align: center;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subtitle {
        font-size: 20px;
        color: #4b5563;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #dc2626;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #b91c1c;
        transform: scale(1.05);
    }
    .stSelectbox {
        background-color: white;
        border-radius: 8px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Chargement des données (optionnel, décommentez si nécessaire) ---
# Placez les CSV dans un dossier 'data/' si utilisés
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

# --- Fonction pour générer le bracket ---
def draw_bracket(round16_winners, quarter_winners, semi_winners, final_winner):
    # Équipes qualifiées pour les huitièmes (hardcoded, mais modifiable via interface)
    group_stage_qualified = [
        'senegal', 'netherlands', 'england', 'iran',
        'argentina', 'poland', 'france', 'denmark',
        'spain', 'germany', 'belgium', 'croatia',
        'brazil', 'serbia', 'uruguay', 'south korea'
    ]

    def make_pairs(team_list):
        return [(team_list[i], team_list[i+1]) for i in range(0, len(team_list), 2)]

    def split_half(lst):
        mid = len(lst) // 2
        return lst[:mid], lst[mid:]

    # Construction des tours
    left_r16, right_r16 = split_half(group_stage_qualified)
    left_r16_pairs = make_pairs(left_r16)
    right_r16_pairs = make_pairs(right_r16)

    left_qf, right