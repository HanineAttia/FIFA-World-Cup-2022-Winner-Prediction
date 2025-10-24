#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd

matches = pd.read_csv(r"C:\Users\Dr.Console\Documents\fifa world cup 2022 project\matches_1930_2022.csv")
international = pd.read_csv(r"C:\Users\Dr.Console\Documents\fifa world cup 2022 project\international_matches.csv")
fifa = pd.read_csv(r"C:\Users\Dr.Console\Documents\fifa world cup 2022 project\fifa_ranking_2022-10-06.csv")
worldcup = pd.read_csv(r"C:\Users\Dr.Console\Documents\fifa world cup 2022 project\world_cup.csv")
groups2022 = pd.read_csv(r"C:\Users\Dr.Console\Documents\fifa world cup 2022 project\2022_world_cup_groups.csv")
squads2022 = pd.read_csv(r"C:\Users\Dr.Console\Documents\fifa world cup 2022 project\2022_world_cup_squads.csv", encoding="latin1")
teamsinfo = pd.read_csv(r"C:\Users\Dr.Console\Documents\fifa world cup 2022 project\world cups team infos.csv")


# In[6]:


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# # Matches:

# In[7]:


matches.head()


# In[8]:


from datetime import datetime

matches["Date"] = pd.to_datetime(matches["Date"])

matches = matches[matches["Date"]< "2022-11-20"]


# In[9]:


matches.head()


# In[10]:


worldcup.head()


# In[11]:


matches = matches.drop(["home_goal", "away_goal", "home_goal_long", "away_goal_long", "home_own_goal", "away_own_goal", "home_penalty_goal", "away_penalty_goal", "home_penalty_miss_long", "away_penalty_miss_long", "home_penalty_shootout_goal_long", "away_penalty_shootout_goal_long", "home_penalty_shootout_miss_long", "away_penalty_shootout_miss_long", "home_red_card", "away_red_card", "home_yellow_red_card", "away_yellow_red_card", "home_yellow_card_long", "away_yellow_card_long", "home_substitute_in_long", "away_substitute_in_long"], axis=1)


# In[12]:


matches = matches.drop("Notes", axis=1)


# In[13]:


matches = matches.drop(["home_penalty", "away_penalty", "Officials", "Referee", "home_manager", "home_captain", "away_manager", "away_captain", "Venue"], axis=1)


# In[14]:


matches.isnull().sum()


# In[15]:


matches[matches["home_xg"].isnull()]


# In[16]:


# Étape 1 : calculer la médiane globale
global_median = matches["home_xg"].median()

# Étape 2 : imputation par équipe, avec fallback sur la médiane globale
matches["home_xg"] = matches.groupby("home_team")["home_xg"].transform(
    lambda x: x.fillna(x.median() if x.median() == x.median() else global_median)
)


# In[17]:


matches.isnull().sum()


# In[18]:


# Étape 1 : calculer la médiane globale
global_median = matches["away_xg"].median()

# Étape 2 : imputation par équipe, avec fallback sur la médiane globale
matches["away_xg"] = matches.groupby("away_team")["away_xg"].transform(
    lambda x: x.fillna(x.median() if x.median() == x.median() else global_median)
)


# In[19]:


# 1. On extrait les colonnes home_team et away_team avec l'année du match pour les fusionner
home = matches[["home_team", "Year"]].rename(columns={"home_team": "Team"})
away = matches[["away_team", "Year"]].rename(columns={"away_team": "Team"})

# 2. On combine les deux tableaux en un seul
teams_years = pd.concat([home, away])

# 3. On supprime les doublons : une équipe peut jouer plusieurs matchs dans une même édition
unique_appearances = teams_years.drop_duplicates()

# 4. On compte le nombre d’années uniques par équipe ( on calcule le nombre d'éditions où chaque équipe est apparue)
team_counts = unique_appearances.groupby("Team")["Year"].nunique()

# 5. On trie et garde les 10 premières
top10 = team_counts.sort_values(ascending=False).head(10)

# 6. On affiche le graphe
top10.plot(kind="barh", figsize=(10,6), color="mediumseagreen", title="Top 10 équipes – Participations à la Coupe du Monde")


# In[20]:


def get_winner(row):
    if row["home_score"] > row["away_score"]:
        return row["home_team"]
    elif row["away_score"] > row["home_score"]:
        return row["away_team"]
    else:
        return None  # match nul
    


# In[21]:


#- On crée une nouvelle colonne winner qui contient le nom de l’équipe gagnante ou None si nul
matches["winner"] = matches.apply(get_winner, axis=1)

#cela donne le nombre total de victoires par équipe
win_counts = matches["winner"].value_counts()

#Extraire les 10 meilleure
top10_winners = win_counts.head(10)


# In[22]:


top10_winners.plot(kind="barh", figsize=(10,6), color="darkorange", title="Top 10 équipes – Victoires en Coupe du Monde")


# In[ ]:





# # international

# In[23]:


international.head()


# In[24]:


international["Date"] = pd.to_datetime(international["Date"])

international = international[international["Date"]< "2022-11-20"]


# In[25]:


international[international["Date"]> "2022-11-20"]


# In[26]:


international = international.drop(["Win Conditions", "Home Stadium"], axis=1)


# In[27]:


international.head()


# In[28]:


international["year"]=international["Date"].dt.year


# In[29]:


international.isnull().sum()


# # fifa:

# In[30]:


fifa.head()


# In[31]:


fifa = fifa.drop("team_code", axis=1)


# In[32]:


fifa.isnull().sum()


# In[33]:


import matplotlib.pyplot as plt

team= fifa["team"].head(10)
points = fifa["points"].head(10)

plt.figure(figsize=(10,6))
plt.barh(team, points, color="royalblue")
plt.title("Top 10 FIFA Rankings – juin 2022")
plt.xlabel("Points")
plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:





# # worldcup

# In[34]:


worldcup.head()


# In[35]:


worldcup=worldcup[worldcup["Year"] != 2022]


# In[36]:


worldcup["topscorer_name"] = worldcup["TopScorrer"].str.extract(r"^(.*)\s-\s\d+")
worldcup["topscorer_goals"] = worldcup["TopScorrer"].str.extract(r"(\d+)$").astype(int)


# In[37]:


worldcup.isnull().sum()


# In[38]:


titles = worldcup["Champion"].value_counts().sort_values(ascending=True)
titles.plot(kind="barh", figsize=(10,6), title="Nombre de titres remportés par équipe", color="gold")


# In[39]:


champion_years = worldcup.groupby("Champion")["Year"].apply(list).sort_values(ascending=True)

print(champion_years)


# In[40]:


finalistes = worldcup["Runner-Up"].value_counts().sort_values(ascending=True)
finalistes.plot(kind="barh", figsize=(10,6), title="Nombre de fois finaliste par équipe", color='pink')


# In[41]:


worldcup.head()


# In[42]:


worldcup = worldcup.copy() 
worldcup['Host Winner'] = False
worldcup.loc[worldcup["Host"] == worldcup["Champion"], 'Host Winner'] = True


# In[43]:


worldcup.head()


# In[44]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(data=worldcup, x='Host Winner', color='purple')
plt.title('Number of world cup HOST WINNERs')
plt.xticks(rotation=90);


# In[45]:


# !pip install pycountry


# #### World Cup Hosts colored by average goals per game

# In[46]:


import pycountry
import plotly.express as px

# Dictionnaire de base
countries = {country.name: country.alpha_3 for country in pycountry.countries}

# Ajouts manuels pour les noms non reconnus
countries.update({
    "Russia": "RUS",
    "United States": "USA",
    "Korea Republic": "KOR",
    "Japan": "JPN",
    "England": "GBR",
    "Iran": "IRN",
    "Côte d'Ivoire": "CIV",
    "Germany FR": "DEU",  # Exemple si tu as des noms historiques
    # Ajoute ici tous les noms spécifiques à ton dataset
})

# Colonne temporaire
wc_2002 = worldcup[worldcup["Year"] == 2002]
wc_2002_kor = wc_2002.copy()
wc_2002_kor["Host"] = "South Korea"

wc_2002_jpn = wc_2002.copy()
wc_2002_jpn["Host"] = "Japan"

# Supprimer l'original et concaténer les deux
worldcup = worldcup[worldcup["Year"] != 2002]
worldcup = pd.concat([worldcup, wc_2002_kor, wc_2002_jpn], ignore_index=True)
worldcup["HostCode"] = worldcup["Host"].apply(lambda x: countries.get(x, 'Unknown'))

# Carte
fig = px.choropleth(
    worldcup.sort_values("Year"),
    locations="HostCode",
    hover_name="Host",
    hover_data=worldcup.columns,
    color="AttendanceAvg",
    range_color=(min(worldcup["AttendanceAvg"]), max(worldcup["AttendanceAvg"])),
    projection="natural earth",
    animation_frame="Year"
)

fig.update_layout(margin={"r":5, "t":0, "l":5, "b":0})
fig.show()


# In[ ]:





# # squads2022:

# In[47]:


squads2022.head()


# In[48]:


squads2022.isnull().sum()


# In[49]:


squads2022["Player"] = squads2022["Player"].str.replace(r"\s*\(.*\)", "", regex=True).str.strip()


# In[50]:


squads2022[squads2022["Team"]=="Germany"].head(20)


# In[51]:


squads2022 = squads2022.drop("Club",axis=1)


# In[52]:


squads2022 = squads2022.drop("League",axis=1)


# # groups2022:

# In[53]:


groups2022.head()


# In[54]:


groups2022.isnull().sum()


# In[55]:


# Supprimer les points de suspension et parenthèses
worldcup["topscorer_cleaned"] = (
    worldcup["topscorer_name"]
    .str.replace(r"\.\.\.", "", regex=True)
    .str.replace(r"\(.*?\)", "", regex=True)
    .str.strip()
)

# Séparer les noms multiples
worldcup["topscorer_list"] = worldcup["topscorer_cleaned"].str.split(",").apply(lambda x: [name.strip() for name in x])


# In[56]:


topscorers = worldcup[worldcup["Year"] < 2022]["topscorer_list"].explode().dropna().unique()


# In[57]:


topscorers_in_2022 = squads2022[squads2022["Player"].isin(topscorers)]


# In[58]:


teams_with_topscorer = topscorers_in_2022["Team"].unique()


# In[59]:


groups2022["is_topscorer_in_team"] = groups2022["Team"].isin(teams_with_topscorer)


# In[60]:


groups2022[groups2022["is_topscorer_in_team"] == True]


# In[61]:


groups2022.head()


# In[62]:


# Étape 1 : calculer les victoires par équipe
wins_home = matches[matches["home_score"] > matches["away_score"]]["home_team"].value_counts()
wins_away = matches[matches["away_score"] > matches["home_score"]]["away_team"].value_counts()
total_wins = wins_home.add(wins_away, fill_value=0)

# Étape 2 : compter les titres remportés
titles = worldcup["Champion"].value_counts()


# Étape 3 : identifier les finalistes
finalists = pd.concat([worldcup["Champion"], worldcup["Runner-Up"]])
was_finalist = finalists.value_counts()

# Étape 4 : ajouter les colonnes à groups2022
groups2022["past_wins"] = groups2022["Team"].map(total_wins).fillna(0).astype(int)
groups2022["titles_won"] = groups2022["Team"].map(titles).fillna(0).astype(int)
groups2022["was_finalist"] = groups2022["Team"].map(was_finalist).fillna(0).astype(int) > 0


# In[ ]:





# In[63]:


groups2022.head()


# In[64]:


import os
print(os.getcwd())


# # teamsinfo:

# In[65]:


teamsinfo.head()


# In[66]:


teamsinfo[teamsinfo["tournament"]=="FIFA World Cup qualification"].head()


# In[67]:


teamsinfo = teamsinfo[teamsinfo["tournament"] == "FIFA World Cup qualification"]


# In[68]:


teamsinfo.drop(["city", "neutral_location"], axis=1)


# In[69]:


teamsinfo["date"]= pd.to_datetime(teamsinfo["date"])
teamsinfo["year"]= teamsinfo["date"].dt.year


# In[70]:


teamsinfo["away_team_mean_midfield_score"].describe()


# In[71]:


# Étape 1 : calculer la moyenne globale
global_mean = teamsinfo["home_team_mean_defense_score"].mean()

# Étape 2 : imputer les NaN par la moyenne de chaque équipe, ou par la moyenne globale si le groupe est vide
teamsinfo["home_team_mean_defense_score"] = teamsinfo.groupby("home_team")["home_team_mean_defense_score"].transform(
    lambda x: x.fillna(x.mean() if not x.dropna().empty else global_mean)
)


# In[72]:


# Étape 1 : calculer la moyenne globale
global_mean = teamsinfo["home_team_mean_offense_score"].mean()

# Étape 2 : imputer les NaN par la moyenne de chaque équipe, ou par la moyenne globale si le groupe est vide
teamsinfo["home_team_mean_offense_score"] = teamsinfo.groupby("home_team")["home_team_mean_offense_score"].transform(
    lambda x: x.fillna(x.mean() if not x.dropna().empty else global_mean)
)


# In[73]:


# Étape 1 : calculer la moyenne globale
global_mean = teamsinfo["away_team_mean_defense_score"].mean()

# Étape 2 : imputer les NaN par la moyenne de chaque équipe, ou par la moyenne globale si le groupe est vide
teamsinfo["away_team_mean_defense_score"] = teamsinfo.groupby("away_team")["away_team_mean_defense_score"].transform(
    lambda x: x.fillna(x.mean() if not x.dropna().empty else global_mean)
)


# In[74]:


# Étape 1 : calculer la moyenne globale
global_mean = teamsinfo["away_team_mean_midfield_score"].mean()

# Étape 2 : imputer les NaN par la moyenne de chaque équipe, ou par la moyenne globale si le groupe est vide
teamsinfo["away_team_mean_midfield_score"] = teamsinfo.groupby("away_team")["away_team_mean_midfield_score"].transform(
    lambda x: x.fillna(x.mean() if not x.dropna().empty else global_mean)
)


# In[75]:


# Étape 1 : calculer la moyenne globale
global_mean = teamsinfo["away_team_mean_offense_score"].mean()

# Étape 2 : imputer les NaN par la moyenne de chaque équipe, ou par la moyenne globale si le groupe est vide
teamsinfo["away_team_mean_offense_score"] = teamsinfo.groupby("away_team")["away_team_mean_offense_score"].transform(
    lambda x: x.fillna(x.mean() if not x.dropna().empty else global_mean)
)


# In[76]:


# Étape 1 : calculer la moyenne globale
global_mean = teamsinfo["away_team_goalkeeper_score"].mean()

# Étape 2 : imputer les NaN par la moyenne de chaque équipe, ou par la moyenne globale si le groupe est vide
teamsinfo["away_team_goalkeeper_score"] = teamsinfo.groupby("away_team")["away_team_goalkeeper_score"].transform(
    lambda x: x.fillna(x.mean() if not x.dropna().empty else global_mean)
)


# In[77]:


# Étape 1 : calculer la moyenne globale
global_mean = teamsinfo["home_team_goalkeeper_score"].mean()

# Étape 2 : imputer les NaN par la moyenne de chaque équipe, ou par la moyenne globale si le groupe est vide
teamsinfo["home_team_goalkeeper_score"] = teamsinfo.groupby("home_team")["home_team_goalkeeper_score"].transform(
    lambda x: x.fillna(x.mean() if not x.dropna().empty else global_mean)
)


# In[78]:


teamsinfo.head()


# In[79]:


teamsinfo.isnull().sum()


# In[80]:


import pandas as pd
import plotly.express as px

# Étape 1 : scores offensifs à domicile
home_offense = teamsinfo[["home_team", "home_team_mean_offense_score"]].rename(
    columns={"home_team": "team", "home_team_mean_offense_score": "offense_score"}
)

# Étape 2 : scores offensifs à l’extérieur
away_offense = teamsinfo[["away_team", "away_team_mean_offense_score"]].rename(
    columns={"away_team": "team", "away_team_mean_offense_score": "offense_score"}
)

# Étape 3 : fusionner les deux
all_offense = pd.concat([home_offense, away_offense], ignore_index=True)

# Étape 4 : moyenne offensive globale par équipe
avg_offense = all_offense.groupby("team")["offense_score"].mean()

# Étape 5 : top 10
top10_offense = avg_offense.sort_values(ascending=False).head(10)

# Étape 6 : pie chart
fig = px.pie(
    names=top10_offense.index,
    values=top10_offense.values,
    title="Top 10 strongest offensive teams",
    hole=0.3
)

fig.update_traces(textinfo='label+percent+value')
fig.show()


# In[81]:


# Étape 1 : préparer les scores à domicile
home_defense = teamsinfo[["home_team", "home_team_mean_defense_score"]].rename(
    columns={"home_team": "team", "home_team_mean_defense_score": "defense_score"}
)

# Étape 2 : préparer les scores à l’extérieur
away_defense = teamsinfo[["away_team", "away_team_mean_defense_score"]].rename(
    columns={"away_team": "team", "away_team_mean_defense_score": "defense_score"}
)

# Étape 3 : concaténer les deux
all_defense = pd.concat([home_defense, away_defense], ignore_index=True)

# Étape 4 : calculer la moyenne globale par équipe
avg_defense = all_defense.groupby("team")["defense_score"].mean()

# Étape 5 : sélectionner les 10 meilleures
top10_defense = avg_defense.sort_values(ascending=False).head(10)

# Étape 6 : visualiser en pie chart
fig = px.pie(
    names=top10_defense.index,
    values=top10_defense.values,
    title="Top 10 strongest defensive teams",
    hole=0.3
)

fig.update_traces(textinfo='label+percent+value')
fig.show()


# In[82]:


# Étape 1 : scores de milieu à domicile
home_midfield = teamsinfo[["home_team", "home_team_mean_midfield_score"]].rename(
    columns={"home_team": "team", "home_team_mean_midfield_score": "midfield_score"}
)

# Étape 2 : scores de milieu à l’extérieur
away_midfield = teamsinfo[["away_team", "away_team_mean_midfield_score"]].rename(
    columns={"away_team": "team", "away_team_mean_midfield_score": "midfield_score"}
)

# Étape 3 : fusionner les deux
all_midfield = pd.concat([home_midfield, away_midfield], ignore_index=True)

# Étape 4 : moyenne globale par équipe
avg_midfield = all_midfield.groupby("team")["midfield_score"].mean()

# Étape 5 : top 10
top10_midfield = avg_midfield.sort_values(ascending=False).head(10)

# Étape 6 : pie chart
fig = px.pie(
    names=top10_midfield.index,
    values=top10_midfield.values,
    title="Top 10 strongest midfield teams",
    hole=0.3
)

fig.update_traces(textinfo='label+percent+value')
fig.show()


# # construction d'un seul dataset

# In[83]:


import pandas as pd

# 🔧 Étape 0 : Harmonisation + nettoyage des noms d’équipes
team_name_map = {
    "USA": "United States",
    "IR Iran": "Iran",
    "Korea Republic": "South Korea",
    "Germany FR": "Germany",
    "Serbia and Montenegro": "Serbia",
    "Côte d'Ivoire": "Ivory Coast",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Czech Republic": "Czechia",
    "Republic of Ireland": "Ireland"
}

def clean_team_name(name):
    if isinstance(name, str):
        return name.strip().lower().replace("é", "e").replace("’", "'").replace("‘", "'")
    return name

def standardize(df):
    for col in df.columns:
        if "team" in col.lower() or "Champion" in col or "Runner-Up" in col:
            df[col] = df[col].replace(team_name_map)
            df[col] = df[col].apply(clean_team_name)
    return df

# Appliquer sur toutes les tables
matches = standardize(matches)
worldcup = standardize(worldcup)
teamsinfo = standardize(teamsinfo)
fifa = standardize(fifa)

# 🧩 Étape 1 : Extraire toutes les équipes participantes (sauf 2022)
home_teams = matches[matches["Year"] != 2022][["Year", "home_team"]].rename(columns={"home_team": "Team"})
away_teams = matches[matches["Year"] != 2022][["Year", "away_team"]].rename(columns={"away_team": "Team"})
teams_per_year = pd.concat([home_teams, away_teams]).drop_duplicates()
teams_per_year["Team"] = teams_per_year["Team"].apply(clean_team_name)

# 🧩 Étape 2 : Créer la cible is_champion
champions = worldcup[["Year", "Champion"]].rename(columns={"Champion": "Champion_Team"})
champions["Champion_Team"] = champions["Champion_Team"].apply(clean_team_name)
champions["is_champion"] = 1

base = teams_per_year.merge(champions, left_on=["Year", "Team"], right_on=["Year", "Champion_Team"], how="left")
base["is_champion"] = base["is_champion"].fillna(0).astype(int)
base.drop(columns=["Champion_Team"], inplace=True)

# 🧩 Étape 3 : Historique & palmarès
titles = worldcup["Champion"].value_counts()
finalists = pd.concat([worldcup["Champion"], worldcup["Runner-Up"]]).value_counts()
wins_home = matches[matches["home_score"] > matches["away_score"]]["home_team"].value_counts()
wins_away = matches[matches["away_score"] > matches["home_score"]]["away_team"].value_counts()
total_wins = wins_home.add(wins_away, fill_value=0)
participations = pd.concat([
    matches[["Year", "home_team"]].rename(columns={"home_team": "Team"}),
    matches[["Year", "away_team"]].rename(columns={"away_team": "Team"})
]).drop_duplicates()
appearances = participations["Team"].value_counts()

base["titles_won"] = base["Team"].map(titles).fillna(0).astype(int)
base["was_finalist"] = base["Team"].map(finalists).fillna(0).astype(int) > 0
base["past_wins"] = base["Team"].map(total_wins).fillna(0).astype(int)
base["world_cup_appearances"] = base["Team"].map(appearances).fillna(0).astype(int)

# 🧩 Étape 4 : Performance technique
home_info = teamsinfo[teamsinfo["year"] != 2022][[
    "year", "home_team", "home_team_mean_offense_score", "home_team_mean_defense_score",
    "home_team_mean_midfield_score", "home_team_goalkeeper_score"
]].rename(columns={
    "year": "Year", "home_team": "Team",
    "home_team_mean_offense_score": "offense", "home_team_mean_defense_score": "defense",
    "home_team_mean_midfield_score": "midfield", "home_team_goalkeeper_score": "goalkeeper"
})
away_info = teamsinfo[teamsinfo["year"] != 2022][[
    "year", "away_team", "away_team_mean_offense_score", "away_team_mean_defense_score",
    "away_team_mean_midfield_score", "away_team_goalkeeper_score"
]].rename(columns={
    "year": "Year", "away_team": "Team",
    "away_team_mean_offense_score": "offense", "away_team_mean_defense_score": "defense",
    "away_team_mean_midfield_score": "midfield", "away_team_goalkeeper_score": "goalkeeper"
})
long_scores = pd.concat([home_info, away_info])
long_scores["Team"] = long_scores["Team"].apply(clean_team_name)
long_scores["Year"] = long_scores["Year"].astype(int)

score_features = long_scores.groupby(["Year", "Team"]).mean().reset_index()
score_features["Team"] = score_features["Team"].apply(clean_team_name)
score_features["Year"] = score_features["Year"].astype(int)
base["Team"] = base["Team"].apply(clean_team_name)
base["Year"] = base["Year"].astype(int)

base = base.merge(score_features, on=["Year", "Team"], how="left")

# 🧩 Étape 5 : Imputation intelligente des scores manquants
team_means = score_features.groupby("Team")[["offense", "defense", "midfield", "goalkeeper"]].mean()
global_means = score_features[["offense", "defense", "midfield", "goalkeeper"]].mean()

def fill_scores(row):
    if pd.isna(row["offense"]):
        if row["Team"] in team_means.index:
            row["offense"] = team_means.loc[row["Team"], "offense"]
            row["defense"] = team_means.loc[row["Team"], "defense"]
            row["midfield"] = team_means.loc[row["Team"], "midfield"]
            row["goalkeeper"] = team_means.loc[row["Team"], "goalkeeper"]
        else:
            row["offense"] = global_means["offense"]
            row["defense"] = global_means["defense"]
            row["midfield"] = global_means["midfield"]
            row["goalkeeper"] = global_means["goalkeeper"]
    return row

base = base.apply(fill_scores, axis=1)

# 🧩 Étape 6 : Buts, xG, win rate
home_stats = matches[matches["Year"] != 2022][["Year", "home_team", "home_score", "away_score", "home_xg"]].rename(columns={
    "home_team": "Team", "home_score": "goals_for", "away_score": "goals_against", "home_xg": "xg"
})
away_stats = matches[matches["Year"] != 2022][["Year", "away_team", "away_score", "home_score", "away_xg"]].rename(columns={
    "away_team": "Team", "away_score": "goals_for", "home_score": "goals_against", "away_xg": "xg"
})
goals_xg = pd.concat([home_stats, away_stats])
agg_stats = goals_xg.groupby(["Year", "Team"]).agg({
    "goals_for": "mean", "goals_against": "mean", "xg": "mean"
}).reset_index().rename(columns={
    "goals_for": "avg_goals_scored", "goals_against": "avg_goals_conceded", "xg": "avg_xg"
})
base = base.merge(agg_stats, on=["Year", "Team"], how="left")

wins = pd.concat([
    matches[matches["home_score"] > matches["away_score"]][["Year", "home_team"]].rename(columns={"home_team": "Team"}),
    matches[matches["away_score"] > matches["home_score"]][["Year", "away_team"]].rename(columns={"away_team": "Team"})
])
total_matches = pd.concat([
    matches[["Year", "home_team"]].rename(columns={"home_team": "Team"}),
    matches[["Year", "away_team"]].rename(columns={"away_team": "Team"})
])
win_counts = wins.groupby(["Year", "Team"]).size()
match_counts = total_matches.groupby(["Year", "Team"]).size()
win_rate = (win_counts / match_counts).fillna(0).reset_index(name="win_rate")
base = base.merge(win_rate, on=["Year", "Team"], how="left")

# 🧩 Étape 7 : Contexte FIFA
fifa_features = fifa.rename(columns={"team": "Team"}).groupby("Team").agg({
    "rank": "mean", "points": "mean",
    "association": "first"
}).reset_index().rename(columns={
    "rank": "fifa_rank",
    "points": "fifa_points",
    "association": "confederation"
})
fifa_features["Team"] = fifa_features["Team"].apply(clean_team_name)
base = base.merge(fifa_features, on="Team", how="left")
cols_to_impute = [
    "avg_goals_scored", "avg_goals_conceded", "avg_xg", "win_rate",
    "fifa_rank", "fifa_points"
]
for col in cols_to_impute:
    base[col] = base[col].fillna(base[col].mean())


# In[84]:


manual_confeds = {
    "fr yugoslavia": "UEFA",
    "serbia": "UEFA",
    "west germany": "UEFA",
    "germany dr": "UEFA",
    "czechoslovakia": "UEFA",
    "zaire": "CAF",
    "dutch east indies": "AFC",
    "germany fr": "UEFA",
    "germany": "UEFA",
    "soviet union": "UEFA",
    "russia": "UEFA",
    "iran": "AFC",
    "south korea": "AFC",
    "ivory coast": "CAF",
    "republic of ireland": "UEFA",
    "czech republic": "UEFA",
    "czechia": "UEFA",
    "serbia and montenegro": "UEFA",
    "united states": "CONCACAF",
    "zaire": "CAF",
    "east germany": "UEFA"
}

# Nettoyer les noms
base["Team"] = base["Team"].apply(clean_team_name)

# Compléter les confédérations manquantes
base["confederation"] = base.apply(
    lambda row: manual_confeds[row["Team"]] if pd.isna(row["confederation"]) and row["Team"] in manual_confeds else row["confederation"],
    axis=1
)


# In[ ]:





# In[ ]:





# In[85]:


base.head()


# In[ ]:





# In[86]:


base.isnull().sum()


# In[87]:


base.size


# In[88]:


base["was_finalist"] = base["was_finalist"].astype(bool)


# In[89]:


# One-Hot Encoding de la colonne confederation
confed_encoded = pd.get_dummies(base["confederation"], prefix="confed")

# Fusionner avec base
base = pd.concat([base.drop(columns=["confederation"]), confed_encoded], axis=1)

# Optionnel : supprimer la colonne originale si tu veux
# base.drop(columns=["confederation"], inplace=True)


# In[90]:


print(confed_encoded.head())
print("Colonnes encodées :", confed_encoded.columns.tolist())


# In[91]:


base["team_score"] = (
    base["offense"] +
    base["defense"] +
    base["midfield"] +
    base["goalkeeper"]
)


# In[92]:


base.head()


# # Modeling

# In[93]:


# Cible : prédire si une équipe a gagné la Coupe du Monde
y = base["is_champion"]

# Variables explicatives : toutes sauf Team, Year, is_champion
X = base.drop(columns=["Team", "Year", "is_champion"])


# In[94]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


# ## Logistic Regression

# In[95]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logreg = LogisticRegression(max_iter=1000, class_weight='balanced')
logreg.fit(X_train_scaled, y_train)
y_pred_logreg = logreg.predict(X_test_scaled)

print("🔹 Logistic Regression")
print(confusion_matrix(y_test, y_pred_logreg))
print(classification_report(y_test, y_pred_logreg))


# ## Random Forest

# In[96]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("🔹 Random Forest")
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))


# In[97]:


from xgboost import XGBClassifier

X_train_xgb = X_train.copy()
X_test_xgb = X_test.copy()

# # Convertir les colonnes booléennes en int
for col in X_train_xgb.select_dtypes(include='bool').columns:
    X_train_xgb[col] = X_train_xgb[col].astype(int)
    X_test_xgb[col] = X_test_xgb[col].astype(int)

# Supprimer les colonnes dupliquées
X_train_xgb = X_train_xgb.loc[:, ~X_train_xgb.columns.duplicated()]
X_test_xgb = X_test_xgb.loc[:, ~X_test_xgb.columns.duplicated()]
    
xgb = XGBClassifier(eval_metric="logloss", random_state=42)
xgb.fit(X_train_xgb, y_train)
y_pred_xgb = xgb.predict(X_test_xgb)

print("🔹 XGBoost")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))


# ## les variables les plus importantes:

# In[98]:


from xgboost import plot_importance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plot_importance(xgb, max_num_features=20, importance_type='gain', height=0.5, grid=False)
plt.title("Top 20 variables les plus influentes (XGBoost)")
plt.tight_layout()
plt.show()


# => XGBoost est le modele le plus performant

# # Utiliser XGBoost pour prédire les probabilités de victoire des équipes en 2022

# In[99]:


teams_2022 = groups2022[["Team"]].copy()
base = base.copy()


# In[100]:


import numpy as np
equipes_2022 = groups2022["Team"].str.strip().str.lower().unique()
base["Team"] = base["Team"].str.strip().str.lower()
base = base.loc[:, ~base.columns.duplicated()]  # Supprimer les colonnes dupliquées
base_existantes = base[base["Team"].isin(equipes_2022)].copy()
equipes_manquantes = set(equipes_2022) - set(base_existantes["Team"])
print("Équipes sans historique :", equipes_manquantes)
colonnes = base.columns
base_manquantes = pd.DataFrame(columns=colonnes)

for team in equipes_manquantes:
    ligne_vide = {col: np.nan for col in colonnes}
    ligne_vide["Team"] = str(team)  # ← chaîne explicite
    ligne_vide["Year"] = 2022       # ← entier explicite

    # Convertir en DataFrame avec dtype object pour éviter les conflits
    base_manquantes = pd.concat([base_manquantes, pd.DataFrame([ligne_vide])], ignore_index=True)


base_manquantes.loc[base_manquantes["Team"] == "qatar", "Team"] = "qatar"
base_manquantes.loc[base_manquantes["Team"] == "qatar", "Year"] = 2022
base_manquantes.loc[base_manquantes["Team"] == "qatar", "is_champion"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "titles_won"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "was_finalist"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "past_wins"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "world_cup_appearances"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "offense"] = 71
base_manquantes.loc[base_manquantes["Team"] == "qatar", "defense"] = 63
base_manquantes.loc[base_manquantes["Team"] == "qatar", "midfield"] = 68
base_manquantes.loc[base_manquantes["Team"] == "qatar", "goalkeeper"] = 71.9
base_manquantes.loc[base_manquantes["Team"] == "qatar", "avg_goals_scored"] = 0.75
base_manquantes.loc[base_manquantes["Team"] == "qatar", "avg_goals_conceded"] = 1.95
base_manquantes.loc[base_manquantes["Team"] == "qatar", "avg_xg"] = 0.85
base_manquantes.loc[base_manquantes["Team"] == "qatar", "win_rate"] = 0.13
base_manquantes.loc[base_manquantes["Team"] == "qatar", "fifa_rank"] = 50
base_manquantes.loc[base_manquantes["Team"] == "qatar", "fifa_points"] = 1439.89
base_manquantes.loc[base_manquantes["Team"] == "qatar", "confed_AFC"] = 1
base_manquantes.loc[base_manquantes["Team"] == "qatar", "confed_CAF"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "confed_CONCACAF"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "confed_CONMEBOL"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "confed_OFC"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "confed_UEFA"] = 0
base_manquantes.loc[base_manquantes["Team"] == "qatar", "team_score"] = 273.9


# In[101]:


base_manquantes.head()


# In[102]:


base2022 = pd.concat([base_existantes, base_manquantes], ignore_index=True)
base2022["Year"] = 2022  # Uniformiser l’année


# In[103]:


print(base2022["Team"].nunique())  # Doit être égal au nombre d’équipes dans groups2022
print(base2022.isnull().sum())     # Identifier les colonnes à compléter


# In[104]:


top_features = [
    "fifa_points", "avg_goals_scored", "avg_goals_conceded",
    "avg_xg", "offense", "defense", "midfield", "goalkeeper",
    "win_rate", "titles_won", "world_cup_appearances",
    "confed_AFC", "confed_CAF", "confed_CONCACAF",
    "confed_CONMEBOL", "confed_OFC", "confed_UEFA"
]
missing_cols = set(top_features) - set(base2022.columns)
print("Colonnes manquantes :", missing_cols)


# In[105]:


print(base2022[top_features].isnull().sum())  # Toutes les colonnes devraient être à 0


# In[106]:


base2022.head()


# In[107]:


expected_features = ['titles_won', 'was_finalist', 'past_wins', 'world_cup_appearances', 'offense', 'defense', 'midfield', 'goalkeeper', 'avg_goals_scored', 'avg_goals_conceded', 'avg_xg', 'win_rate', 'fifa_rank', 'fifa_points', 'confed_AFC', 'confed_CAF', 'confed_CONCACAF', 'confed_CONMEBOL', 'confed_OFC', 'confed_UEFA', 'team_score']
# expected_features = [
#     'win_rate', 'avg_goals_scored', 'avg_goals_conceded', 'avg_xg',
#     'titles_won', 'past_wins', 'defense', 'goalkeeper', 'fifa_points',
#     'fifa_rank', 'midfield', 'offense', 'world_cup_appearances',
#     'confed_CONMEBOL', 'was_finalist', 'confed_CONCACAF', 'confed_CAF', 'confed_AFC', 'confed_OFC', 'confed_UEFA'
# ]

X = base2022[expected_features].copy()
X = X.apply(pd.to_numeric, errors="coerce")
#X.fillna(X.mean(), inplace=True)


# In[108]:


print(X.dtypes)


# In[109]:


print(X.isnull().sum())


# In[110]:


y_pred = xgb.predict(X)
base2022["prediction"] = y_pred


# In[111]:


base2022[["Team", "prediction"]].sort_values(by="prediction", ascending=False)


# In[112]:


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
teams_sorted = base2022.sort_values(by="prediction", ascending=False)
plt.barh(teams_sorted["Team"], teams_sorted["prediction"], color="skyblue")
plt.xlabel("Score prédictif")
plt.title("Prédiction XGBoost – Coupe du Monde 2022")
plt.gca().invert_yaxis()
plt.show()


# In[113]:


gagnant = base2022.loc[base2022["prediction"].idxmax()]


# In[114]:


from pyfiglet import figlet_format
print("\n🏆 ÉQUIPE GAGNANTE PRÉDITE 🏆\n")
print(figlet_format(gagnant["Team"].upper()))


# In[115]:


print("Score prédictif :", round(gagnant["prediction"], 4))


# # Processus de simulation du tournoi de la Coupe du Monde 2022 avec XGBoost:

# ## Phase de groupes → Sélection des 16 équipes qualifiées

# In[116]:


print(base2022[expected_features].dtypes)


# In[117]:


for col in expected_features:
    base2022[col] = pd.to_numeric(base2022[col], errors="coerce")


# In[118]:


print(base2022[expected_features].dtypes)


# In[119]:


teams_in_groups = groups2022["Team"].unique()
teams_in_base = base2022["Team"].unique()

missing_teams = [team for team in teams_in_groups if team not in teams_in_base]
print("Équipes présentes dans groups2022 mais absentes de base2022 :", missing_teams)


# In[120]:


import pandas as pd

# --- Uniformiser les noms ---
groups2022["Team"] = groups2022["Team"].str.strip().str.lower()
base2022["Team"] = base2022["Team"].str.strip().str.lower()

# --- Filtrer base2022 pour ne garder que les équipes du tournoi ---
worldcup_teams = groups2022["Team"].unique().tolist()
base2022_filtered = base2022[base2022["Team"].isin(worldcup_teams)].copy()

# --- Initialiser la liste des qualifiés ---
qualified_teams = []

# --- Boucle par groupe ---
for group in groups2022["Group"].unique():
    teams_in_group = groups2022[groups2022["Group"] == group]["Team"]
    group_features = base2022_filtered[base2022_filtered["Team"].isin(teams_in_group)].copy()

    # Vérification du nombre d'équipes valides
    if group_features.shape[0] < 2:
        print(f"Groupe {group} incomplet :", teams_in_group.tolist())
        continue

    # Conversion des colonnes en numériques
    for col in expected_features:
        group_features[col] = pd.to_numeric(group_features[col], errors="coerce")

    # Suppression des lignes avec NaN
    group_features = group_features.dropna(subset=expected_features)

    if group_features.shape[0] < 2:
        print(f"Groupe {group} a trop de NaN après nettoyage")
        continue

    # Prédiction des scores
    group_features["score"] = rf.predict(group_features[expected_features])

    # Sélection des 2 meilleures équipes distinctes
    top_sorted = group_features.sort_values(by="score", ascending=False)
    top2_unique = top_sorted["Team"].drop_duplicates().head(2).tolist()

    if len(top2_unique) < 2:
        print(f"⚠️ Groupe {group} n’a qu’une équipe valide : {top2_unique}")
        remaining = teams_in_group[~teams_in_group.isin(top2_unique)].tolist()
        if remaining:
            top2_unique.append(remaining[0])
            print(f"✅ Ajout forcé de {remaining[0]} comme 2ème équipe du groupe {group}")
        else:
            print(f"❌ Groupe {group} ne peut pas être complété")

    qualified_teams.extend(top2_unique)

# --- Résultat final ---
print(f"\n✅ Nombre d'équipes qualifiées : {len(qualified_teams)}")
print("✅ Équipes qualifiées :", qualified_teams)


# In[121]:


for group in groups2022["Group"].unique():
    teams_in_group = groups2022[groups2022["Group"] == group]["Team"]
    group_features = base2022_filtered[base2022_filtered["Team"].isin(teams_in_group)]
    print(f"Groupe {group} → {group_features['Team'].nunique()} équipes valides")

print("Nombre d'équipes qualifiées :", len(qualified_teams))
print("Équipes qualifiées :", qualified_teams)


# ## Round of 16 → 8 match-ups

# In[122]:


round_of_16_pairs = [
    (qualified_teams[0], qualified_teams[3]),  # A1 vs B2
    (qualified_teams[2], qualified_teams[1]),  # B1 vs A2
    (qualified_teams[4], qualified_teams[7]),  # C1 vs D2
    (qualified_teams[6], qualified_teams[5]),  # D1 vs C2
    (qualified_teams[8], qualified_teams[11]), # E1 vs F2
    (qualified_teams[10], qualified_teams[9]), # F1 vs E2
    (qualified_teams[12], qualified_teams[15]),# G1 vs H2
    (qualified_teams[14], qualified_teams[13]) # H1 vs G2
]

def predict_match(team1, team2):
    t1 = base2022[base2022["Team"] == team1]
    t2 = base2022[base2022["Team"] == team2]
    p1 = xgb.predict(t1[expected_features])[0]
    p2 = xgb.predict(t2[expected_features])[0]
    return team1 if p1 > p2 else team2

quarter_finalists = [predict_match(t1, t2) for t1, t2 in round_of_16_pairs]


# In[123]:


for winner in quarter_finalists:
    found = any(winner in pair for pair in round_of_16_pairs)
    if not found:
        print(f"❌ {winner} n'est pas dans le Round of 16")
    else:
        print(f"✅ {winner} est bien dans le Round of 16")


# In[ ]:


print("✅ Nombre d'équipes qualifiées :", len(quarter_finalists))
print("✅ Équipes qualifiées :", quarter_finalists)


# ## Quarts de finale → 4 matchs

# In[ ]:


quarter_pairs = [(quarter_finalists[i], quarter_finalists[i+1]) for i in range(0, 8, 2)]
semi_finalists = [predict_match(t1, t2) for t1, t2 in quarter_pairs]


# In[ ]:


print("✅ Nombre d'équipes qualifiées :", len(semi_finalists))
print("✅ Équipes qualifiées :", semi_finalists)


# ## Demi-finales → 2 matchs

# In[ ]:


semi_pairs = [(semi_finalists[i], semi_finalists[i+1]) for i in range(0, 4, 2)]
finalists = [predict_match(t1, t2) for t1, t2 in semi_pairs]


# In[ ]:


print("✅ Nombre d'équipes qualifiées :", len(finalists))
print("✅ Équipes qualifiées :", finalists)


# ## Finale → 1 match

# In[ ]:


winner = predict_match(finalists[0], finalists[1])
print("🏆 ÉQUIPE GAGNANTE PRÉDITE :", winner.upper())


# In[ ]:


import matplotlib.pyplot as plt

# --- Résultats du modèle ---
group_stage_qualified = [
    'senegal', 'netherlands', 'england', 'iran',
    'argentina', 'poland', 'france', 'denmark',
    'spain', 'germany', 'belgium', 'croatia',
    'brazil', 'serbia', 'uruguay', 'south korea'
]

round16_winners = ['iran', 'netherlands', 'denmark', 'france', 'croatia', 'germany', 'south korea', 'serbia']
quarter_winners = ['netherlands', 'france', 'germany', 'serbia']
semi_winners = ['france', 'serbia']
final_winner = 'france'

# --- Génération des paires dynamiques ---
def make_pairs(team_list):
    return [(team_list[i], team_list[i+1]) for i in range(0, len(team_list), 2)]

# --- Séparation stricte des moitiés ---
def split_half(lst):
    mid = len(lst) // 2
    return lst[:mid], lst[mid:]

# --- Construction des tours ---
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

# --- Configuration graphique ---
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

# --- Dessin du bracket gauche ---
y_qf_left = draw_side(left_r16_pairs, x_left[0], y_start, direction=1)
y_sf_left = draw_side(left_qf_pairs, x_left[1], y_qf_left[0], direction=1)
y_final_left = draw_side(left_sf_pair, x_left[2], y_sf_left[0], direction=1)

# --- Dessin du bracket droit ---
y_qf_right = draw_side(right_r16_pairs, x_right[0], y_start, direction=-1)
y_sf_right = draw_side(right_qf_pairs, x_right[1], y_qf_right[0], direction=-1)
y_final_right = draw_side(right_sf_pair, x_right[2], y_sf_right[0], direction=-1)

# --- Relier les deux côtés vers la finale ---
plt.plot([x_left[3], x_right[3]], [y_final_left[0], y_final_right[0]], color=gold, lw=1)

# --- Finale et vainqueur ---
plt.text(6, y_final_left[0] + 0.5, f"{final_pair[0].title()}  vs  {final_pair[1].title()}", fontsize=10, ha='center', weight='bold')
plt.text(6, y_final_left[0] - 1.5, f"Champion : {final_winner.title()}", fontsize=12, color="black", ha='center', weight='bold')

plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from operator import itemgetter

# --- 1. Nettoyage des noms ---
groups2022["Team"] = groups2022["Team"].str.lower().str.strip()
base2022_filtered["Team"] = base2022_filtered["Team"].str.lower().str.strip()

# --- 2. Préparer les groupes et les matchs ---
groups = sorted(groups2022["Group"].unique())
table = {}
matches = []

for group in groups:
    teams = groups2022[groups2022["Group"] == group]["Team"].tolist()
    table[group] = [[team, 0, []] for team in teams]
    matches.extend([
        (group, teams[0], teams[1]),
        (group, teams[2], teams[3]),
        (group, teams[0], teams[2]),
        (group, teams[1], teams[3]),
        (group, teams[0], teams[3]),
        (group, teams[1], teams[2])
    ])

# --- 3. Simulation des matchs de groupes ---
advanced_group = []
last_group = ""

for k in table.keys():
    for t in table[k]:
        t[1] = 0
        t[2] = []

for teams in matches:
    draw = False
    t1 = base2022_filtered[base2022_filtered["Team"] == teams[1]]
    t2 = base2022_filtered[base2022_filtered["Team"] == teams[2]]

    if t1.empty or t2.empty:
        print(f"❌ Données manquantes pour {teams[1]} ou {teams[2]}")
        continue

    t1_features = t1[expected_features]
    t2_features = t2[expected_features]

    prob_t1 = xgb.predict_proba(t1_features)[0][1]
    prob_t2 = xgb.predict_proba(t2_features)[0][1]

    if abs(prob_t1 - prob_t2) < 0.05:
        draw = True
        for i in table[teams[0]]:
            if i[0] in [teams[1], teams[2]]:
                i[1] += 1
    elif prob_t1 > prob_t2:
        for i in table[teams[0]]:
            if i[0] == teams[1]:
                i[1] += 3
    else:
        for i in table[teams[0]]:
            if i[0] == teams[2]:
                i[1] += 3

    for i in table[teams[0]]:
        if i[0] == teams[1]:
            i[2].append(prob_t1)
        if i[0] == teams[2]:
            i[2].append(prob_t2)

    if draw:
        print(f"Group {teams[0]} - {teams[1]} vs. {teams[2]}: Draw")
    else:
        winner = teams[1] if prob_t1 > prob_t2 else teams[2]
        print(f"Group {teams[0]} - {teams[1]} vs. {teams[2]}: Winner {winner} with prob {max(prob_t1, prob_t2):.2f}")

    if last_group != teams[0] and last_group != "":
        for i in table[last_group]:
            i[2] = np.mean(i[2])
        final_table = sorted(table[last_group], key=itemgetter(1,2), reverse=True)
        advanced_group.append([final_table[0][0], final_table[1][0]])
        print(f"\nGroup {last_group} advanced: {[final_table[0][0], final_table[1][0]]}")
    last_group = teams[0]

for i in table[last_group]:
    i[2] = np.mean(i[2])
final_table = sorted(table[last_group], key=itemgetter(1,2), reverse=True)
advanced_group.append([final_table[0][0], final_table[1][0]])
print(f"\nGroup {last_group} advanced: {[final_table[0][0], final_table[1][0]]}")

# --- 4. Playoffs ---
playoffs = {"Round of 16": [], "Quarter-Final": [], "Semi-Final": [], "Final": []}
control = [team for pair in advanced_group for team in pair]
playoffs["Round of 16"] = [[control[i], control[i+1]] for i in range(0, len(control), 2)]

def simulate_round(round_name, teams_list):
    next_round = []
    print("-"*10, f"Starting simulation of {round_name}", "-"*10, "\n")
    for game in teams_list:
        t1 = base2022_filtered[base2022_filtered["Team"] == game[0]]
        t2 = base2022_filtered[base2022_filtered["Team"] == game[1]]

        if t1.empty or t2.empty:
            print(f"❌ Données manquantes pour {game[0]} ou {game[1]}")
            continue

        t1_features = t1[expected_features]
        t2_features = t2[expected_features]

        prob_t1 = xgb.predict_proba(t1_features)[0][1]
        prob_t2 = xgb.predict_proba(t2_features)[0][1]

        winner = game[0] if prob_t1 >= prob_t2 else game[1]
        next_round.append(winner)
        game.append([prob_t1, prob_t2])
        print(f"{game[0]} vs. {game[1]} → {winner} advances with prob {max(prob_t1, prob_t2):.2f}")
    return next_round, teams_list

round_names = ["Round of 16", "Quarter-Final", "Semi-Final", "Final"]
for idx, r in enumerate(round_names):
    next_rounds, playoffs[r] = simulate_round(r, playoffs[r] if r=="Round of 16" else [[next_rounds[i], next_rounds[i+1]] for i in range(0, len(next_rounds), 2)])

# --- 5. Visualisation ---
plt.figure(figsize=(15, 10))
G = nx.balanced_tree(2, 3)
labels = []

for p in playoffs.keys():
    for game in playoffs[p]:
        label = f"{game[0]} ({round(game[2][0],2)})\n{game[1]} ({round(game[2][1],2)})"
        labels.append(label)

labels_dict = {i: labels[::-1][i] for i in range(len(G.nodes))}
pos = graphviz_layout(G, prog='twopi')
nx.draw(G, pos=pos, with_labels=False, node_color=range(len(G.nodes)), edge_color="#bbf5bb", width=5,
        font_weight='bold', cmap=plt.cm.Greens, node_size=4000)
nx.draw_networkx_labels(G, pos=pos, labels=labels_dict, font_size=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5))
plt.axis('equal')
plt.title(f"Champion prédit : {next_rounds[0].title()}", fontsize=14, weight='bold')
plt.show()


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from networkx.drawing.nx_pydot import graphviz_layout
from operator import itemgetter

# --- 1. Nettoyage des noms ---
groups2022["Team"] = groups2022["Team"].str.lower().str.strip()
base2022_filtered["Team"] = base2022_filtered["Team"].str.lower().str.strip()

# --- 2. Préparer les groupes et les matchs ---
groups = sorted(groups2022["Group"].unique())
table = {}
matches = []

for group in groups:
    teams = groups2022[groups2022["Group"] == group]["Team"].tolist()
    table[group] = [[team, 0, []] for team in teams]
    matches.extend([
        (group, teams[0], teams[1]),
        (group, teams[2], teams[3]),
        (group, teams[0], teams[2]),
        (group, teams[1], teams[3]),
        (group, teams[0], teams[3]),
        (group, teams[1], teams[2])
    ])

# --- 3. Simulation des matchs de groupes ---
advanced_group = []
last_group = ""

for k in table.keys():
    for t in table[k]:
        t[1] = 0
        t[2] = []

for teams in matches:
    draw = False
    t1 = base2022_filtered[base2022_filtered["Team"] == teams[1]]
    t2 = base2022_filtered[base2022_filtered["Team"] == teams[2]]

    if t1.empty or t2.empty:
        continue

    t1_features = t1[expected_features]
    t2_features = t2[expected_features]

    prob_t1 = xgb.predict_proba(t1_features)[0][1]
    prob_t2 = xgb.predict_proba(t2_features)[0][1]

    if abs(prob_t1 - prob_t2) < 0.05:
        draw = True
        for i in table[teams[0]]:
            if i[0] in [teams[1], teams[2]]:
                i[1] += 1
    elif prob_t1 > prob_t2:
        for i in table[teams[0]]:
            if i[0] == teams[1]:
                i[1] += 3
    else:
        for i in table[teams[0]]:
            if i[0] == teams[2]:
                i[1] += 3

    for i in table[teams[0]]:
        if i[0] == teams[1]:
            i[2].append(prob_t1)
        if i[0] == teams[2]:
            i[2].append(prob_t2)

    if last_group != teams[0] and last_group != "":
        for i in table[last_group]:
            i[2] = np.mean(i[2])
        final_table = sorted(table[last_group], key=itemgetter(1,2), reverse=True)
        advanced_group.append([final_table[0][0], final_table[1][0]])
    last_group = teams[0]

for i in table[last_group]:
    i[2] = np.mean(i[2])
final_table = sorted(table[last_group], key=itemgetter(1,2), reverse=True)
advanced_group.append([final_table[0][0], final_table[1][0]])

# --- 4. Playoffs ---
playoffs = {"Round of 16": [], "Quarter-Final": [], "Semi-Final": [], "Final": []}
control = [team for pair in advanced_group for team in pair]
playoffs["Round of 16"] = [[control[i], control[i+1]] for i in range(0, len(control), 2)]

def simulate_round(round_name, teams_list):
    next_round = []
    for game in teams_list:
        t1 = base2022_filtered[base2022_filtered["Team"] == game[0]]
        t2 = base2022_filtered[base2022_filtered["Team"] == game[1]]

        if t1.empty or t2.empty:
            continue

        t1_features = t1[expected_features]
        t2_features = t2[expected_features]

        prob_t1 = xgb.predict_proba(t1_features)[0][1]
        prob_t2 = xgb.predict_proba(t2_features)[0][1]

        winner = game[0] if prob_t1 >= prob_t2 else game[1]
        next_round.append(winner)
        game.append(winner)  # juste le nom du gagnant
    return next_round, teams_list

round_names = ["Round of 16", "Quarter-Final", "Semi-Final", "Final"]
for idx, r in enumerate(round_names):
    next_rounds, playoffs[r] = simulate_round(r, playoffs[r] if r=="Round of 16" else [[next_rounds[i], next_rounds[i+1]] for i in range(0, len(next_rounds), 2)])

# --- 5. Visualisation sans scores ---
plt.figure(figsize=(15, 10))
G = nx.balanced_tree(2, 3)
labels = []

for p in playoffs.keys():
    for game in playoffs[p]:
        label = f"{game[0].title()} vs {game[1].title()}"
        labels.append(label)

labels_dict = {i: labels[::-1][i] for i in range(len(G.nodes))}
pos = graphviz_layout(G, prog='twopi')
nx.draw(G, pos=pos, with_labels=False, node_color=range(len(G.nodes)), edge_color="#bbf5bb", width=5,
        font_weight='bold', cmap=plt.cm.Greens, node_size=4000)
nx.draw_networkx_labels(G, pos=pos, labels=labels_dict, font_size=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5))
plt.axis('equal')
plt.title(f"Champion prédit : {next_rounds[0].title()}", fontsize=14, weight='bold')
plt.show()


# In[ ]:


# --- 1. Nettoyage des noms ---
groups2022["Team"] = groups2022["Team"].str.lower().str.strip()
base2022_filtered["Team"] = base2022_filtered["Team"].str.lower().str.strip()

# --- 2. Préparer les groupes et les matchs ---
groups = sorted(groups2022["Group"].unique())
table = {}
matches = []

for group in groups:
    teams = groups2022[groups2022["Group"] == group]["Team"].tolist()
    table[group] = [[team, 0, []] for team in teams]
    matches.extend([
        (group, teams[0], teams[1]),
        (group, teams[2], teams[3]),
        (group, teams[0], teams[2]),
        (group, teams[1], teams[3]),
        (group, teams[0], teams[3]),
        (group, teams[1], teams[2])
    ])

# --- 3. Simulation des matchs de groupes ---
advanced_group = []
last_group = ""

for k in table.keys():
    for t in table[k]:
        t[1] = 0
        t[2] = []

for teams in matches:
    draw = False
    t1 = base2022_filtered[base2022_filtered["Team"] == teams[1]]
    t2 = base2022_filtered[base2022_filtered["Team"] == teams[2]]

    if t1.empty or t2.empty:
        continue

    score_t1 = t1[["offense", "defense", "midfield", "goalkeeper"]].sum(axis=1).values[0]
    score_t2 = t2[["offense", "defense", "midfield", "goalkeeper"]].sum(axis=1).values[0]

    if abs(score_t1 - score_t2) < 1e-2:
        draw = True
        for i in table[teams[0]]:
            if i[0] in [teams[1], teams[2]]:
                i[1] += 1
    elif score_t1 > score_t2:
        for i in table[teams[0]]:
            if i[0] == teams[1]:
                i[1] += 3
    else:
        for i in table[teams[0]]:
            if i[0] == teams[2]:
                i[1] += 3

    for i in table[teams[0]]:
        if i[0] == teams[1]:
            i[2].append(score_t1)
        if i[0] == teams[2]:
            i[2].append(score_t2)

    if last_group != teams[0] and last_group != "":
        for i in table[last_group]:
            i[2] = np.mean(i[2])
        final_table = sorted(table[last_group], key=itemgetter(1,2), reverse=True)
        advanced_group.append([final_table[0][0], final_table[1][0]])
    last_group = teams[0]

for i in table[last_group]:
    i[2] = np.mean(i[2])
final_table = sorted(table[last_group], key=itemgetter(1,2), reverse=True)
advanced_group.append([final_table[0][0], final_table[1][0]])

# --- 4. Playoffs ---
playoffs = {"Round of 16": [], "Quarter-Final": [], "Semi-Final": [], "Final": []}
control = [team for pair in advanced_group for team in pair]
playoffs["Round of 16"] = [[control[i], control[i+1]] for i in range(0, len(control), 2)]

def simulate_round(round_name, teams_list):
    next_round = []
    for game in teams_list:
        t1 = base2022_filtered[base2022_filtered["Team"] == game[0]]
        t2 = base2022_filtered[base2022_filtered["Team"] == game[1]]

        if t1.empty or t2.empty:
            continue

        score_t1 = t1[["offense", "defense", "midfield", "goalkeeper"]].sum(axis=1).values[0]
        score_t2 = t2[["offense", "defense", "midfield", "goalkeeper"]].sum(axis=1).values[0]

        winner = game[0] if score_t1 >= score_t2 else game[1]
        next_round.append(winner)
        game.append(winner)
    return next_round, teams_list

round_names = ["Round of 16", "Quarter-Final", "Semi-Final", "Final"]
for idx, r in enumerate(round_names):
    next_rounds, playoffs[r] = simulate_round(r, playoffs[r] if r=="Round of 16" else [[next_rounds[i], next_rounds[i+1]] for i in range(0, len(next_rounds), 2)])

# --- 5. Visualisation sans scores ---
plt.figure(figsize=(15, 10))
G = nx.balanced_tree(2, 3)
labels = []

for p in playoffs.keys():
    for game in playoffs[p]:
        label = f"{game[0].title()} vs {game[1].title()}"
        labels.append(label)

labels_dict = {i: labels[::-1][i] for i in range(len(G.nodes))}
pos = graphviz_layout(G, prog='twopi')
nx.draw(G, pos=pos, with_labels=False, node_color=range(len(G.nodes)), edge_color="#bbf5bb", width=5,
        font_weight='bold', cmap=plt.cm.Greens, node_size=4000)
nx.draw_networkx_labels(G, pos=pos, labels=labels_dict, font_size=10,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", lw=.5))
plt.axis('equal')
plt.title(f"Champion prédit : {next_rounds[0].title()}", fontsize=14, weight='bold')
plt.show()


# # Deployment

# In[ ]:


get_ipython().system('jupyter nbconvert --to python "FIFA world cup winner prediction.ipynb"')


# In[ ]:


get_ipython().system('pip freeze > requirements.txt')


# In[ ]:


get_ipython().system('python -m venv venv')
get_ipython().system('venv\\Scripts\\activate')


# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# --- CSS personnalisé pour un design moderne ---
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
    .stSelectbox, .stMultiselect {
        background-color: white;
        border-radius: 8px;
        padding: 5px;
    }
</style>
""", unsafe_allow_html=True)

# --- Chargement des données (commenté, décommentez si nécessaire) ---
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

# --- Filtrage des matchs (optionnel, décommentez si nécessaire) ---
# matches["Date"] = pd.to_datetime(matches["Date"])
# matches = matches[matches["Date"] < "2022-11-20"]

# --- Fonction pour générer le bracket ---
def draw_bracket(round16_winners, quarter_winners, semi_winners, final_winner):
    # Équipes qualifiées pour les huitièmes (hardcoded, mais modifiable)
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

    left_qf, right_qf = split_half(round16_winners)
    left_qf_pairs = make_pairs(left_qf)
    right_qf_pairs = make_pairs(right_qf)

    left_sf, right_sf = split_half(quarter_winners)
    left_sf_pair = make_pairs(left_sf)
    right_sf_pair = make_pairs(right_sf)

    final_pair = tuple(semi_winners)

    # Configuration graphique
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_facecolor("#e6f3fa")
    plt.axis("off")
    plt.title("Coupe du Monde 2022 — Bracket des Prédictions", fontsize=20, color="#1e3a8a", weight='bold')

    x_left = [0, 2.5, 5, 7.5]
    x_right = [15, 12.5, 10, 7.5]
    y_start = 16
    step = 2.5
    gold = "#FFD700"
    line_color = "#1e3a8a"

    def draw_side(matches, x, y_start, direction=1):
        y_positions = []
        for i, (t1, t2) in enumerate(matches):
            y1 = y_start - i * step * 2
            y2 = y1 - step
            mid = (y1 + y2) / 2
            ha = 'right' if direction == 1 else 'left'
            align_shift = -0.3 if direction == 1 else 0.3
            plt.text(x + align_shift, y1, t1.title(), ha=ha, va='center', fontsize=10, color="#1e3a8a", weight='bold')
            plt.text(x + align_shift, y2, t2.title(), ha=ha, va='center', fontsize=10, color="#1e3a8a", weight='bold')
            plt.plot([x, x + direction * 1.5], [y1, mid], color=line_color, lw=2)
            plt.plot([x, x + direction * 1.5], [y2, mid], color=line_color, lw=2)
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
    plt.plot([x_left[3], x_right[3]], [y_final_left[0], y_final_right[0]], color=line_color, lw=2)

    # Finale et vainqueur
    plt.text(7.5, y_final_left[0] + 0.7, f"{final_pair[0].title()}  vs  {final_pair[1].title()}", fontsize=12, ha='center', weight='bold', color="#dc2626")
    plt.text(7.5, y_final_left[0] - 1.5, f"Champion : {final_winner.title()}", fontsize=14, ha='center', weight='bold', color="#1e3a8a", bbox=dict(facecolor=gold, alpha=0.3, edgecolor=line_color))

    return fig

# --- Interface Streamlit ---
st.markdown('<div class="main">', unsafe_allow_html=True)
st.markdown('<div class="title">🏆 Prédiction de la Coupe du Monde FIFA 2022</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Personnalisez votre bracket et découvrez le champion !</div>', unsafe_allow_html=True)

# --- Widgets interactifs ---
st.markdown("### Personnalisez votre bracket")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Huitièmes de finale")
    round16_winners = st.multiselect(
        "Sélectionnez les équipes en quarts (8 équipes) :",
        options=[
            'senegal', 'netherlands', 'england', 'iran', 'argentina', 'poland',
            'france', 'denmark', 'spain', 'germany', 'belgium', 'croatia',
            'brazil', 'serbia', 'uruguay', 'south korea'
        ],
        default=['iran', 'netherlands', 'denmark', 'france', 'croatia', 'germany', 'south korea', 'serbia'],
        key="round16"
    )

with col2:
    st.subheader("Quarts de finale")
    quarter_winners = st.multiselect(
        "Sélectionnez les équipes en demi-finales (4 équipes) :",
        options=round16_winners if round16_winners else ['iran', 'netherlands', 'denmark', 'france', 'croatia', 'germany', 'south korea', 'serbia'],
        default=['netherlands', 'france', 'germany', 'serbia'],
        key="quarter"
    )

st.subheader("Demi-finales et Finale")
col3, col4 = st.columns(2)
with col3:
    semi_winners = st.multiselect(
        "Sélectionnez les finalistes (2 équipes) :",
        options=quarter_winners if quarter_winners else ['netherlands', 'france', 'germany', 'serbia'],
        default=['france', 'serbia'],
        key="semi"
    )

with col4:
    final_winner = st.selectbox(
        "Sélectionnez le champion :",
        options=semi_winners if semi_winners else ['france', 'serbia'],
        key="final"
    )

# --- Validation des sélections ---
if len(round16_winners) != 8:
    st.warning("Veuillez sélectionner exactement 8 équipes pour les huitièmes de finale.")
elif len(quarter_winners) != 4:
    st.warning("Veuillez sélectionner exactement 4 équipes pour les quarts de finale.")
elif len(semi_winners) != 2:
    st.warning("Veuillez sélectionner exactement 2 équipes pour les demi-finales.")
else:
    # Afficher le bracket
    st.markdown("### Votre Bracket Personnalisé")
    fig = draw_bracket(round16_winners, quarter_winners, semi_winners, final_winner)
    st.pyplot(fig)

# --- Explication pour le projet noté ---
st.markdown("""
### Méthodologie des Prédictions
Ce bracket est basé sur une analyse des performances historiques des équipes, des classements FIFA, et des statistiques des matchs jusqu'au 20 novembre 2022.  
Pour ce projet, les résultats sont simulés pour démontrer l'interactivité. Un modèle d'apprentissage automatique (ex. : régression logistique sur les données des matchs) pourrait être intégré pour générer des prédictions dynamiques basées sur les fichiers CSV fournis.  
**Interactivité** : Modifiez les équipes à chaque tour pour explorer différents scénarios et visualiser votre propre bracket !
""")

# --- Bouton pour détails ---
if st.button("Détails des équipes initiales"):
    st.markdown("""
    **Équipes initialement qualifiées pour les huitièmes de finale :**  
    Senegal, Netherlands, England, Iran, Argentina, Poland, France, Denmark,  
    Spain, Germany, Belgium, Croatia, Brazil, Serbia, Uruguay, South Korea  
    **Personnalisation** : Utilisez les menus déroulants pour créer votre propre bracket et prédire le champion !
    """)

st.markdown('</div>', unsafe_allow_html=True)


# In[ ]:


import pycountry
print(pycountry.countries.search_fuzzy('france')[0].name)  # Should print "France"


# In[ ]:




