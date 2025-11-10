import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 🔧 Configuration de la page
# -----------------------------
st.set_page_config(page_title="Indicateur de domination - Projet Foot", layout="centered")
st.title("⚽ Indicateur de domination – Football Analytics ⚽")

st.markdown("""
Ce projet vise à **quantifier la domination d'une équipe pendant un match** à partir des événements de jeu. Ce projet s’appuie sur les données disponibles au lien suivant : https://github.com/metrica-sports/sample-data.

L'idée est de créer un **indicateur de domination** basé sur plusieurs critères :
- **Durée des possessions** (plus une équipe garde le ballon longtemps, plus elle domine),
- **Position moyenne sur le terrain** (plus proche du but adverse = plus de danger),
- **Nombre de passes réussies** (fluidité du jeu),
- **Nombre de tirs** (menace offensive).
            
À noter, ces indicateurs ont été choisis car ils sont de bons indicateurs de la domination d’une équipe mais selon les staffs, la définition de la domination peut varier donc cet indicateur peut être modulé en fonction des besoins.

L'indicateur peut ensuite être agrégé par tranches de temps (1, 5, 10, 15 ou 30 minutes) pour visualiser 
l’évolution du **momentum** d’une équipe au fil du match.

Les barres :
- En **violet**, la domination de l’équipe à domicile (Home),
- En **rose**, celle de l’équipe à l’extérieur (Away).

Les étoiles rouges indiquent les **buts**.
""")


# -----------------------------
# 🎯 Sélection du match
# -----------------------------
match_choice = st.selectbox(
    "Choisir un match :",
    ["Match 1", "Match 2"]
)

file_map = {
    "Match 1": "Sample_Game_1_RawEventsData.csv",
    "Match 2": "Sample_Game_2_RawEventsData.csv"
}

match_file = file_map[match_choice]

# -----------------------------
# ⏱️ Sélection du découpage temporel
# -----------------------------
segmentation = st.selectbox(
    "Choisir une domination par fenêtre temporelle (en minutes) :",
    [1, 5, 10, 15, 30],
    index=0
)

# -----------------------------
# 📂 Lecture du jeu de données
# -----------------------------
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

match_info = load_data(match_file)

# -----------------------------
# 🕒 Création de la colonne minute
# -----------------------------
match_info['minute'] = (match_info['Start Time [s]'] // 60).astype(int)

shots = match_info[match_info['Type'] == 'SHOT']

goals = shots[shots['Subtype'].str.contains('GOAL', na=False)]

# -----------------------------
# 🧮 Calcul du score de domination
# -----------------------------
def compute_domination(df, segmentation):
    df['minute'] = (df['Start Time [s]'] // 60).astype(int)
    df['segment'] = (df['minute'] // segmentation).astype(int)

    # Détection des possessions
    df['possession_id'] = (df['Team'] != df['Team'].shift(1)).cumsum()

    possession_info = (
        df.groupby(['Team', 'possession_id'])
        .agg(
            mean_x=('Start X', 'mean'),
            period=('Period', 'first'),
            minute=('minute', 'first'),
            segment=('segment', 'first'),
            duration=('End Time [s]', lambda x: x.max() - x.min()),
            n_passes=('Type', lambda x: (x == 'PASS').sum()),
            n_shots=('Type', lambda x: (x == 'SHOT').sum())
        )
        .reset_index()
    )

    # --- inversion selon période ---
    possession_info['x_corrected'] = possession_info.apply(
        lambda row: 1 - row['mean_x'] if (row['Team'] == 'Home' and row['period'] == 2) or 
                                         (row['Team'] == 'Away' and row['period'] == 1)
        else row['mean_x'],
        axis=1
    )

    # --- poids selon la zone ---
    def zone_weight(x):
        if x < 0.33:
            return 1
        elif x < 0.66:
            return 2
        else:
            return 3

    possession_info['zone_score'] = possession_info['x_corrected'].apply(zone_weight)

    # --- durée pondérée (tranches de 10s, max 5) ---
    possession_info['duration_weight'] = (possession_info['duration'] // 10 + 1).clip(upper=5)

    # --- score offensif ---
    possession_info['offensive_score'] = (
        possession_info['n_shots'] * 4 +
        possession_info['n_passes'] * 1
    )

    # --- score total ---
    possession_info['possession_strength'] = (
        possession_info['zone_score'] * possession_info['duration_weight'] * 2 +
        possession_info['offensive_score'] * 1.5
    )

    # --- regroupement par segment temporel ---
    segment_domination = (
        possession_info
        .groupby(['segment', 'Team'])
        .agg(domination_score=('possession_strength', 'sum'))
        .reset_index()
    )

    pivot = segment_domination.pivot(index='segment', columns='Team', values='domination_score').fillna(0)
    home_team, away_team = pivot.columns[0], pivot.columns[1]

    pivot['momentum'] = (pivot[home_team] - pivot[away_team]) / (pivot[home_team] + pivot[away_team] + 1e-6)
    
    # --- renommer l’index pour affichage clair ---
    pivot.index = pivot.index * segmentation
    pivot.index.name = f"Segment ({segmentation} min)"
    
    return pivot


domination = compute_domination(match_info, segmentation)

# -----------------------------
# 📊 BARPLOT : moyenne du momentum par segment
# -----------------------------
momentum_df = domination[['momentum']].copy()
momentum_df['minute'] = momentum_df.index
momentum_df = momentum_df.reset_index(drop=True)

# déterminer les équipes
teams = match_info['Team'].unique()
home_team = [t for t in teams if t.lower() == 'home'][0]
away_team = [t for t in teams if t.lower() == 'away'][0]

# créer la colonne segment
momentum_df['segment_start'] = (momentum_df['minute'] // segmentation) * segmentation

# moyenne du momentum par segment
segment_avg = momentum_df.groupby('segment_start', as_index=False)['momentum'].mean()

# préparation du barplot
x = segment_avg['segment_start'].values
y = segment_avg['momentum'].values
bar_width = segmentation

fig, ax = plt.subplots(figsize=(14, 6))

y_pos = np.where(y >= 0, y, 0)
y_neg = np.where(y < 0, y, 0)

ax.bar(x, y_pos, width=bar_width, color='#7B1FA2', align='edge', zorder=2, label=home_team)
ax.bar(x, y_neg, width=bar_width, color='#EC407A', align='edge', zorder=2, label=away_team)

# ligne d’équilibre
ax.axhline(0, color='black', linewidth=1.2, zorder=3)

# affichage des buts (à la vraie minute)
goals = shots[shots['Subtype'].str.contains('GOAL', na=False)]
for _, g in goals.iterrows():
    goal_min = g['Start Time [s]'] / 60  # minute réelle du but
    x_star = goal_min
    team = g['Team']
    y_star = 0.95 if team == home_team else -0.95
    ax.plot([x_star, x_star], [0, y_star], color='black', linewidth=1.5, zorder=4)
    ax.scatter(x_star, y_star, color='red', edgecolors='black', s=200, marker='*', zorder=5)

# mise en forme
ax.set_title("Indicateur de domination du match - Momentum", fontsize=16, pad=15)
ax.set_xlabel(f"Minute")
ax.set_xlim(0, match_info['minute'].max() + 1)
ax.set_ylim(-1, 1)
ax.set_yticks([])
ax.set_ylabel("")
ax.grid(alpha=0.3, axis='y', zorder=1)
ax.legend(loc='upper right', frameon=False)

st.pyplot(fig)



