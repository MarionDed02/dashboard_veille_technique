import pandas as pd
import numpy as np
import seaborn as sns
import pickle
import lightgbm as lgb
import streamlit as st
import requests
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Rectangle, Circle
from matplotlib import cm
from matplotlib.patches import Arc

# Fonctions relatives à la jauge de risque du client
def degree_range(n):
    start = np.linspace(0,180,n+1, endpoint=True)[0:-1]
    end = np.linspace(0,180,n+1, endpoint=True)[1::]
    mid_points = start + ((end-start)/2.)
    return np.c_[start, end], mid_points


def rot_text(ang):
    rotation = np.degrees(np.radians(ang) * np.pi / np.pi - np.radians(90))
    return rotation


def gauge(arrow=0.3, labels=['Faible', 'Moyen', 'Elevé', 'Très élevé'],
          title='', min_val=0, max_val=100, threshold=-1.0,
          colors='RdYlGn_r', n_colors=-1, ax=None, figsize=(3, 2)):
    N = len(labels)
    n_colors = n_colors if n_colors > 0 else N
    if isinstance(colors, str):
        cmap = cm.get_cmap(colors, n_colors)
        cmap = cmap(np.arange(n_colors))
        colors = cmap[::-1]
    if isinstance(colors, list):
        n_colors = len(colors)
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    ang_range, _ = degree_range(n_colors)

    for ang, c in zip(ang_range, colors):
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor='w', lw=2, alpha=0.5))
        ax.add_patch(Wedge((0., 0.), .4, *ang, width=0.10, facecolor=c, lw=0, alpha=0.8))

    _, mid_points = degree_range(N)
    labels = labels[::-1]
    a = 0.45
    for mid, lab in zip(mid_points, labels):
        ax.text(a * np.cos(np.radians(mid)), a * np.sin(np.radians(mid)), lab, \
                horizontalalignment='center', verticalalignment='center', fontsize=8, \
                fontweight='bold', rotation=rot_text(mid))

    ax.add_patch(Rectangle((-0.4, -0.1), 0.8, 0.1, facecolor='w', lw=2))
    ax.text(0, -0.10, title, horizontalalignment='center', verticalalignment='center', fontsize=20, fontweight='bold')

    # Seuils de la jauge
    if threshold > min_val and threshold < max_val:
        pos = 180 * (max_val - threshold) / (max_val - min_val)
        a = 0.25;
        b = 0.18;
        x = np.cos(np.radians(pos));
        y = np.sin(np.radians(pos))
        ax.arrow(a * x, a * y, b * x, b * y, width=0.01, head_width=0.0, head_length=0, ls='--', fc='r', ec='r')

    # Flèche de la jauge
    pos = 180 - (180 * (max_val - arrow) / (max_val - min_val))
    pos_normalized = (arrow - min_val) / (max_val - min_val)
    angle_range = 180
    pos_degrees = angle_range * (1 - pos_normalized)

    ax.arrow(0, 0, 0.225 * np.cos(np.radians(pos_degrees)), 0.225 * np.sin(np.radians(pos_degrees)), \
             width=0.02, head_width=0.07, head_length=0.1, fc='k', ec='k')

    ax.add_patch(Circle((0, 0), radius=0.02, facecolor='k'))
    ax.set_frame_on(False)
    ax.axes.set_xticks([])
    ax.axes.set_yticks([])
    ax.axis('equal')
    return ax


# Paramètres Streamlit (v1.25)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_option('deprecation.showfileUploaderEncoding', False)

# Ajout logo
logo_path = 'logo.png'
st.sidebar.image(logo_path, use_column_width=True)

# Définition groupe de colonnes
categorical_columns = ['Type contrat',
                       'Genre',
                       'Diplôme etude supérieure',
                       'Ville travail différente ville_résidence']

# Charger le modèle à partir du fichier pickle
model_path = './saved_model/'
with open(model_path+'LightGBM_smote_tuned.pkl', 'rb') as f:
    model = pickle.load(f)

# Charger les dataframes du dashboard
df_feature_importance = pd.read_csv('df_feature_importance_25.csv')
df_feature_importance.drop('Unnamed: 0', axis=1, inplace=True)
df_dashboard_final = pd.read_csv('df_dashboard_final.csv')
df_dashboard_final.drop('Unnamed: 0', axis=1, inplace=True)

# Titre du dashboard
st.title("Analyse du risque de crédit client")

# Encadré sur la partie gauche
st.sidebar.title('Client à analyser')
selected_client = st.sidebar.selectbox("Choix de l'dentifiant client :", df_dashboard_final['ID client'])
predict_button = st.sidebar.button("Lancer l'analyser")

# Obtention de l'index correspondant à l'ID client sélectionné
index = df_dashboard_final[df_dashboard_final['ID client'] == selected_client].index[0]

# Affichage des informations du client sélectionné
client_info = df_dashboard_final[df_dashboard_final['ID client'] == selected_client]
st.subheader('Informations sur le client :')
client_info.index = client_info['ID client']
st.write(client_info[['Prédiction crédit', 'Score client (sur 100)', 
                      'Type contrat', 'Genre', 'Âge',"Nombre d'enfants",'Revenu annuel total']])

# Obtention de la catégorie de prédiction crédit du client sélectionné
selected_client_cat = df_dashboard_final.loc[index, 'Prédiction crédit']

# DataFrame contenant la même catégorie de prédiction crédit que le client sélectionné
df_customer = df_dashboard_final[df_dashboard_final['Prédiction crédit'] == selected_client_cat].copy()

# Affichage de la jauge score client
st.subheader('Evaluation du risque de crédit :')
score = client_info['Score client (sur 100)'].values[0]  # Récupérer le score du client
fig, ax = plt.subplots(figsize=(5, 3))
gauge(arrow=score, ax=ax)  # Appeler la fonction gauge() en passant le score du client
st.pyplot(fig)

# Partie droite de l'écran
st.sidebar.title('Personnalisation des graphiques')
univariate_options = [col for col in df_dashboard_final.columns if col not in ['ID client', 'Prédiction crédit']]
bivariate_options = [col for col in df_dashboard_final.columns if col not in ['ID client', 'Prédiction crédit']]

# Graphique univarié
univariate_feature = st.sidebar.selectbox('Choix de la variable univariée :', univariate_options)
df_customer.replace([np.inf, -np.inf], 0, inplace=True)
st.subheader('Analyse univariée :')
plt.figure()
plt.hist(df_customer[univariate_feature], color='lightgreen', label='Population')
plt.xlabel(univariate_feature)
plt.axvline(client_info[univariate_feature].values[0], color='red', linestyle='--', label='Client sélectionné')
plt.legend()
st.pyplot(plt.gcf())

# Graphique bivarié
bivariate_feature1 = st.sidebar.selectbox('Analyse bivariée - Choix de la variable 1 :', bivariate_options)
bivariate_feature2 = st.sidebar.selectbox('Analyse bivariée - Choix de la variable 2 :', bivariate_options)
st.subheader('Analyse bivariée :')
plt.figure()
sns.scatterplot(data=df_dashboard_final, x=bivariate_feature1, y=bivariate_feature2,
                c=df_dashboard_final['Score client'], cmap='viridis',
                alpha=0.5, label='Population')
sns.scatterplot(data=client_info, x=bivariate_feature1, y=bivariate_feature2,
                color='red', marker='o', s=100, label='Client sélectionné')
plt.xlabel(bivariate_feature1)
plt.ylabel(bivariate_feature2)
plt.legend()
st.pyplot(plt.gcf())

# Graphique feature importance globale
df_sorted = df_feature_importance.sort_values('Features_importance_shapley', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Features_importance_shapley', y='Features', data=df_sorted, color='lightgreen')
plt.xlabel('Importance SHAP')
plt.ylabel('Variables')
st.subheader('Importance globale des variables :')
st.pyplot(plt.gcf())

