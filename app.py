import os
import joblib
import pandas as pd
from fpdf import FPDF
import streamlit as st
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

import tempfile
# Chemins de base
current_path = os.getcwd()
base_dir = os.path.dirname(current_path)

# Chargement des données
@st.cache_data
def load_data():
    data_path = os.path.join('/home/leon/dev/a_dev_bossGID/ia_dev/data/cleaned_customer_loyalty_data.csv')
    if not os.path.exists(data_path):
        st.error(f"Fichier de données introuvable à : {data_path}")
        return pd.DataFrame()
    return pd.read_csv(data_path)


# Chargement du modèle
@st.cache_resource
def load_model():
    models_path = os.path.join(base_dir, 'ia_dev/models/RandomForestClassifier_new_version.pkl')
    if not os.path.exists(models_path):
        st.error(f"Fichier du modèle introuvable à : {models_path}")
        return None
    return joblib.load(models_path)

# Fonction pour les clusters------------------------:06/08/25
def get_cluster_summary(df):
    # Étape 1 : calculer les agrégats de base
    summary = df.groupby("Cluster").agg(
        Nb_clients=("Age", "count"),
        Age_moyen=("Age", "mean"),
        Depense_moyenne=("Total_Spent", "mean"),
        Depense_min=("Total_Spent", "min"),
        Depense_max=("Total_Spent", "max")
    ).round(2)

    # Étape 2 : calculer le "meilleur groupe" selon ta formule
    summary["Score_segment"] = (
        summary["Depense_moyenne"] * summary["Nb_clients"]
        + summary["Depense_max"] * summary["Nb_clients"]
    ).round(2)

    # Étape 3 : réinitialiser l'index si besoin
    return summary.reset_index()


# Fonction depense par clusters------------------------:06/08/25
# Nouvelle fonction pour sauvegarder les graphiques en images
def save_cluster_plots(df):
    image_paths = []

    # Graph 1: Histogramme des dépenses
    plt.figure(figsize=(6, 4))
    sns.histplot(data=df, x="Total_Spent", hue="Cluster", multiple="stack", bins=30)
    plt.title("Distribution des dépenses par cluster")
    plt.xlabel("Total Spent ($)")
    plt.ylabel("Nombre de clients")
    hist_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.savefig(hist_path)
    plt.close()
    image_paths.append(hist_path)

    # Graph 2: Boxplot des âges
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Cluster", y="Age")
    plt.title("Répartition des âges par cluster")
    boxplot_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    plt.savefig(boxplot_path)
    plt.close()
    image_paths.append(boxplot_path)

    return image_paths



# Fonction pour déterminer l'action à partir de la proba
def determine_action(proba_fidel):
    if proba_fidel >= 0.8:
        return "Recompense premum"
    elif proba_fidel >= 0.5:
        return "Recompense special"
    elif proba_fidel >= 0.3:
        return "Livraison gratuit"
    else:
        return "Merci pour votre fidelité"

# Fonction de génération du PDF (améliorée)
def generate_pdf(df, full_data):
    from fpdf import FPDF
    pdf = FPDF()
    pdf.add_page()

    # En-tête
    pdf.set_font("Arial", "B", 12)
    pdf.cell(100, 10, "Smart Loyalty Office", ln=0, align="L")
    pdf.cell(w=0, h=10, txt=datetime.today().strftime("Date : %d/%m/%Y %H:%M:%S"), ln=True,align="R")
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, "Rapport de niveau de fidélité des clients", ln=1, align="C")
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(100, 100, 100)
    pdf.cell(0, 8, "Document confidentiel à usage interne uniquement", ln=1, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(10)

    # Tableau des prédictions
    col_widths = {
        'Age': 9, 'Gender': 14, 'Total_Spent':20, 'Quantity': 15,
        'Channel': 15, 'Category': 16, 'Cluster': 14, 'Avg_Price':18,
        'predict': 14, 'pbt_f': 10, 'pbt_nf': 14, 'action': 30
    }
    cell_height = 20
    pdf.set_font("Arial", 'B', 10)
    for col_name in df.columns:
        pdf.cell(col_widths.get(col_name, 5), cell_height, str(col_name), border=1, align='C')
    pdf.ln()
    pdf.set_font("Arial", 'I', 8)
    for _, row in df.iterrows():
        for col_name, item in row.items():
            text = str(round(item, 4)) if isinstance(item, float) else str(item)
            pdf.cell(col_widths.get(col_name, 5), cell_height, text, border=1, align='C')
        pdf.ln()

    pdf.ln(5)

    # --- Résumé des clusters ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Résumé par Cluster", ln=True,align="C")

    summary_df = get_cluster_summary(full_data)
    headers = ["Cluster", "Nb clients", "Âge moyen", "Dépense moyenne ($)", "Dépense min ($)","Dépense max ($)","Score_segment ($)"]
    col_widths = [18, 18, 24, 40, 30,30,30]

    pdf.set_font("Arial", "B", 10)
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], 10, header, 1, 0, 'C')
    pdf.ln()

    pdf.set_font("Arial", "", 9)
    for _, row in summary_df.iterrows():
        pdf.cell(col_widths[0], 10, str(int(row['Cluster'])), 1,align='C')
        pdf.cell(col_widths[1], 10, str(int(row['Nb_clients'])), 1,align='C')
        pdf.cell(col_widths[2], 10, str(row['Age_moyen']), 1,align='C')
        pdf.cell(col_widths[3], 10, str(row['Depense_moyenne']), 1,align='C')
        pdf.cell(col_widths[4], 10, str(row['Depense_min']), 1,align='C')
        pdf.cell(col_widths[4], 10, str(row['Depense_max']), 1,align='C')
        pdf.cell(col_widths[4], 10, str(row['Score_segment']), 1,align='C')
        pdf.ln()

    pdf.ln(5)

    # --- Recommandations ---
    # --- Recommandations dynamiques selon Score_segment ---
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Recommandations basées sur Score_segment", ln=True)
    pdf.set_font("Arial", "", 10)

# Trier le summary par Score_segment décroissant
    reco_summary = summary_df.sort_values(by="Score_segment", ascending=False).reset_index(drop=True)

# Liste des recommandations selon le rang
    recommandations = [
    "Offre VIP + Email ciblé",
    "Communication régulière",
    "Relance + Promotions",
    "Analyse comportementale recommandée",
    "Encourager à dépenser davantage"
    ]

# Boucle : associer chaque cluster à une reco
    for i, row in reco_summary.iterrows():
        cluster_id = int(row["Cluster"])
        reco = recommandations[i] if i < len(recommandations) else "Suivi personnalisé"
        pdf.cell(0, 8, f"Groupe {cluster_id} : {reco}", ln=True)


    pdf.ln(5)

    # --- Ajouter les graphiques ---
    image_paths = save_cluster_plots(full_data)
    for img_path in image_paths:
        pdf.image(img_path, w=170)
        pdf.ln(5)

    return pdf.output(dest="S").encode("latin-1")



# --- Application Streamlit ---
st.set_page_config(
    page_title="Classification de fidelité",
    page_icon="💸",
    layout="centered"
)

st.title("Classification de fidelité")
#st.markdown("---")


# Formulaire utilisateur
with st.form("loyalty_form"):
    st.header("Informations sur la fidelité")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        Age = st.number_input("Age (ans)", min_value=0,max_value=100,step=1)
        Gender = st.selectbox("Genre ?", ["Male", "Female"])

    with col2:
        Total_Spent = st.number_input("Total_Spent ($)", min_value=0,max_value=100000,step=10)
        Quantity = st.number_input("Quantity ?", min_value=0,max_value=1000,step=1)

    with col3:
        Channel = st.selectbox("Channel", ["Email", "Social", "In-Store", "Online"])
        Category = st.selectbox("Category ?", ["Bags", "Outerwear", "Accessories", "Footwear", "Clothing"])

    with col4:
        Cluster = st.selectbox("Cluster", [0, 1, 2, 3, 4])
        Avg_Price = st.number_input("Avg_Price ($)", min_value=0,max_value=100000,step=10)

    submit_button = st.form_submit_button("Analyser la Fidelité")

# Variables globales à utiliser après le formulaire
pdf_bytes = None

# Exécution après soumission du formulaire
if submit_button:
    if Total_Spent <= 0 or Quantity <= 0 or Avg_Price <= 0:
        st.warning("⚠️ Veuillez entrer des valeurs numériques positives")
    else:
        st.success("✅ Informations soumises. Analyse en cours...")
        input_data = pd.DataFrame({
                'Age': [Age],
                'Gender': [Gender],
                'Total_Spent': [Total_Spent],
                'Quantity': [Quantity],
                'Channel': [Channel],
                'Category': [Category],
                'Cluster': [Cluster],
                'Avg_Price': [Avg_Price]
                })

        model = load_model()
        if model is not None:
            prediction = model.predict(input_data)[0]
            proba = model.predict_proba(input_data)[0]

            label = "fidel" if prediction == 1 else "non_fidel"
            proba_fidel = round(proba[1],4)
            proba_non_fidel = round(proba[0],4)
            action = determine_action(proba_fidel)

            st.subheader("Resultat d'analyse")
            if prediction == 1:
                st.error(f"**Statut:** {label}")
            else:
                st.success(f"**Statut:** {label}")
            st.markdown(f"**Probabilité de Fidelité :** `{proba_fidel * 100:.2f}%`")
            st.markdown(f"**Probabilité de non Fidelité :** `{proba_non_fidel * 100:.2f}%`")
            st.markdown(f"**Action recommandé:** **{action}**")

            # Résultats dans DataFrame
            result_df = input_data.copy()
            result_df['prediction'] = [label]
            result_df['probabilité_fidel'] = [proba_fidel]
            result_df['probabilité_non_fidel'] = [proba_non_fidel]
            result_df['action'] = [action]

            st.markdown("---")
            st.subheader("Détails de la Classification de Fidelité")
            st.dataframe(result_df.style.highlight_max(axis=1, subset=['probabilité_fidel'], color='red')
            .highlight_min(axis=1, subset=['probabilité_non_fidel'], color='lightgreen'))

            # Préparer PDF
            pdf_df = result_df.copy()
            pdf_df.columns = [
                "Age", "Gender", "Total_Spent",
                "Quantity","Channel", "Category", "Cluster","Avg_Price",
                "predict", "pbt_f", "pbt_nf", "action"
            ]
            full_data = load_data()  # Pour passer tout le dataset
            pdf_bytes = generate_pdf(pdf_df, full_data)

        else:
            st.error("Le modèle de classification n'a pas pu être chargé. Veuillez vérifier le chemin d'accès.")

# Affichage du bouton de téléchargement APRÈS la soumission du formulaire
if pdf_bytes:
    st.download_button(
        "📥 Télécharger le Rapport PDF", 
        data=pdf_bytes, 
        file_name=f"rapport_fidelité_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", 
        mime="application/pdf"
    )



# --- Section Analyse des clusters ---
st.header("📊 Visualisation des données par Cluster")

df = load_data()


# Sidebar - Dataset
st.sidebar.title("Informations sur le Dataset")
if not df.empty:
    if st.sidebar.checkbox("Afficher un extrait du Dataset"):
        st.subheader("Extrait du jeu de données original")
        st.write(df.sample(n=5, random_state=10))
        st.write(df.sample(n=5, random_state=0))
        st.write("Dimensions du dataset:", df.shape)

    if st.sidebar.checkbox("Afficher les statistiques par cluster"):
        st.subheader("Résumé statistique par cluster")
        st.dataframe(get_cluster_summary(df))

  

