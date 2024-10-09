import streamlit as st
import pandas as pd
import numpy as np
import smtplib
import datetime
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Configuration initiale de la page - doit être la première ligne de commande Streamlit utilisée.
st.set_page_config(layout="wide")

# Fonction pour charger les délimiteurs à partir d'un fichier texte
def load_delimiters(file_path):
    delimiters = {}
    with open(file_path, 'r') as file:
        for line in file:
            # Suppression des espaces et sauts de ligne
            line = line.strip()
            if line:
                # Tenter de séparer le délimiteur d'affichage et le délimiteur réel
                parts = line.split('\t')
                if len(parts) == 2:
                    display_delim, actual_delim = parts
                    delimiters[display_delim] = actual_delim
                else:
                    print(f"Attention : la ligne suivante est incorrecte et sera ignorée : {line}")
    return delimiters

# Chemin d'accès au fichier delimiters.txt
file_path = os.path.join(os.getcwd(), "delimiters.txt")

# Charger les délimiteurs depuis le fichier
DELIMITERS = load_delimiters(file_path)

# Fonction pour charger le fichier CSS externe
def load_css(file_path):
    if os.path.exists(file_path):
        with open(file_path) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.error(f"Le fichier {file_path} est introuvable.")

def main():
    # Chemin complet du fichier CSS
    css_file_path = os.path.join(os.getcwd(), "styles.css")

    # Charger le CSS
    load_css(css_file_path)

    # Initialisation de l'état de la session
    if 'page' not in st.session_state:
        st.session_state.page = "Accueil"
        st.session_state.upload_status = {
            'principal': False,
            'statutory': False,
            'a_nouveaux': False
        }
        st.session_state.dataframes = {
            'principal': [],
            'statutory': [],
            'a_nouveaux': []
        }
        st.session_state.compiled_df = pd.DataFrame()  # Pour stocker les données compilées
        st.session_state.data_prepared = False  # Indique si les données sont prêtes pour l'analyse
        st.session_state.analyze_triggered = False  # Pour indiquer si l'analyse est lancée
        st.session_state.analyze_dates = False  # Pour indiquer si l'analyse des dates est lancée

    # Barre de navigation dans la sidebar
    st.sidebar.markdown(" ")
    st.sidebar.markdown(" ")

    # Boutons pour la navigation
    if st.sidebar.button("Accueil"):
        st.session_state.page = "Accueil"
    if st.sidebar.button("Charger mes données"):
        st.session_state.page = "Charger mes données"
    if st.sidebar.button("Préparer mes données"):
        st.session_state.page = "Préparer mes données"
    if st.sidebar.button("Analyser mes données"):
        st.session_state.page = "Analyser mes données"
    if st.sidebar.button("Contacter l'équipe data"):
        st.session_state.page = "Contacter l'équipe data"
    
    # Bouton "Réinitialiser les données" avec le style personnalisé
    if st.sidebar.button("Réinitialiser les données", key="reset", help="Supprimer toutes les données chargées et compilées", use_container_width=True):
        reset_data()

    # Afficher la page sélectionnée
    display_page()

def display_page():
    pages = {
        "Accueil": show_home,
        "Charger mes données": show_data_upload,
        "Préparer mes données": show_prepare_data,
        "Analyser mes données": show_analyze_data,
        "Contacter l'équipe data": show_contact_page
    }
    if st.session_state.page in pages:
        pages[st.session_state.page]()  # Appel de la fonction correspondante

def show_home():
    st.markdown("""
    <div style="background-color: #00323c; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 0;">
        <h2 style="color: white; font-size: 1.5em; margin: 0;">Page d'accueil</h2>
    </div>
    """, unsafe_allow_html=True)

    file_path = "/app/FECTORY/texte de présentation.txt"

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            home_text = file.read()
            st.markdown(home_text, unsafe_allow_html=True)
    except FileNotFoundError:
        st.error("Le fichier de texte pour la page d'accueil est introuvable.")
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la lecture du fichier: {e}")

def show_data_upload():
    st.markdown("""
    <div style="background-color: #00323c; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 0;">
        <h2 style="color: white; font-size: 1.5em; margin: 0;">Charger mes données</h2>
    </div>
    """, unsafe_allow_html=True)

    st.write("Cette section vous permet de charger vos fichiers de données pour les analyser.")

    # Charger les fichiers principaux
    file_upload('principal')

    st.markdown(" ")

    # Question 1: Souhaitez-vous ajouter des écritures statutaires ?
    st.markdown('<p style="font-size: 1.25em; font-weight: bold; color: #FFFFFF; margin: 0;">Souhaitez-vous ajouter des écritures statutaires ?</p>', unsafe_allow_html=True)
    add_statutory_entries = st.radio("", ("Non", "Oui"), key='statutory')
    if add_statutory_entries == "Oui":
        file_upload('statutory')

    st.markdown(" ")

    # Question 2: Souhaitez-vous ajouter des a-nouveaux ?
    st.markdown('<p style="font-size: 1.25em; font-weight: bold; color: #FFFFFF; margin: 0;">Souhaitez-vous ajouter des a-nouveaux ?</p>', unsafe_allow_html=True)
    add_a_nouveaux = st.radio("", ("Non", "Oui"), key='a_nouveaux')
    if add_a_nouveaux == "Oui":
        file_upload('a_nouveaux')

    # Bouton pour compiler les données
    if st.session_state.upload_status['principal'] or st.session_state.upload_status['statutory'] or st.session_state.upload_status['a_nouveaux']:
        if st.button("Compiler les données"):
            compile_data()

def file_upload(category):
    delimiter = st.selectbox(f"Choisissez le délimiteur pour {category}", options=["\\t (tabulation)", "| (pipe)", "; (semicolon)", ", (comma)"], key=f'delimiter_{category}')
    delimiter = DELIMITERS[delimiter.split()[0]]

    st.markdown(" ")

    uploaded_files = st.file_uploader(f"Uploader vos fichiers {category}", type=['txt', 'csv', 'xlsx', 'xls'], accept_multiple_files=True, key=f'uploader_{category}')
    
    if uploaded_files:
        # Réinitialiser la liste de DataFrames pour la catégorie actuelle pour éviter les duplications
        st.session_state.dataframes[category] = []
        
        for file in uploaded_files:
            df = load_file(file, delimiter)
            if df is not None:
                st.session_state.dataframes[category].append(df)
                st.write(f"Aperçu des données ({category}):")
                st.dataframe(df)
        st.session_state.upload_status[category] = True
        st.success(f"Fichiers {category} chargés avec succès!")

@st.cache_data
def load_file(file, delimiter):
    file_type = file.name.split('.')[-1]
    encodings = ['utf-8', 'latin1', 'iso-8859-1', 'cp1252']
    if file_type in ['csv', 'txt']:
        for enc in encodings:
            try:
                df = pd.read_csv(file, delimiter=delimiter, encoding=enc)
                if df.empty:
                    st.error("Le fichier chargé est vide.")
                    return None
                return df
            except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError):
                continue
        st.error("Impossible de lire le fichier avec les encodages courants")
        return None
    elif file_type in ['xlsx', 'xls']:
        try:
            df = pd.read_excel(file)
            if df.empty:
                st.error("Le fichier Excel chargé est vide.")
                return None
            return df
        except Exception as e:
            st.error(f"Erreur lors de la lecture du fichier Excel : {e}")
            return None
    else:
        st.error("Format de fichier non supporté")
        return None

def compile_data():
    compiled_df = pd.DataFrame()
    metadata = []
    for category in ['principal', 'statutory', 'a_nouveaux']:
        for idx, df in enumerate(st.session_state.dataframes[category]):
            compiled_df = pd.concat([compiled_df, df], ignore_index=True, sort=False)
            metadata.append({
                "Catégorie": category,
                "Index": idx + 1,
                "Nombre de lignes": df.shape[0],
                "Nombre de colonnes": df.shape[1]
            })
    
    st.session_state.compiled_df = compiled_df  # Sauvegarder les données compilées dans l'état de session
    
    st.write(f"Total lignes compilées: {compiled_df.shape[0]}, Colonnes: {compiled_df.shape[1]}")
    st.write("Données compilées :")
    st.dataframe(compiled_df)

    # Afficher les métadonnées
    st.markdown("### Métadonnées des fichiers chargés")
    for data in metadata:
        st.markdown(f"**{data['Catégorie'].capitalize()} {data['Index']}**: {data['Nombre de lignes']} lignes, {data['Nombre de colonnes']} colonnes")
    
    # Métadonnées du fichier compilé
    st.markdown("### Métadonnées du fichier compilé")
    st.markdown(f"**Fichier compilé**: {compiled_df.shape[0]} lignes, {compiled_df.shape[1]} colonnes")

    # Réinitialiser les états
    st.session_state.upload_status = {
        'principal': False,
        'statutory': False,
        'a_nouveaux': False
    }
    st.session_state.dataframes = {
        'principal': [],
        'statutory': [],
        'a_nouveaux': []
    }

def display_format_matrix(df, title):
    # Chemin vers le fichier texte contenant les formats de colonnes
    file_path = os.path.join(os.getcwd(), "format colonne.txt")

    # Lire le fichier texte pour obtenir les formats attendus
    try:
        format_df = pd.read_csv(file_path)
        expected_formats = dict(zip(format_df['ColumnName'], format_df['ExpectedFormat']))
    except FileNotFoundError:
        st.error(f"Le fichier {file_path} est introuvable.")
        return
    except Exception as e:
        st.error(f"Une erreur est survenue lors de la lecture du fichier: {e}")
        return

    data_to_display = []
    columns = df.columns.tolist()
    original_formats = ["Texte" if dtype == "object" else "Numérique" for dtype in df.dtypes.tolist()]
    expected_formats_list = [expected_formats.get(col, 'Non spécifié') for col in columns]
    verification = ['OK' if original == expected else 'KO' for original, expected in zip(original_formats, expected_formats_list)]
    data_to_display.extend(zip(columns, original_formats, expected_formats_list, verification))

    df_display = pd.DataFrame(data_to_display, columns=["Intitulé des colonnes", "Format original des colonnes", "Format attendus des colonnes", "Vérification"])
    st.markdown(f"#### {title}")
    st.table(df_display.style.apply(lambda x: ["color: #FF2F00" if v == "KO" else "color: #00E32C" for v in x], subset=["Vérification"], axis=0))

def show_prepare_data():
    st.markdown(f"""
    <div style="background-color: #00323c; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 0;">
        <h2 style="color: white; font-size: 1.5em; margin: 0;">Préparer mes données</h2>
    </div>
    """, unsafe_allow_html=True)

    # Vérifier s'il y a des données compilées ou chargées
    if not st.session_state.compiled_df.empty or st.session_state.dataframes['principal'] or st.session_state.dataframes['statutory'] or st.session_state.dataframes['a_nouveaux']:
        st.session_state.data_prepared = True
        st.markdown('<p style="font-size: 1.25em; font-weight: bold; color: #FFFFFF; margin: 0;">Données Chargées</p>', unsafe_allow_html=True)

        selected_category = st.selectbox("Sélectionner une catégorie", options=['principal', 'statutory', 'a_nouveaux', 'données compilées'], key="category_selector")

        if selected_category != 'données compilées' and st.session_state.dataframes[selected_category]:
            df = st.session_state.dataframes[selected_category][0]
        elif selected_category == 'données compilées' and not st.session_state.compiled_df.empty:
            df = st.session_state.compiled_df
        else:
            st.write("Aucune donnée disponible pour cette catégorie.")
            return

        # Afficher la matrice des formats
        format_matrix_container = st.empty()
        with format_matrix_container:
            display_format_matrix(df, f"Affichage : {selected_category.capitalize()}")

        st.sidebar.subheader("Modifier le format des colonnes")
        selected_columns = st.sidebar.multiselect("Sélectionner les colonnes à modifier", options=df.columns.tolist())
        new_format = st.sidebar.selectbox("Nouveau format", options=["Texte", "Numérique", "Date (jj-mm-aaaa)"])
        
        if st.sidebar.button("Appliquer les modifications"):
            apply_format(df, selected_columns, new_format)
            st.session_state.compiled_df = df  # Sauvegarder les modifications dans le dataframe compilé
            st.success("Formats des colonnes modifiés avec succès!")
            # Mettre à jour la matrice des formats
            format_matrix_container.empty()
            display_format_matrix(df, f"Données {selected_category.capitalize()}")

        # Section mise à jour pour créer Débit, Crédit, Solde
        if 'Montant' in df.columns and 'Sens' in df.columns:
            st.sidebar.subheader("Créer Débit, Crédit et Solde")
            create_debit = st.sidebar.checkbox("Créer Débit")
            create_credit = st.sidebar.checkbox("Créer Crédit")
            create_solde = st.sidebar.checkbox("Créer Solde")

            if st.sidebar.button("Ajouter les colonnes sélectionnées"):
                if create_debit:
                    df['Débit'] = np.where(df['Sens'] == 'D', pd.to_numeric(df['Montant'], errors='coerce').fillna(0).round(2), 0.00)
                if create_credit:
                    df['Crédit'] = np.where(df['Sens'] == 'C', pd.to_numeric(df['Montant'], errors='coerce').fillna(0).round(2), 0.00)
                if create_solde:
                    # Assurez-vous que les colonnes Débit et Crédit existent avant de créer Solde
                    if 'Débit' not in df.columns:
                        df['Débit'] = np.where(df['Sens'] == 'D', pd.to_numeric(df['Montant'], errors='coerce').fillna(0).round(2), 0.00)
                    if 'Crédit' not in df.columns:
                        df['Crédit'] = np.where(df['Sens'] == 'C', pd.to_numeric(df['Montant'], errors='coerce').fillna(0).round(2), 0.00)
                    df['Solde'] = df['Débit'] - df['Crédit']

                st.session_state.compiled_df = df  # Sauvegarder les modifications dans le dataframe compilé
                st.success("Colonnes créées avec succès!")
                # Mettre à jour la matrice des formats
                format_matrix_container.empty()
                display_format_matrix(df, f"Données {selected_category.capitalize()}")

        # Nouvelle section pour créer seulement la colonne Solde si Débit et Crédit existent déjà
        if 'Debit' in df.columns and 'Credit' in df.columns:
            st.sidebar.subheader("Créer uniquement la colonne Solde")
            create_only_solde = st.sidebar.checkbox("Créer Solde uniquement")

            if st.sidebar.button("Créer colonne Solde", key='create_only_solde'):
                df['Solde'] = df['Debit'] - df['Credit']
                st.session_state.compiled_df = df  # Sauvegarder les modifications dans le dataframe compilé
                st.success("Colonne Solde créée avec succès!")
                # Mettre à jour la matrice des formats
                format_matrix_container.empty()
                display_format_matrix(df, f"Données {selected_category.capitalize()}")

        st.sidebar.subheader("Supprimer des colonnes")
        columns_to_delete = st.sidebar.multiselect("Sélectionner les colonnes à supprimer", options=df.columns.tolist())
        
        if st.sidebar.button("Supprimer les colonnes"):
            df.drop(columns=columns_to_delete, inplace=True)
            st.session_state.compiled_df = df  # Sauvegarder les modifications dans le dataframe compilé
            st.success("Colonnes supprimées avec succès!")
            # Mettre à jour la matrice des formats
            format_matrix_container.empty()
            display_format_matrix(df, f"Données {selected_category.capitalize()}")

        st.sidebar.subheader("Renommer les colonnes")
        selected_column_to_rename = st.sidebar.selectbox("Sélectionner la colonne à renommer", options=df.columns.tolist())
        new_column_name = st.sidebar.text_input("Nouveau nom de colonne")

        if st.sidebar.button("Renommer la colonne"):
            if new_column_name:
                df.rename(columns={selected_column_to_rename: new_column_name}, inplace=True)
                st.session_state.compiled_df = df  # Sauvegarder les modifications dans le dataframe compilé
                st.success(f"Colonne '{selected_column_to_rename}' renommée en '{new_column_name}' avec succès!")
                # Mettre à jour la matrice des formats
                format_matrix_container.empty()
                display_format_matrix(df, f"Données {selected_category.capitalize()}")
            else:
                st.sidebar.error("Veuillez entrer un nouveau nom pour la colonne.")
    else:
        st.session_state.data_prepared = False
        st.write("Aucune donnée compilée n'a été trouvée. Veuillez charger et compiler les données dans l'onglet 'Charger mes données'.")

def apply_format(df, columns, new_format):
    for column in columns:
        try:
            if new_format == "Texte":
                df[column] = df[column].astype(str)
            elif new_format == "Numérique":
                if df[column].dtype == 'object':
                    decimal_separator = df[column].str.extract(r'([\.,])').dropna().iloc[:, 0].unique()
                    if len(decimal_separator) == 1:
                        if decimal_separator[0] == ',':
                            df[column] = df[column].str.replace(',', '.')
                        df[column] = pd.to_numeric(df[column], errors='coerce').round(2)
                    elif len(decimal_separator) > 1:
                        st.warning(f"Il y a plus d'un séparateur décimal différent dans la colonne '{column}'. Veuillez vérifier les données.")
                    else:
                        df[column] = pd.to_numeric(df[column], errors='coerce').round(2)
                else:
                    st.warning(f"La colonne '{column}' n'est pas de type 'object'. Veuillez vérifier les données.")
            elif new_format == "Date (jj-mm-aaaa)":
                df[column] = pd.to_datetime(df[column], errors='coerce').dt.strftime('%d-%m-%Y')
        except ValueError as e:
            st.error(f"Impossible de convertir la colonne '{column}' en '{new_format}': {str(e)}")
            continue

def analyze_all_columns(df):
    """
    Analyse toutes les colonnes d'un DataFrame et affiche un graphique en barres empilées
    pour les données 'valid', 'NaN', 'None', 'vides', 'uniques', et 'doubles'.
    Affiche également les lignes en anomalie par colonne dans des volets fermés.
    """
    st.subheader("Analyse des Colonnes")

    # Initialisation des données pour le graphique
    column_data = {
        'column': [],
        'valid': [],
        'NaN': [],
        'None': [],
        'vides': [],
        'uniques': [],
        'doubles': [],
    }

    # Analyse de chaque colonne
    for column_name in df.columns:
        total_count = len(df)

        # Compter les différentes catégories de données
        nan_count = df[column_name].isna().sum()  # NaN values
        none_count = df[column_name].isnull().sum()  # None values (same as NaN in pandas)
        empty_count = df[column_name].astype(str).str.strip().eq('').sum()  # Empty strings
        unique_count = df[column_name].nunique(dropna=False)  # Unique values
        double_count = df[column_name].duplicated(keep=False).sum()  # Duplicated values

        # Compter les valeurs valides (non None, non NaN, non empty)
        valid_count = total_count - nan_count - empty_count

        # Ajouter les résultats à notre dictionnaire de données
        column_data['column'].append(column_name)
        column_data['valid'].append(valid_count)
        column_data['NaN'].append(nan_count)
        column_data['None'].append(none_count - nan_count)  # Pour éviter le double comptage
        column_data['vides'].append(empty_count)
        column_data['uniques'].append(unique_count)
        column_data['doubles'].append(double_count)

    # Création du graphique en barres empilées (vue horizontale)
    fig = go.Figure()

    # Ajouter les barres pour chaque type de données
    for anomaly_type in ['valid', 'NaN', 'None', 'vides', 'uniques', 'doubles']:
        fig.add_trace(go.Bar(
            x=column_data['column'],
            y=column_data[anomaly_type],
            name=anomaly_type,
            marker=dict(line=dict(color='rgba(255, 255, 255, 1.0)', width=2))
        ))

    # Configuration du layout du graphique avec un style adapté
    fig.update_layout(
        title={
            'text': 'Analyse des Colonnes du DataFrame',
            'y': 0.9,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(color='white')
        },
        xaxis=dict(
            title='Colonnes',
            titlefont=dict(size=14, color='white'),
            tickfont=dict(size=12, color='white')
        ),
        yaxis=dict(
            title='Nombre de lignes',
            titlefont=dict(size=14, color='white'),
            tickfont=dict(size=12, color='white')
        ),
        barmode='stack',
        plot_bgcolor='rgba(0, 0, 0, 0)',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        height=600,
        width=1000,
        margin=dict(l=40, r=40, t=100, b=80),
        legend=dict(
            title='Type de données',
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='white')
        ),
        bargap=0.15,
        bargroupgap=0.1
    )

    # Affichage du graphique dans Streamlit
    st.plotly_chart(fig, use_container_width=True)

    # Module pour afficher les lignes en anomalie par colonne dans des volets fermés
    st.subheader("Lignes en Anomalie par Colonne")

    for column_name in df.columns:
        # Afficher uniquement les colonnes qui ont des anomalies
        anomalies_in_column = df[df[column_name].isna() | df[column_name].astype(str).str.strip().eq('')]
        
        if not anomalies_in_column.empty:
            with st.expander(f"Anomalies dans la colonne {column_name} ({len(anomalies_in_column)} lignes)"):
                display_filtered_rows(column_name, df)

def display_filtered_rows(column_name, df):
    """
    Affiche les lignes du DataFrame df en fonction des anomalies de la colonne sélectionnée.
    """
    if column_name is None:
        return

    # Filtrer les données selon le type d'anomalie et afficher avec des expander
    nan_rows = df[df[column_name].isna()]
    if not nan_rows.empty:
        st.write(f"**NaN**: {len(nan_rows)} lignes")
        st.dataframe(nan_rows)

    none_rows = df[df[column_name].isnull() & ~df[column_name].isna()]
    if not none_rows.empty:
        st.write(f"**None**: {len(none_rows)} lignes")
        st.dataframe(none_rows)

    empty_rows = df[df[column_name].astype(str).str.strip().eq('')]
    if not empty_rows.empty:
        st.write(f"**Vides**: {len(empty_rows)} lignes")
        st.dataframe(empty_rows)

    unique_rows = df[df[column_name].duplicated(keep=False) == False]
    if not unique_rows.empty:
        st.write(f"**Uniques**: {len(unique_rows)} lignes")
        st.dataframe(unique_rows)

    double_rows = df[df[column_name].duplicated(keep=False)]
    if not double_rows.empty:
        st.write(f"**Doubles**: {len(double_rows)} lignes")
        st.dataframe(double_rows)

def aggregate_data(compiled_df):
    st.sidebar.subheader("Agrégation des données")

    aggregation_column = st.sidebar.selectbox("Sélectionner une colonne d'agrégation", options=compiled_df.columns.tolist())

    columns_to_aggregate = st.sidebar.multiselect("Sélectionner une ou plusieurs colonnes à agréger", options=compiled_df.select_dtypes(include=np.number).columns.tolist())

    return aggregation_column, columns_to_aggregate

def show_analyze_data():
    st.markdown(f"""
    <div style="background-color: #00323c; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 0;">
        <h2 style="color: white; font-size: 1.5em; margin: 0;">Analyser mes données</h2>
    </div>
    """, unsafe_allow_html=True)

    # Vérification des données préparées
    if not st.session_state.data_prepared:
        st.write("Aucune donnée n'a été chargée ou préparée pour l'analyse. Veuillez charger et compiler les données dans l'onglet 'Charger mes données'.")
        return

    st.markdown('<p style="font-size: 1.25em; font-weight: bold; color: #00323c; margin: 0;">Données disponibles pour analyse</p>', unsafe_allow_html=True)

    # Sélection des données à afficher
    selected_category = st.selectbox("Sélectionner une catégorie de données", options=['principal', 'statutory', 'a_nouveaux', 'données compilées'], key="analyze_category_selector")

    if selected_category != 'données compilées' and st.session_state.dataframes[selected_category]:
        df = st.session_state.dataframes[selected_category][0]
    elif selected_category == 'données compilées' and not st.session_state.compiled_df.empty:
        df = st.session_state.compiled_df
    else:
        st.write("Aucune donnée disponible pour cette catégorie.")
        return

    # Ajouter le titre et le bouton "Lancer l'analyse" dans le bandeau latéral sous "Métadonnées"
    st.sidebar.subheader("Métadonnées")
    if st.sidebar.button("Lancer l'analyse"):
        st.session_state.analyze_triggered = True  # Activer l'indicateur pour lancer l'analyse

    if st.session_state.analyze_triggered:
        analyze_all_columns(df)  # Lancer l'analyse pour toutes les colonnes

    # Appel de la fonction pour ajouter les widgets d'analyse de date
    analyze_date_columns(df)

    aggregation_column, columns_to_aggregate = aggregate_data(df)

    if aggregation_column and columns_to_aggregate:
        st.subheader("Balance générale")
        with st.container():
            pivot_table = pd.pivot_table(df, values=columns_to_aggregate, index=aggregation_column, aggfunc=np.sum)
            st.dataframe(pivot_table)

    st.sidebar.subheader("Calculs supplémentaires")

    # Nouvelle section pour calculer le bilan, le résultat et le solde total
    if st.sidebar.button("Calculer les soldes"):
        calculate_balances(df)

def calculate_balances(df):
    st.subheader("Calculs des Soldes")
    
    # Vérifier la présence des colonnes 'CompteNum', 'Solde' et 'Montant'
    if 'CompteNum' in df.columns and 'Solde' in df.columns:
        # Calculer le bilan
        bilan = df[df['CompteNum'].astype(str).str.startswith(('1', '2', '3', '4', '5'))]['Solde'].sum()
        # Calculer le résultat
        resultat = df[df['CompteNum'].astype(str).str.startswith(('6', '7'))]['Solde'].sum()

        st.write(f"**Bilan**: {bilan:,.2f}")
        st.write(f"**Résultat**: {resultat:,.2f}")
    
    # Calculer le solde total
    if 'Solde' in df.columns and 'Montant' in df.columns:
        total_balance = df['Solde'].sum() + df['Montant'].sum()
        st.write(f"**Solde total (Solde + Montant)**: {total_balance:,.2f}")
    elif 'Solde' in df.columns:
        total_balance = df['Solde'].sum()
        st.write(f"**Solde total (Solde)**: {total_balance:,.2f}")
    elif 'Montant' in df.columns:
        total_balance = df['Montant'].sum()
        st.write(f"**Solde total (Montant)**: {total_balance:,.2f}")
    else:
        st.error("Les colonnes 'Solde' et 'Montant' ne sont pas disponibles dans les données sélectionnées.")

def reset_data():
    st.session_state.dataframes = {
        'principal': [],
        'statutory': [],
        'a_nouveaux': []
    }
    st.session_state.compiled_df = pd.DataFrame()
    st.session_state.data_prepared = False  # Réinitialiser l'état de préparation des données
    st.session_state.analyze_triggered = False  # Réinitialiser l'indicateur d'analyse
    st.success("Toutes les données ont été réinitialisées.")

def show_contact_page():
    st.markdown(f"""
    <div style="background-color: #00323c; padding: 10px; border-radius: 10px; text-align: center; margin-bottom: 0;">
        <h2 style="color: white; font-size: 1.5em; margin: 0;">Contacter l'équipe data</h2>
    </div>
    """, unsafe_allow_html=True)

    email = st.text_input("Votre adresse email professionnelle")
    page = st.selectbox("Sélectionnez la page où vous rencontrez un problème", ["Charger mes données", "Préparer mes données", "Analyser mes données"])
    subject = st.selectbox("Sélectionnez le sujet de votre commentaire", ["Problème", "Suggestion"])
    message = st.text_area("Expliquez votre problème ou suggestion")

    if st.button("Envoyer"):
        send_email(email, page, subject, message)
        st.success("Votre message a été envoyé avec succès !")

def send_email(email, page, subject, message):
    sender_email = "your_email@example.com"
    receiver_email = "e.gilain@nexia-sa.fr"
    password = "your_password"

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = f"{subject} sur {page}"

    body = f"""
    Adresse email: {email}

    Page: {page}
    Sujet: {subject}

    Message:
    {message}
    """
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.example.com", 587)
        server.starttls()
        server.login(sender_email, password)
        text = msg.as_string()
        server.sendmail(sender_email, receiver_email, text)
        server.quit()
    except Exception as e:
        st.error(f"Erreur lors de l'envoi de l'email: {str(e)}")

if __name__ == "__main__":
    main()
