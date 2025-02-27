import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns

# Seitenkonfiguration
st.set_page_config(page_title="Entscheidungsbaum Creator", layout="wide")

# Titel und Einführung
st.title("Interaktiver Entscheidungsbaum Creator")
st.markdown("""
Mit dieser App kannst du Schritt für Schritt einen Entscheidungsbaum erstellen,
die Schwellenwerte (Thresholds) für die Knoten anpassen und die Ergebnisse sofort visualisieren.
""")
def predict_with_manual_rules(X, rules):
    predictions = np.zeros(len(X))
    
    # Standardklasse ist 0, wenn keine Regeln zutreffen
    for idx, row in X.reset_index(drop=True).iterrows():
        for rule_idx, (feature, operator, threshold) in enumerate(rules):
            if operator == "<" and row[feature] < threshold:
                predictions[idx] = 1
                break
            elif operator == ">" and row[feature] > threshold:
                predictions[idx] = 1
                break
            elif operator == "<=" and row[feature] <= threshold:
                predictions[idx] = 1
                break
            elif operator == ">=" and row[feature] >= threshold:
                predictions[idx] = 1
                break
            elif operator == "==" and row[feature] == threshold:
                predictions[idx] = 1
                break
    
    return predictions
# Seitenleiste für Dateneingabe und globale Parameter
with st.sidebar:
    st.header("Daten und Parameter")
    
    # Option zum Hochladen eigener Daten oder Verwenden von Beispieldaten
    data_option = st.radio(
        "Datenquelle wählen:",
        ["Beispieldaten verwenden", "Eigene Daten hochladen"]
    )
    
    if data_option == "Eigene Daten hochladen":
        uploaded_file = st.file_uploader("CSV-Datei hochladen", type=["csv"])
        if uploaded_file is not None:
            data = pd.read_csv(uploaded_file)
            st.success("Daten erfolgreich geladen!")
        else:
            st.info("Bitte lade eine CSV-Datei hoch oder wähle Beispieldaten.")
            # Fallback auf Beispieldaten, wenn keine Datei hochgeladen wurde
            data_option = "Beispieldaten verwenden"
    
    if data_option == "Beispieldaten verwenden":
        # Beispieldaten generieren
        np.random.seed(42)
        n_samples = 200
        
        # Zwei Features für einfache Visualisierung
        X = np.random.rand(n_samples, 2) * 10
        
        # Zielwert basierend auf Schwellenwerten (kann später vom Benutzer angepasst werden)
        y = np.zeros(n_samples)
        y[(X[:, 0] > 5) & (X[:, 1] > 5)] = 1  # Obere rechte Ecke
        y[(X[:, 0] < 3) & (X[:, 1] < 4)] = 1  # Untere linke Ecke
        
        # DataFrame erstellen
        data = pd.DataFrame({
            'Feature_1': X[:, 0],
            'Feature_2': X[:, 1],
            'Target': y
        })
        
        st.subheader("Beispieldaten")
        st.dataframe(data.head())
    
    # Globale Parameter für den Entscheidungsbaum
    st.subheader("Globale Baumparameter")
    max_depth = st.slider("Maximale Tiefe des Baums", 1, 10, 3)
    min_samples_split = st.slider("Mindestanzahl Samples für Split", 2, 20, 2)
    min_samples_leaf = st.slider("Mindestanzahl Samples pro Blatt", 1, 20, 1)
    
    # Test-Train-Split
    test_size = st.slider("Testdaten-Anteil", 0.1, 0.5, 0.3)
    
    # Feature-Auswahl, wenn Daten geladen sind
    if 'data' in locals():
        st.subheader("Features auswählen")
        feature_cols = [col for col in data.columns if col != 'Target']
        selected_features = st.multiselect(
            "Features für den Entscheidungsbaum auswählen",
            feature_cols,
            default=feature_cols[:2]
        )
        
        target_col = st.selectbox(
            "Zielvariable auswählen",
            [col for col in data.columns],
            index=data.columns.get_loc('Target') if 'Target' in data.columns else 0
        )

# Hauptbereich
if 'data' in locals():
    # Tabs für verschiedene Funktionen
    tab1, tab2, tab3 = st.tabs(["Datenvisualisierung", "Baumerstellung", "Vorhersage & Bewertung"])
    
    with tab1:
        st.header("Datenvisualisierung")
        
        if len(selected_features) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Streudiagramm")
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    data[selected_features[0]], 
                    data[selected_features[1]],
                    c=data[target_col],
                    cmap='viridis',
                    alpha=0.6
                )
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                ax.set_title("Datenverteilung")
                legend = ax.legend(*scatter.legend_elements(), title="Klassen")
                st.pyplot(fig)
            
            with col2:
                st.subheader("Korrelationsmatrix")
                selected_data = data[selected_features + [target_col]]
                corr = selected_data.corr()
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
        
        st.subheader("Datenstatistik")
        st.dataframe(data[selected_features + [target_col]].describe())
        
    with tab2:
        st.header("Entscheidungsbaum erstellen")
        
        # Daten vorbereiten
        X = data[selected_features]
        y = data[target_col]
        
        # Train-Test-Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        
        st.write(f"Training mit {len(X_train)} Datenpunkten, Testen mit {len(X_test)} Datenpunkten")
        
        # Interaktive Schwellenwerte für Knoten (simplere Version)
        st.subheader("Manuelle Schwellenwerte einstellen")
        st.info("Hier kannst du manuell Schwellenwerte festlegen, die die ersten Entscheidungen im Baum beeinflussen.")
        
        # Container für manuelle Entscheidungsregeln
        manual_rules = []
        num_rules = st.slider("Anzahl der manuellen Regeln", 0, 5, 1)
        
        for i in range(num_rules):
            st.markdown(f"#### Regel {i+1}")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                rule_feature = st.selectbox(f"Feature für Regel {i+1}", selected_features, key=f"feature_{i}")
            
            with col2:
                rule_operator = st.selectbox(f"Operator für Regel {i+1}", ["<", ">", "<=", ">=", "=="], key=f"operator_{i}")
            
            with col3:
                min_val = float(data[rule_feature].min())
                max_val = float(data[rule_feature].max())
                rule_threshold = st.slider(
                    f"Schwellenwert für Regel {i+1}", 
                    min_val, 
                    max_val, 
                    (min_val + max_val) / 2,
                    key=f"threshold_{i}"
                )
            
            manual_rules.append((rule_feature, rule_operator, rule_threshold))
        
        # Modell trainieren
        model_option = st.radio(
            "Modelltyp",
            ["Automatischer Entscheidungsbaum", "Manueller Entscheidungsbaum mit benutzerdefinierten Regeln"]
        )
        
        if st.button("Entscheidungsbaum trainieren"):
            with st.spinner("Modell wird trainiert..."):
                if model_option == "Automatischer Entscheidungsbaum":
                    # Standardmodell mit scikit-learn
                    model = DecisionTreeClassifier(
                        max_depth=max_depth,
                        min_samples_split=min_samples_split,
                        min_samples_leaf=min_samples_leaf,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    
                    # Visualisierung des automatischen Baums
                    st.subheader("Visualisierung des Entscheidungsbaums")
                    fig, ax = plt.subplots(figsize=(20, 10))
                    plot_tree(
                        model, 
                        feature_names=selected_features,
                        class_names=[str(c) for c in model.classes_],
                        filled=True,
                        rounded=True,
                        ax=ax
                    )
                    st.pyplot(fig)
                    
                    # Feature Importance
                    st.subheader("Feature Importance")
                    feature_importance = pd.DataFrame({
                        'Feature': selected_features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.barplot(x='Importance', y='Feature', data=feature_importance, ax=ax)
                    ax.set_title("Feature Importance")
                    st.pyplot(fig)
                
                else:
                    # Manuelle Regeln anwenden für Visualisierung
                    st.subheader("Manueller Entscheidungsbaum mit benutzerdefinierten Regeln")
                    
                    # Eine einfache Funktion, um Vorhersagen basierend auf manuellen Regeln zu treffen
                    
                    
                    # Textuelle Darstellung der Regeln
                    st.markdown("### Deine Entscheidungsregeln:")
                    for i, (feature, operator, threshold) in enumerate(manual_rules):
                        st.markdown(f"Regel {i+1}: Wenn `{feature} {operator} {threshold:.2f}`, dann Klasse = 1")
                    
                    # Visualisierung des manuellen Entscheidungsbaums
                    if len(selected_features) >= 2 and len(manual_rules) > 0:
                        st.subheader("Visualisierung der manuellen Regeln")
                        
                        # Erstelle ein Gitter für die Darstellung
                        x_min, x_max = data[selected_features[0]].min() - 1, data[selected_features[0]].max() + 1
                        y_min, y_max = data[selected_features[1]].min() - 1, data[selected_features[1]].max() + 1
                        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                                             np.arange(y_min, y_max, 0.1))
                        
                        # Erstelle Dataframe für Vorhersage
                        grid_data = pd.DataFrame({
                            selected_features[0]: xx.ravel(),
                            selected_features[1]: yy.ravel()
                        })
                        
                        # Fülle mit 0 für nicht ausgewählte Features
                        for feature in selected_features:
                            if feature not in grid_data.columns:
                                grid_data[feature] = 0
                        
                        # Vorhersage mit manuellen Regeln
                        Z = predict_with_manual_rules(grid_data[selected_features], manual_rules)
                        Z = Z.reshape(xx.shape)
                        
                        # Visualisierung
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
                        scatter = ax.scatter(
                            data[selected_features[0]],
                            data[selected_features[1]],
                            c=data[target_col],
                            cmap='viridis',
                            edgecolors='black'
                        )
                        ax.set_xlabel(selected_features[0])
                        ax.set_ylabel(selected_features[1])
                        ax.set_title("Entscheidungsgrenzen mit manuellen Regeln")
                        legend = ax.legend(*scatter.legend_elements(), title="Klassen")
                        st.pyplot(fig)
                    
                    # Evaluierung der manuellen Regeln
                    y_pred_manual = predict_with_manual_rules(X_test[selected_features], manual_rules)
                    manual_accuracy = accuracy_score(y_test, y_pred_manual)
                    st.metric("Genauigkeit der manuellen Regeln", f"{manual_accuracy:.2%}")
                
    with tab3:
        st.header("Vorhersage und Bewertung")
        
        if "model" in locals() or len(manual_rules) > 0:
            # Testdaten verwenden
            st.subheader("Modellbewertung auf Testdaten")
            
            if model_option == "Automatischer Entscheidungsbaum" and "model" in locals():
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Genauigkeit", f"{accuracy:.2%}")
                    
                    # Klassifikationsbericht
                    st.subheader("Klassifikationsbericht")
                    report = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                
                with col2:
                    # Konfusionsmatrix
                    st.subheader("Konfusionsmatrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Vorhergesagte Klasse')
                    ax.set_ylabel('Tatsächliche Klasse')
                    st.pyplot(fig)
            
            elif len(manual_rules) > 0:
                y_pred_manual = predict_with_manual_rules(X_test[selected_features], manual_rules)
                manual_accuracy = accuracy_score(y_test, y_pred_manual)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Genauigkeit manueller Regeln", f"{manual_accuracy:.2%}")
                    
                    # Klassifikationsbericht
                    st.subheader("Klassifikationsbericht")
                    report = classification_report(y_test, y_pred_manual, output_dict=True)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df)
                
                with col2:
                    # Konfusionsmatrix
                    st.subheader("Konfusionsmatrix")
                    cm = confusion_matrix(y_test, y_pred_manual)
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Vorhergesagte Klasse')
                    ax.set_ylabel('Tatsächliche Klasse')
                    st.pyplot(fig)
            
            # Live-Vorhersage mit benutzerdefinierten Werten
            st.subheader("Live-Vorhersage")
            st.markdown("Gib Werte für die Features ein, um eine Vorhersage zu erhalten:")
            
            # Eingabefelder für jedes Feature
            live_input = {}
            cols = st.columns(min(3, len(selected_features)))
            
            for i, feature in enumerate(selected_features):
                col_index = i % len(cols)
                with cols[col_index]:
                    min_val = float(data[feature].min())
                    max_val = float(data[feature].max())
                    mean_val = float(data[feature].mean())
                    
                    live_input[feature] = st.slider(
                        f"Wert für {feature}",
                        min_val,
                        max_val,
                        mean_val
                    )
            
            # Vorhersagebutton
            if st.button("Vorhersage treffen"):
                input_df = pd.DataFrame([live_input])
                
                if model_option == "Automatischer Entscheidungsbaum" and "model" in locals():
                    prediction = model.predict(input_df[selected_features])
                    prediction_proba = model.predict_proba(input_df[selected_features])
                    
                    st.success(f"Vorhergesagte Klasse: {prediction[0]}")
                    st.write(f"Wahrscheinlichkeiten: {prediction_proba[0]}")
                    
                    # Visualisierung des Vorhersagepfads
                    st.subheader("Entscheidungspfad")
                    node_indicator = model.decision_path(input_df[selected_features])
                    leaf_id = model.apply(input_df[selected_features])
                    
                    # Textuelle Darstellung des Pfads
                    st.markdown("### Pfad durch den Entscheidungsbaum:")
                    feature = model.tree_.feature
                    threshold = model.tree_.threshold
                    
                    # Extrahiere den Pfad
                    node_index = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]
                    
                    # Zeige den Pfad an
                    for i, node_id in enumerate(node_index):
                        if i == len(node_index) - 1:  # Wenn es ein Blattknoten ist
                            st.markdown(f"**Endknoten {node_id}**: Vorhergesagte Klasse = {prediction[0]}")
                        else:
                            # Anzeigen der Entscheidung an diesem Knoten
                            if feature[node_id] != -2:  # -2 bedeutet, dass es ein Blattknoten ist
                                feature_name = selected_features[feature[node_id]]
                                feature_val = input_df[feature_name].values[0]
                                
                                if feature_val <= threshold[node_id]:
                                    decision = "<="
                                else:
                                    decision = ">"
                                
                                st.markdown(f"**Knoten {node_id}**: {feature_name} = {feature_val:.2f} {decision} {threshold[node_id]:.2f}")
                
                elif len(manual_rules) > 0:
                    # Anwenden der manuellen Regeln
                    prediction = 0  # Standardklasse ist 0
                    rule_triggered = False
                    
                    for i, (feature, operator, threshold) in enumerate(manual_rules):
                        feature_val = input_df[feature].values[0]
                        condition_met = False
                        
                        if operator == "<" and feature_val < threshold:
                            condition_met = True
                        elif operator == ">" and feature_val > threshold:
                            condition_met = True
                        elif operator == "<=" and feature_val <= threshold:
                            condition_met = True
                        elif operator == ">=" and feature_val >= threshold:
                            condition_met = True
                        elif operator == "==" and feature_val == threshold:
                            condition_met = True
                        
                        if condition_met:
                            prediction = 1
                            rule_triggered = True
                            st.success(f"Vorhergesagte Klasse: {prediction} (Regel {i+1} angewendet)")
                            break
                    
                    if not rule_triggered:
                        st.success(f"Vorhergesagte Klasse: {prediction} (keine Regel angewendet)")
else:
    st.warning("Bitte wähle eine Datenquelle aus der Seitenleiste.")

# Zusätzliche Informationen
st.markdown("---")
st.markdown("""
### Tipps zur Verwendung
- **Beispieldaten**: Nutze die vordefinierten Beispieldaten, um die App kennenzulernen.
- **Eigene Daten**: Lade eine CSV-Datei hoch, um mit deinen eigenen Daten zu arbeiten.
- **Manuelle Regeln**: Definiere eigene Schwellenwerte, um zu sehen, wie sie die Klassifizierung beeinflussen.
- **Automatischer Baum**: Lass den Algorithmus optimale Schwellenwerte finden.
- **Live-Vorhersage**: Teste dein Modell mit individuellen Werten in Echtzeit.
""")