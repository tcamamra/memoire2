from flask import Flask, request, jsonify, render_template
import pandas as pd
import xgboost as xgb

app = Flask(__name__)

# Charger les données clients à partir d'un fichier CSV pour l'analyse et la prédiction
df = pd.read_csv('df_dashboard.csv')

# Charger le modèle XGBoost pré-entraîné à partir du fichier JSON
model = xgb.Booster()
model.load_model('model_streamlit_xgb.json')

@app.route('/')
def home():
    # Page d'accueil de l'application, affichant un formulaire pour entrer l'ID client
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        client_id = int(request.form['client_id'])
        client_data = df[df['id'] == client_id]

        if not client_data.empty:
            # Supprimer la colonne 'id' et garder les caractéristiques nécessaires pour la prédiction
            feature_cols = ['Résultat d exploitation', 'Capitaux propres', 'Rentabilité économique', 
                            'Rentabilité Financière', 'Endettement sur capitaux propres', 'Liquidité relative', 
                            'Liquidité immédiate', 'Dettes fiscales et sociales annee N', 'Endettement sur actifs', 
                            'Variation du chiffre d\'affaire', 'mod_date_annee_cloture_exo', 'Total Actif Circulant', 
                            'Total Immobilisation', 'Actif net d exploitation', 'Ventes annee N', 'Ventes annee N-1', 
                            'Marge commerciale', 'Dotations nettes', 'Marge sur EBE', 'Marge sur REX', 'Marge sur RCAI', 
                            'Marge sur Résultat Net', 'Salaires annee N', 'Masse salariale nette', 'Charges sociales annee N', 
                            'Masse salariale brute', 'Masse salariale nette sur Ch.Ex', 'Masse salariale brute sur Ch.Ex', 
                            'Rotation de l actif', 'Rendement net de l actif', 'Etat de l outil', 'Immobilisation brute', 
                            'Poids du BFR _ CA', 'Rotation des stocks', 'Delai recouvre client', 
                            'Delai règlement fournisseurs', 'Emprunts', 'Disponibilités et ECA', 'Resultat net']
            
            client_features = client_data[feature_cols]

            # Préparer les données pour XGBoost
            dmatrix = xgb.DMatrix(client_features)
            prediction = model.predict(dmatrix)[0]
            result = int(round(prediction))  # Convertir la prédiction en entier (0 ou 1)

            return render_template('result.html', prediction=result)
        else:
            return render_template('result.html', error="Identifiant client non trouvé.")
    except Exception as e:
        return render_template('result.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        client_id = int(request.form['client_id'])
        client_data = df[df['id'] == client_id]

        if not client_data.empty:
            # Supprimer la colonne 'id' et garder les caractéristiques nécessaires pour la prédiction
            feature_cols = ['Résultat d exploitation', 'Capitaux propres', 'Rentabilité économique', 
                            'Rentabilité Financière', 'Endettement sur capitaux propres', 'Liquidité relative', 
                            'Liquidité immédiate', 'Dettes fiscales et sociales annee N', 'Endettement sur actifs', 
                            'Variation du chiffre d\'affaire', 'mod_date_annee_cloture_exo', 'Total Actif Circulant', 
                            'Total Immobilisation', 'Actif net d exploitation', 'Ventes annee N', 'Ventes annee N-1', 
                            'Marge commerciale', 'Dotations nettes', 'Marge sur EBE', 'Marge sur REX', 'Marge sur RCAI', 
                            'Marge sur Résultat Net', 'Salaires annee N', 'Masse salariale nette', 'Charges sociales annee N', 
                            'Masse salariale brute', 'Masse salariale nette sur Ch.Ex', 'Masse salariale brute sur Ch.Ex', 
                            'Rotation de l actif', 'Rendement net de l actif', 'Etat de l outil', 'Immobilisation brute', 
                            'Poids du BFR _ CA', 'Rotation des stocks', 'Delai recouvre client', 
                            'Delai règlement fournisseurs', 'Emprunts', 'Disponibilités et ECA', 'Resultat net']

            client_features = client_data[feature_cols]

            # Préparer les données pour XGBoost
            dmatrix = xgb.DMatrix(client_features)
            prediction = model.predict(dmatrix)[0]
            result = int(round(prediction))  # Convertir la prédiction en entier (0 ou 1)
            return jsonify({'prediction': result})
        else:
            return jsonify({'error': "ID client non trouvé dans nos enregistrements.", 'prediction': None})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(port=8000, debug=True)
