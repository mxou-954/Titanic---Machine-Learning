# 🎯 Titanic - Machine Learning avec PyTorch

Ce projet utilise un modèle de réseau de neurones (PyTorch) pour prédire la survie des passagers du Titanic à partir de données issues de Kaggle.

---

## 📦 Dépendances

Assure-toi d’avoir Python 3.10+ et d’installer les modules suivants :

```bash
pip install pandas torch scikit-learn kagglehub
```
		

## 📁 Import et chargement du dataset
Le dataset est automatiquement téléchargé depuis Kaggle via kagglehub :

```bash
import kagglehub

path = kagglehub.dataset_download("ibrahimelsayed182/titanic-dataset")
```

Ensuite, on charge les données avec pandas :

```bash
import pandas as pd
import os

file_path = os.path.join(path, "titanic.csv")  # adapte le nom si nécessaire

df = pd.read_csv(file_path, usecols=[
    'sex', 'age', 'sibsp', 'parch', 'embarked', 'class', 'who', 'alone'
])
df2 = pd.read_csv(file_path, usecols=['survived'])
```
		

## 🚀 Lancer le script
Depuis le terminal, exécute simplement :

```bash
python TITANIC.py
```

## 🔧 Ce que fait ce script :

- Télécharge les données Titanic depuis Kaggle
- Nettoie et normalise les colonnes utiles
- Entraîne un modèle de réseau de neurones
- Prédit la survie d’un passager
	

## 📈 Objectif
- Comprendre le fonctionnement d’un réseau de neurones
- Travailler avec DataLoader, Dropout, LeakyReLU, Scheduler, etc.
- Améliorer les performances du modèle par expérimentation
		

## 🤖 Auteur
Projet développé par @mxou-954 dans le cadre d’un entraînement à l’intelligence artificielle avec PyTorch.
