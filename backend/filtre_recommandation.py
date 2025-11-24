from dataclasses import dataclass
from typing import List, Optional
import pandas as pd
import numpy as np


# ============================
# 1. Modèle des réponses du questionnaire
# ============================

@dataclass
class QuestionnaireAnswers:
    """
    Représente les réponses envoyées par le front (questionnaire).
    """
    Q1: List[str]                 # Moment/type de plat (choix multiples)
    Q2: str                       # Temps de préparation max (choix unique)
    Q3_Calories: Optional[str]    # "Low" / "Medium" / "High" ou None
    Q3_Protein: Optional[str]     # "Low" / "Medium" / "High" ou None
    Q5: List[str]                 # Régimes de base (Vegetarian, Vegan, No pork, ...)
    Allergie_Nuts: bool
    Allergie_Dairy: bool
    Allergie_Egg: bool
    Allergie_Fish: bool
    Allergie_Soy: bool


# ============================
# 2. Helpers de mapping
# ============================

def map_q1_to_internal_meal_types(q1_answers: List[str]) -> List[str]:
    """
    Q1 options :
      - "Main course"
      - "Side dish / accompaniment"
      - "Snack / treat"
      - "Breakfast / Brunch"
      - "Dessert"

    On les mappe vers des types internes : "main", "dessert", "snack", "breakfast".
    """
    mapping = {
        "Main course": "main",
        "Side dish / accompaniment": "main",
        "Snack / treat": "snack",
        "Breakfast / Brunch": "breakfast",
        "Dessert": "dessert",
    }
    meal_types = set()
    for ans in q1_answers:
        if ans in mapping:
            meal_types.add(mapping[ans])
    return list(meal_types)


def map_q2_to_max_time(q2_answer: str) -> Optional[int]:
    """
    Q2 options :
      - "Less than 15 min"
      - "Less than 30 min"
      - "Less than 45 min"
      - "Up to 1h"
      - "Not important"
    """
    mapping = {
        "Less than 15 min": 15,
        "Less than 30 min": 30,
        "Less than 45 min": 45,
        "Up to 1h": 60,
        "Not important": None
    }
    return mapping.get(q2_answer, None)


def build_meal_type_mask(df: pd.DataFrame, meal_types: List[str]) -> pd.Series:
    """
    On reconstruit le type de plat à partir de Is_Breakfast_Brunch et Is_Dessert.
    - "breakfast" -> Is_Breakfast_Brunch == 1
    - "dessert"   -> Is_Dessert == 1
    - "snack"     -> assimilé à Dessert
    - "main"      -> ni breakfast, ni dessert
    """
    if not meal_types:
        return pd.Series(True, index=df.index)

    mask = pd.Series(False, index=df.index)

    if "breakfast" in meal_types and "Is_Breakfast_Brunch" in df.columns:
        mask |= (df["Is_Breakfast_Brunch"] == 1)

    if "dessert" in meal_types and "Is_Dessert" in df.columns:
        mask |= (df["Is_Dessert"] == 1)

    if "snack" in meal_types and "Is_Dessert" in df.columns:
        mask |= (df["Is_Dessert"] == 1)

    if "main" in meal_types and {"Is_Breakfast_Brunch", "Is_Dessert"}.issubset(df.columns):
        mask |= (df["Is_Breakfast_Brunch"] == 0) & (df["Is_Dessert"] == 0)

    return mask


def safe_norm(s: pd.Series) -> pd.Series:
    s = s.astype(float)
    if s.min() == s.max():
        return pd.Series(0.5, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


# ============================
# 3. Filtrage à partir du questionnaire
# ============================

def filter_recipes_with_questionnaire(df: pd.DataFrame, q: QuestionnaireAnswers) -> pd.DataFrame:
    recipes = df.copy()

    # --- Q1 : type de plat / moment ---
    internal_meal_types = map_q1_to_internal_meal_types(q.Q1)
    meal_mask = build_meal_type_mask(recipes, internal_meal_types)
    recipes = recipes[meal_mask]

    # --- Q2 : temps max ---
    max_time = map_q2_to_max_time(q.Q2)
    if max_time is not None and "TotalTime_min" in recipes.columns:
        recipes = recipes[recipes["TotalTime_min"] <= max_time]

    # --- Q3a : objectif calories (Calorie_Category) ---
    if q.Q3_Calories in ("Low", "Medium", "High") and "Calorie_Category" in recipes.columns:
        recipes = recipes[recipes["Calorie_Category"] == q.Q3_Calories]

    # --- Q3b : objectif protéines (Protein_Category) ---
    if q.Q3_Protein in ("Low", "Medium", "High") and "Protein_Category" in recipes.columns:
        recipes = recipes[recipes["Protein_Category"] == q.Q3_Protein]

    # --- Q5 : régimes de base ---
    if "Vegetarian" in q.Q5 and "Is_Vegetarian" in recipes.columns:
        recipes = recipes[recipes["Is_Vegetarian"] == 1]

    if "Vegan" in q.Q5 and "Is_Vegan" in recipes.columns:
        recipes = recipes[recipes["Is_Vegan"] == 1]

    if "No pork" in q.Q5 and "Contains_Pork" in recipes.columns:
        recipes = recipes[recipes["Contains_Pork"] == 0]

    if "No alcohol" in q.Q5 and "Contains_Alcohol" in recipes.columns:
        recipes = recipes[recipes["Contains_Alcohol"] == 0]

    if "Gluten-free" in q.Q5 and "Contains_Gluten" in recipes.columns:
        recipes = recipes[recipes["Contains_Gluten"] == 0]

    # (optionnel si un jour vous ajoutez ces choix textuels dans Q5)
    if "No nuts" in q.Q5 and "Contains_Nuts" in recipes.columns:
        recipes = recipes[recipes["Contains_Nuts"] == 0]

    if "No dairy" in q.Q5 and "Contains_Dairy" in recipes.columns:
        recipes = recipes[recipes["Contains_Dairy"] == 0]

    if "No egg" in q.Q5 and "Contains_Egg" in recipes.columns:
        recipes = recipes[recipes["Contains_Egg"] == 0]

    if "No fish" in q.Q5 and "Contains_Fish" in recipes.columns:
        recipes = recipes[recipes["Contains_Fish"] == 0]

    if "No soy" in q.Q5 and "Contains_Soy" in recipes.columns:
        recipes = recipes[recipes["Contains_Soy"] == 0]

    # --- Allergies (booléens venant du questionnaire) ---
    if q.Allergie_Nuts and "Contains_Nuts" in recipes.columns:
        recipes = recipes[recipes["Contains_Nuts"] == 0]

    if q.Allergie_Dairy and "Contains_Dairy" in recipes.columns:
        recipes = recipes[recipes["Contains_Dairy"] == 0]

    if q.Allergie_Egg and "Contains_Egg" in recipes.columns:
        recipes = recipes[recipes["Contains_Egg"] == 0]

    if q.Allergie_Fish and "Contains_Fish" in recipes.columns:
        recipes = recipes[recipes["Contains_Fish"] == 0]

    if q.Allergie_Soy and "Contains_Soy" in recipes.columns:
        recipes = recipes[recipes["Contains_Soy"] == 0]

    return recipes


# ============================
# 4. Scoring (qualité + nutrition)
# ============================

def add_scores_with_questionnaire(df: pd.DataFrame, q: QuestionnaireAnswers) -> pd.DataFrame:
    recipes = df.copy()

    # --- Quality score : AggregatedRating + ReviewCount ---
    if "AggregatedRating" in recipes.columns:
        rating = recipes["AggregatedRating"].fillna(recipes["AggregatedRating"].median())
    else:
        rating = pd.Series(0.0, index=recipes.index)

    if "ReviewCount" in recipes.columns:
        reviews = recipes["ReviewCount"].fillna(0)
    else:
        reviews = pd.Series(0.0, index=recipes.index)

    rating_norm = safe_norm(rating)
    reviews_norm = safe_norm(np.log1p(reviews))

    recipes["quality_score"] = 0.7 * rating_norm + 0.3 * reviews_norm

    # --- Nutrition score : basé sur Calories + ProteinContent + objectifs Q3 ---
    calories = recipes["Calories"].astype(float).fillna(recipes["Calories"].median())
    proteins = recipes["ProteinContent"].astype(float).fillna(recipes["ProteinContent"].median())

    # Score calories : par défaut neutre
    cal_score = pd.Series(0.5, index=recipes.index)
    if q.Q3_Calories == "Low":
        cal_score = safe_norm(-calories)    # moins = mieux
    elif q.Q3_Calories == "High":
        cal_score = safe_norm(calories)     # plus = mieux

    # Score protéines
    prot_score = pd.Series(0.5, index=recipes.index)
    if q.Q3_Protein == "High":
        prot_score = safe_norm(proteins)    # plus = mieux
    elif q.Q3_Protein == "Low":
        prot_score = safe_norm(-proteins)   # moins = mieux

    recipes["nutrition_score"] = 0.5 * cal_score + 0.5 * prot_score

    # --- Score total ---
    w_quality = 0.6
    w_nutrition = 0.4
    recipes["score_total"] = (
        w_quality * recipes["quality_score"] +
        w_nutrition * recipes["nutrition_score"]
    )

    return recipes


# ============================
# 5. Algorithme complet : sélectionner 15 recettes
# ============================

def select_15_recipes_from_questionnaire(
    recipes_df: pd.DataFrame,
    q: QuestionnaireAnswers,
    n_display: int = 15,
    pool_size: int = 100,
    random_state: int = 42
):
    """
    1) Filtre les recettes avec les réponses du questionnaire
    2) Calcule un score qualité + nutrition
    3) Trie par score_total
    4) Garde un pool (Top pool_size)
    5) Tire n_display recettes au hasard dans ce pool
    6) Retourne UNIQUEMENT la liste des RecipeId recommandés
    """

    filtered = filter_recipes_with_questionnaire(recipes_df, q)

    # fallback si filtres trop stricts
    if filtered.empty:
        filtered = recipes_df.copy()

    scored = add_scores_with_questionnaire(filtered, q)
    scored = scored.sort_values("score_total", ascending=False)

    pool = scored.head(min(pool_size, len(scored)))
    n_to_sample = min(n_display, len(pool))

    selected = pool.sample(n=n_to_sample, random_state=random_state)

    # 👉 On renvoie uniquement les 15 RecipeId
    recipe_ids = selected["RecipeId"].tolist()

    return recipe_ids
