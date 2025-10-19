import streamlit as st
import pandas as pd
import joblib
import json
from pathlib import Path
from call_tagger import CallTagger

# --- Настройки страницы ---
st.set_page_config(page_title="Call Tagger", layout="wide")

st.title("Call Tagger — классификация звонков и извлечение сущностей")

# --- Боковая панель ---
st.sidebar.header("Настройки")

mode = st.sidebar.radio(
    "Выберите режим работы:",
    ["Анализ одного звонка", "Пакетная обработка CSV"]
)

model_dir = Path("models")
data_dir = Path("data")

# --- Загрузка моделей и словарей ---
@st.cache_resource
def load_tagger():
    model_main = model_dir / "model_main.pkl"
    model_brand = model_dir / "model_brand.pkl"
    model_employee = model_dir / "model_employee.pkl"

    mlb_main = joblib.load(model_dir / "mlb_main.pkl")
    mlb_brand = joblib.load(model_dir / "mlb_brand.pkl")
    mlb_employee = joblib.load(model_dir / "mlb_employee.pkl")
    with open(data_dir / "entity_dict.json", "r", encoding="utf-8") as f:
        entity_dict = json.load(f)

    tagger = CallTagger(
        model_main_path=model_main,
        model_brand_path=model_brand,
        model_employee_path=model_employee,
        mlb_main=mlb_main,
        mlb_brand=mlb_brand,
        mlb_employee=mlb_employee,
        entity_dict=entity_dict
    )
    return tagger

tagger = load_tagger()

# --- Режим 1: Анализ одного звонка ---
if mode == "Анализ одного звонка":
    st.subheader("Введите текст звонка")

    example_text = """Марина, здравствуйте! Хотел уточнить по материалу плёнка..."""
    user_text = st.text_area("Текст звонка:", example_text, height=200)

    if st.button("🔍 Анализировать звонок"):
        if not user_text.strip():
            st.warning("Введите текст для анализа.")
        else:
            with st.spinner("Анализируем..."):
                lemm_text = tagger.preprocess_text(user_text)
                X_main = pd.DataFrame({"lemmas_text": [lemm_text]})
                base_proba = tagger.model_main.predict_proba(X_main)

                df_result = tagger.merge_predictions_with_probs(
                    texts_raw=pd.Series([user_text]),
                    texts_lemm=pd.Series([lemm_text]),
                    base_probas=base_proba,
                    tag_names=tagger.mlb_main.classes_
                )

            st.success("✅ Готово!")

            res = df_result.iloc[0]
            st.markdown("### 📊 Основные теги")
            st.dataframe(pd.DataFrame(res["main_tags"], columns=["Тег", "Вероятность"]))

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 🏷 Бренды")
                st.table(pd.DataFrame(res["brands"], columns=["Бренд", "Score"]) if res["brands"] else pd.DataFrame())

                st.markdown("### 👥 Сотрудники")
                st.table(pd.DataFrame(res["employees"], columns=["Сотрудник", "Score"]) if res["employees"] else pd.DataFrame())

            with col2:
                st.markdown("### 🧱 Материалы")
                st.table(pd.DataFrame(res["materials"], columns=["Материал", "Score"]) if res["materials"] else pd.DataFrame())

                st.markdown("### ⚔️ Конкуренты")
                st.table(pd.DataFrame(res["conquerors"], columns=["Конкурент", "Score"]) if res["conquerors"] else pd.DataFrame())

# --- Режим 2: Пакетная обработка CSV ---
else:
    st.subheader("Загрузите CSV-файл со звонками")

    uploaded = st.file_uploader("Выберите CSV-файл", type=["csv"])

    if uploaded:
        df = pd.read_csv(uploaded)
        text_column = st.selectbox("Выберите колонку с текстом звонка", df.columns)

        if st.button(" Обработать все звонки"):
            with st.spinner("Анализируем звонки..."):
                df["lemmas_text"] = df[text_column].apply(tagger.preprocess_text)
                X_main = pd.DataFrame({"lemmas_text": df["lemmas_text"]})
                base_proba = tagger.model_main.predict_proba(X_main)

                df_res = tagger.merge_predictions_with_probs(
                    texts_raw=df[text_column],
                    texts_lemm=df["lemmas_text"],
                    base_probas=base_proba,
                    tag_names=tagger.mlb_main.classes_
                )

                st.success("✅ Обработка завершена!")

                # Выведем часть результата
                st.dataframe(df_res.head())

                # Кнопка для скачивания
                csv_out = df_res.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    "Скачать результаты CSV",
                    csv_out,
                    "call_tagger_results.csv",
                    "text/csv"
                )
