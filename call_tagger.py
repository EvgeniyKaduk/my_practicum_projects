# ✅ class_CallTagger
import pandas as pd
import pymorphy3
import joblib
import re
from functools import lru_cache
from transliterate import translit
from flashtext import KeywordProcessor
from nltk.corpus import stopwords

class CallTagger:
    def __init__(
        self,
        model_main_path,
        model_brand_path,
        model_employee_path,
        mlb_main,
        mlb_brand,
        mlb_employee,
        entity_dict,
        thresholds=None
                ):
        # --- Модели ---
        self.model_main = joblib.load(model_main_path)
        self.model_brand = joblib.load(model_brand_path)
        self.model_employee = joblib.load(model_employee_path)

        # --- MultiLabelBinarizers ---
        self.mlb_main = mlb_main
        self.mlb_brand = mlb_brand
        self.mlb_employee = mlb_employee

        # --- Морфология и кэш ---
        self.morph = pymorphy3.MorphAnalyzer()
        self._morph_cache = {}

        # --- Словари сущностей и KeywordProcessor ---
        self.extractors = self.build_fast_extractor(entity_dict)

        # --- Пороги ---
        self.thresholds = thresholds or {
            "main": 0.3,
            "brand": 0.3,
            "employee": 0.3,
            "material": 0.1,
            "conqueror": 0.1
        }
        self.threshold_main = self.thresholds.get("main", 0.3)
        self.threshold_brand = self.thresholds.get("brand", 0.3)
        self.threshold_employee = self.thresholds.get("employee", 0.3)
        self.threshold_material = self.thresholds.get("material", 0.1)
        self.threshold_conqueror = self.thresholds.get("conqueror", 0.1)

        # --- Стоп-слова ---
        try:
            self.stopwords = set(stopwords.words("russian"))
        except LookupError:
            import nltk
            nltk.download("stopwords")
            self.stopwords = set(stopwords.words("russian"))

    # ==================================================
    # 🔹 ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ==================================================

    @lru_cache(maxsize=100_000)
    def normalize_word(self, word: str) -> str:
        word = word.strip().lower()
        parsed = self.morph.parse(word)
        return parsed[0].normal_form if parsed else word

    @lru_cache(maxsize=50_000)
    def generate_fast_variants(self, word: str) -> list:
        """
        Генерирует множество устойчивых вариантов для ключевого имени.
        """
        if not isinstance(word, str) or not word.strip():
            return []

        w_raw = word.strip()
        w = w_raw.lower()
        variants = set([w])

        # Нормальная форма
        try:
            parsed = self.morph.parse(w)
            if parsed:
                variants.add(parsed[0].normal_form)
        except Exception:
            pass

        # Падежи и формы (lexeme)
        try:
            parsed = self.morph.parse(w)
            if parsed:
                for form in parsed[0].lexeme:
                    variants.add(form.word.lower())
        except Exception:
            pass

        # Транслитерации
        try:
            t1 = translit(w_raw, "ru")
            t2 = translit(w_raw, "ru", reversed=True)
            variants.update([t1.lower(), t2.lower()])
        except Exception:
            pass

        # Очистка и варианты без пробелов/дефисов
        cleaned = set()
        for v in variants:
            if not isinstance(v, str):
                continue
            v_clean = re.sub(r"[^а-яa-zё0-9\s-]", " ", v.lower())
            v_clean = re.sub(r"\s+", " ", v_clean).strip()
            if v_clean:
                cleaned.add(v_clean)
                cleaned.add(v_clean.replace(" ", ""))
                cleaned.add(v_clean.replace("-", " "))
                cleaned.add(v_clean.replace("-", ""))

        return list(cleaned)
    
    # =========================================================
    # 🔹 Построение KeywordProcessor по словарю
    def build_fast_extractor(self, entity_dict: dict):
        """
        Создаёт KeywordProcessor для каждой категории сущностей.
        Payload — оригинальное написание из словаря (как в entity_dict).
        """
        extractors = {}
        for tag, words in entity_dict.items():
            kp = KeywordProcessor(case_sensitive=False)
            for w in words:
                # получаем варианты через метод класса
                variants = self.generate_fast_variants(w)

                # payload — берём оригинальную форму из словаря ровно как есть
                original_form = w.strip()

                # безопасное приведение variants к списку
                if isinstance(variants, (str, bytes)):
                    variants = [variants]
                elif isinstance(variants, (set, tuple)):
                    variants = list(variants)
                elif not isinstance(variants, list):
                    try:
                        variants = list(variants)
                    except Exception:
                        variants = [str(variants)]

                # добавляем в KeywordProcessor: ключ = вариант, значение = оригинальное имя
                for v in variants:
                    if isinstance(v, str) and v.strip():
                        kp.add_keyword(v.strip(), original_form)

            extractors[tag] = kp

        return extractors
    
    # ==================================================
    # 🔹 ПРЕПРОЦЕССИНГ
    # ==================================================

    def preprocess_text(
        self,
        text: str,
        lemmatize: bool = True,
        remove_stopwords: bool = True,
        keep_english: bool = True,
    ) -> str:
        """Очищает и лемматизирует текст звонка"""
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        text = re.sub(r"\[\s*\d+\.\d+s\s*->\s*\d+\.\d+s\s*\]", " ", text)
        text = re.sub(r"\d+\.\d+s\s*[-–>\s]+\s*\d+\.\d+s[:\s]*", " ", text)
        text = re.sub(r"http\S+|www\S+|[\w\.-]+@[\w\.-]+", " ", text)
        text = re.sub(r"\+?\d[\d\-\(\) ]{7,}\d", " ", text)
        text = re.sub(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b", " DATE ", text)
        text = re.sub(r"\b\d+%+\b", " PERCENT ", text)
        text = re.sub(r"\b\d{5,}\b", " ", text)
        text = re.sub(r"[a-z]*\d+[a-z]+", " ", text)
        text = re.sub(r"\b\d+([\.,]\d+)?\b", " NUM ", text)
        text = re.sub(r"[^а-яa-zё\sNUMDATEPERCENT]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        tokens = re.findall(r"[а-яa-zё]+|NUM|DATE|PERCENT", text)

        if lemmatize:
            lemmas = []
            for word in tokens:
                if word in ("NUM", "DATE", "PERCENT"):
                    lemmas.append(word)
                    continue
                if keep_english and re.fullmatch(r"[a-z]+", word):
                    lemmas.append(word)
                    continue
                if remove_stopwords and word in self.stopwords:
                    continue
                if len(word) == 1 and word != "я":
                    continue
                if word in self._morph_cache:
                    lemma = self._morph_cache[word]
                else:
                    lemma = self.morph.parse(word)[0].normal_form
                    self._morph_cache[word] = lemma
                lemmas.append(lemma)
            tokens = lemmas

        return " ".join(tokens)

    def preprocess_series(self, series: pd.Series) -> pd.Series:
        """Применяет препроцессинг к серии"""
        return series.apply(self.preprocess_text)

    # ==================================================
    # 🔹 ОСНОВНОЙ ПРЕДИКТ
    # ==================================================
    
    def merge_predictions_with_probs(
        self,
        texts_raw,
        texts_lemm,
        base_probas,
        tag_names,
        brand_feature="lemmas_text",
        employee_feature="lemmas_text") -> pd.DataFrame:

        """Объединяет предсказания моделей и KeywordProcessor"""
        results = []
        n = len(texts_raw)

        # --- Предсказания моделей брендов и сотрудников ---
        brand_model_probas = None
        employee_model_probas = None
        if self.model_brand is not None:
            Xb = pd.DataFrame({brand_feature: texts_lemm})
            brand_model_probas = self.model_brand.predict_proba(Xb)
        if self.model_employee is not None:
            Xe = pd.DataFrame({employee_feature: texts_lemm})
            employee_model_probas = self.model_employee.predict_proba(Xe)

        # --- Предварительное извлечение сущностей ---
        expected_keys = {"brand", "employee", "material", "conqueror"}
        pre_extracted = {k: [[] for _ in range(n)] for k in expected_keys}

        if self.extractors:
            for tag, ext in self.extractors.items():
                normalized_tag = tag.lower().strip()
                if normalized_tag in expected_keys:
                    pre_extracted[normalized_tag] = [
                        ext.extract_keywords(t) if isinstance(t, str) else [] for t in texts_lemm
                                                 ]

        # --- Построчная обработка ---
        for i in range(n):
            raw_text = texts_raw[i] if not isinstance(texts_raw, pd.DataFrame) else texts_raw.iloc[i, 0]
            lemm_text = texts_lemm[i] if not isinstance(texts_lemm, pd.DataFrame) else texts_lemm.iloc[i, 0]
            row = {}

            probs = base_probas[i]
            main_tags = [(tag_names[j], round(float(probs[j]), 4)) for j in range(len(tag_names))]
            row["main_tags"] = sorted(main_tags, key=lambda x: -x[1])

            brands, employees, materials, conquerors = [], [], [], []
            active_tags = {t for t, p in main_tags if p >= self.threshold_main}

            # --- Бренды ---
            if brand_model_probas is not None:
                model_br = [
                    (self.mlb_brand.classes_[j], round(float(brand_model_probas[i][j]), 4))
                    for j in range(len(self.mlb_brand.classes_))
                    if brand_model_probas[i][j] >= self.threshold_brand
                            ]
                brands.extend(model_br)
                model_brand_names = {n for n, _ in model_br}
            else:
                model_brand_names = set()

            if "brand" in pre_extracted:
                for found in pre_extracted["brand"][i]:
                    if found not in model_brand_names:
                        brands.append((found, 1.0))

            # --- Сотрудники ---
            if employee_model_probas is not None:
                model_em = [
                    (self.mlb_employee.classes_[j], round(float(employee_model_probas[i][j]), 4))
                    for j in range(len(self.mlb_employee.classes_))
                    if employee_model_probas[i][j] >= self.threshold_employee
                            ]
                employees.extend(model_em)
                model_employee_names = {n for n, _ in model_em}
            else:
                model_employee_names = set()

            if "employee" in pre_extracted:
                for found in pre_extracted["employee"][i]:
                    if found not in model_employee_names:
                        employees.append((found, 1.0))

            # --- Материалы ---
            material_prob_main = next((p for t, p in main_tags if t == "Консультация по материалу"), 0.0)
            if "material" in pre_extracted and (
                "Консультация по материалу" in active_tags or material_prob_main >= self.threshold_material
                                               ):
                materials.extend([(m, 1.0) for m in pre_extracted["material"][i]])

            # --- Конкуренты ---
            conqueror_prob_main = next((p for t, p in main_tags if t == "Конкуренты"), 0.0)
            if "conqueror" in pre_extracted and (
                "Конкуренты" in active_tags or conqueror_prob_main >= self.threshold_conqueror
                                                ):
                conquerors.extend([(c, 1.0) for c in pre_extracted["conqueror"][i]])

            # --- Если вероятность 1, но не найден явно ---
            if not conquerors and conqueror_prob_main >= 0.95:
                conquerors.append(("не найден явно", 1.0))

            
            # --- 3️⃣ Объединим и удалим дубли ---
            def merge_list(items):
                """
                Объединяет повторяющиеся элементы, но сохраняет оригинальный регистр (как в словаре).
                """
                seen = {}
                name_map = {}

                for name, prob in items:
                    name_lc = str(name).lower().strip()
                    if name_lc not in seen or prob > seen[name_lc]:
                        seen[name_lc] = prob
                        name_map[name_lc] = name  # запоминаем оригинальную форму

                # возвращаем оригинальные имена (payload), а не строчные
                merged = [(name_map[k], seen[k]) for k in seen]
                return sorted(merged, key=lambda x: -x[1])

            row["brands"] = merge_list(brands)
            row["employees"] = merge_list(employees)
            row["materials"] = merge_list(materials)
            row["conquerors"] = merge_list(conquerors)
            row["call_text"] = raw_text

            results.append(row)

        return pd.DataFrame(results)

    