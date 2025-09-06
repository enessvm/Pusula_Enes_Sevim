import pandas as pd
import numpy as np
import re
import unicodedata
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


DATA_PATH = "Talent_Academy_Case_DT_2025.xlsx"

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class DataPreprocessor:
    """Handles data preprocessing tasks"""

    def __init__(self, df):
        self.df = df

    def info(self):
        """Displays basic information about the dataset"""
        print("Dataset Information:")
        print(self.df.info())
        print("\nMissing Values:")
        print(self.df.isnull().sum())

    def save(self, path):
        """Saves the processed dataframe to an Excel file"""
        self.df.to_excel(
            path,
            index=False,
        )

    def fill_missing_values(self):
        """
        Fill missing values in the dataset based on specific strategies for each missing column.
        """

        # Fill missing columns with patient-specific values
        columns_to_fill = ["KanGrubu", "Cinsiyet", "KronikHastalik"]
        for column in columns_to_fill:
            self._fill_missing_values_by_patient(column)

        # Normalize text columns
        text_columns = [
            "Cinsiyet",
            "KanGrubu",
            "Uyruk",
            "KronikHastalik",
            "Bolum",
            "Alerji",
            "Tanilar",
            "TedaviAdi",
            "UygulamaYerleri",
        ]
        for column in text_columns:
            self.df[column] = self.df[column].apply(self._text_normalizer)

        # Fill categorical columns with "bilinmiyor"
        self.df["Cinsiyet"].fillna("bilinmiyor", inplace=True)
        self.df["KanGrubu"].fillna("bilinmiyor", inplace=True)
        self.df["Alerji"].fillna("bilinmiyor", inplace=True)
        self.df["KronikHastalik"].fillna("bilinmiyor", inplace=True)

        # Group-based filling
        self._fill_groupby(target_col="Bolum", group_col="TedaviAdi")
        self._fill_groupby(target_col="UygulamaYerleri", group_col="TedaviAdi")
        self._fill_groupby(target_col="Tanilar", group_col="TedaviAdi")

    def fix_typos(self):
        """
        Fix common typos in specific columns.
        """

        self.df["Alerji"] = self.df["Alerji"].str.replace(
            "volteren", "voltaren", case=False, regex=False
        )

        self.df["KronikHastalik"] = self.df["KronikHastalik"].str.replace(
            "hipotirodizm", "hiportiroidizm", case=False, regex=False
        )

    def feature_cleaning(self):
        """
        Perform feature cleaning tasks such as dropping unnecessary columns,
        creating age groups, extracting body parts, and numeric extraction.
        """
        self._create_age_groups()
        self._extract_body_parts()
        self._numeric_extraction()

        # Drop unnecessary columns
        self.df.drop(columns=["HastaNo", "Yas"], inplace=True)

    def encode_categorical(self):
        """
        Encode categorical features using one-hot encoding and multi-label binarization.
        """

        # YaşGrubu encoding
        self._encode_simple_categorical("YaşGrubu")

        # Cinsiyet encoding
        self.df["Erkek"] = np.where(self.df["Cinsiyet"] == "Erkek", 1, 0)
        self.df["Kadin"] = np.where(self.df["Cinsiyet"] == "Kadın", 1, 0)
        self.df.drop(columns=["Cinsiyet"], inplace=True)

        # KanGrubu encoding
        self._encode_simple_categorical("KanGrubu")

        # Uyruk encoding
        self._encode_simple_categorical("Uyruk")

        # KronikHastalik encoding
        self._encode_multivalue_categorical("KronikHastalik")

        # Bolum encoding
        self._encode_simple_categorical("Bolum")

        # Alerji encoding
        self._encode_multivalue_categorical("Alerji")

        # Tanilar encoding
        self._encode_multivalue_categorical("Tanilar", top_n=10)

        # TedaviAdi encoding
        self._encode_multivalue_categorical("TedaviAdi", top_n=10)

        # UygulamaYerleri encoding
        self._encode_multivalue_categorical("UygulamaYerleri")

    def encode_numeric(self):
        """
        Encode numeric features using Min-Max scaling.
        """
        scaler = MinMaxScaler()
        self.df["UygulamaSuresi"] = scaler.fit_transform(self.df[["UygulamaSuresi"]])

    def _encode_simple_categorical(self, column):
        """
        Encode simple categorical columns using one-hot encoding.

        Args:
            column (str): Column name to encode
        """

        if column not in self.df.columns:
            print(f"Warning: Column '{column}' not found in dataframe\n")
            return

        dummies = pd.get_dummies(self.df[column], prefix=column, dtype=int)
        self.df = pd.concat([self.df, dummies], axis=1)
        self.df.drop(columns=[column], inplace=True)

    def _encode_multivalue_categorical(self, column, top_n=None):
        """
        Encode multi-value categorical columns using multi-label binarization.

        Args:
            column (str): Column name to encode
        """

        if column not in self.df.columns:
            print(f"Warning: Column '{column}' not found in dataframe\n")
            return

        # Split multi-value entries into lists
        self.df[column] = self.df[column].apply(
            lambda x: list(set(x.strip().split(","))) if isinstance(x, str) else x
        )

        # Get unique values
        if top_n is not None:
            unique_values = set(
                self.df[column].explode().value_counts().head(top_n).index
            )
        else:
            unique_values = set()
            self.df[column].dropna().apply(lambda x: unique_values.update(x))

        # Create binary columns for each unique value
        for value in unique_values:
            self.df[f"{column}_{value}"] = self.df[column].apply(
                lambda x: 1 if isinstance(x, list) and value in x else 0
            )

        # Create 'diger' column for values not in top_n
        if top_n is not None:
            self.df[f"{column}_diger"] = self.df[column].apply(
                lambda x: (
                    1
                    if any(v not in unique_values for v in x)
                    else 0 if isinstance(x, list) else 0
                )
            )

        # Drop original column
        self.df.drop(columns=[column], inplace=True)

    def _create_age_groups(self):
        """
        Create age groups based on the "Yas" column.

        Age groups:
            - 0-12: Child
            - 13-24: Teen
            - 25-35: Young Adult
            - 36-59: Adult
            - 60+: Senior
        """

        bins = [0, 12, 24, 35, 59, np.inf]
        labels = ["Çocuk", "Genç", "Genç Yetişkin", "Yetişkin", "Yaşlı"]
        self.df["YaşGrubu"] = pd.cut(self.df["Yas"], bins=bins, labels=labels)

    def _numeric_extraction(self):
        """
        Extract numeric values from the "TedaviSuresi" and "UygulamaSuresi" column.
        """
        self.df["TedaviSuresi"] = (
            self.df["TedaviSuresi"].str.extract(r"(\d+)").astype(int)
        )
        self.df["UygulamaSuresi"] = (
            self.df["UygulamaSuresi"].str.extract(r"(\d+)").astype(int)
        )

    def _extract_body_parts(self):
        """
        Extract body parts from the "UygulamaYerleri"
        """

        def clean_body_parts(text):
            if pd.isna(text):
                return text

            # Split by comma, clean each part, then rejoin
            parts = []
            for part in text.split(","):
                # Remove Sağ/Sol/Bölgesi and strip whitespace
                cleaned = re.sub(
                    r"\b(sag|sol|bolgesi)\b", "", part, flags=re.IGNORECASE
                ).strip()
                if cleaned:  # Only add non-empty parts
                    parts.append(cleaned)

            return ",".join(parts) if parts else text

        self.df["UygulamaYerleri"] = self.df["UygulamaYerleri"].apply(clean_body_parts)

    def _fill_groupby(self, target_col, group_col):
        """
        Fill missing values in target_col based on the most frequent value in group_col.

        Args:
            target_col (str): Column to fill missing values
            group_col (str): Column to group by
        """

        if target_col not in self.df.columns or group_col not in self.df.columns:
            print(
                f"Warning: Column '{target_col}' or '{group_col}' not found in dataframe\n"
            )
            return

        # Create mapping of group to most frequent target value
        mode_map = (
            self.df.groupby(group_col)[target_col]
            .apply(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
            .to_dict()
        )

        simple_imputer = SimpleImputer(strategy="most_frequent")
        most_frequent_value = simple_imputer.fit_transform(
            self.df[[target_col]]
        ).ravel()[0]

        # Fill missing values using the mapping if target group exists, otherwise use overall most frequent value
        def fill_value(row):
            if pd.isna(row[target_col]):
                group_mode = mode_map.get(row[group_col], None)
                return group_mode if pd.notna(group_mode) else most_frequent_value
            return row[target_col]

        self.df[target_col] = self.df.apply(fill_value, axis=1)

    def _text_normalizer(self, text):
        """
        Normalize text by removing extra whitespace and converting to lowercase.

        Args:
            text (str): Text to normalize

        Returns:
            str : Comma separated text
        """

        def strip_accents(text):
            text = unicodedata.normalize("NFD", text)
            text = "".join(c for c in text if unicodedata.category(c) != "Mn")
            return text

        if pd.isna(text):
            return np.nan

        items = [
            re.sub(r"\s+", " ", strip_accents(item.strip().lower())).strip()
            for item in text.split(",")
        ]

        items = [i for i in items if i != ""]

        return ",".join(items) if items else None

    def _fill_missing_values_by_patient(self, column):
        """
        Fill missing values for a specific column using patient-based forward fill.

        Args:
            column (str): Column name to process
        """

        if column not in self.df.columns:
            print(f"Warning: Column '{column}' not found in dataframe\n")
            return

        # Create mapping of patient ID to known values
        column_map = (
            self.df.groupby("HastaNo")[column]
            .apply(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None)
            .to_dict()
        )

        # Fill missing values using the mapping
        mask = self.df[column].isna()
        self.df.loc[mask, column] = self.df.loc[mask, "HastaNo"].map(column_map)


def main():
    df = pd.read_excel(DATA_PATH)
    preprocessor = DataPreprocessor(df)
    preprocessor.fill_missing_values()
    preprocessor.fix_typos()
    preprocessor.feature_cleaning()
    preprocessor.encode_categorical()
    preprocessor.encode_numeric()
    preprocessor.info()
    preprocessor.save("cleaned_data.xlsx")


if __name__ == "__main__":
    main()
