from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
MASTER_FILE = BASE_DIR / "data" / "processed" / "dataset_maestro_proyectos.xlsx"
DB_FILE = BASE_DIR / "data" / "processed" / "proyectos.db"
DASHBOARD_JSON = BASE_DIR / "data" / "dashboard" / "proyectos_from_db.json"
ML_DATASET = BASE_DIR / "data" / "processed" / "ml_dataset.csv"

KEYWORD_STOPWORDS = {
    "actual",
    "ademas",
    "analisis",
    "aplicacion",
    "aplicada",
    "aplicado",
    "aplicar",
    "base",
    "bogota",
    "caracterizacion",
    "colombia",
    "colombiano",
    "colombiana",
    "como",
    "con",
    "contribuir",
    "debe",
    "del",
    "desarrollo",
    "desarrollar",
    "determinar",
    "diferentes",
    "diseno",
    "disenar",
    "entre",
    "establecer",
    "esta",
    "este",
    "estas",
    "estos",
    "evaluacion",
    "evaluar",
    "forma",
    "generar",
    "ingenieria",
    "investigacion",
    "mediante",
    "modelo",
    "modelos",
    "para",
    "permitan",
    "permitir",
    "por",
    "proceso",
    "procesos",
    "producto",
    "productos",
    "proponer",
    "propuesta",
    "proyecto",
    "realizar",
    "sistema",
    "sistemas",
    "sobre",
    "traves",
    "través",
    "universidad",
    "unal",
    "una",
    "uso",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", " ").replace("\n", " ")
    return re.sub(r"\s+", " ", text).strip()


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    ascii_text = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    ascii_text = re.sub(r"[^a-z0-9]+", "_", ascii_text)
    return ascii_text.strip("_")


def split_multi_value(value: str) -> list[str]:
    parts = re.split(r"\s*;\s*", clean_text(value))
    return sorted({part for part in parts if part})


def split_list_value(value: object) -> list[str]:
    parts = re.split(r"\s*(?:;|,|\|)\s*", clean_text(value))
    return sorted({part for part in parts if part})


def keyword_token(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower())
    ascii_text = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
    return re.sub(r"[^a-z0-9]+", "", ascii_text)


def infer_keywords(*values: object, limit: int = 8) -> list[str]:
    counts: dict[str, int] = {}
    display: dict[str, str] = {}
    text = " ".join(clean_text(value) for value in values if clean_text(value))
    for raw_word in re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ0-9]{4,}", text):
        token = keyword_token(raw_word)
        if len(token) < 4 or token in KEYWORD_STOPWORDS:
            continue
        counts[token] = counts.get(token, 0) + 1
        display.setdefault(token, raw_word.lower())

    ordered = sorted(counts, key=lambda token: (-counts[token], display[token]))
    return [display[token] for token in ordered[:limit]]


def normalize_department(value: str) -> str:
    text = clean_text(value)
    text = re.sub(r"^\d+\s*-\s*", "", text)
    return text.title() if text.isupper() else text


def normalize_estado(value: str) -> str:
    text = clean_text(value)
    lowered = text.lower()
    if "final" in lowered:
        return "Finalizado"
    if "ejec" in lowered or "activo" in lowered:
        return "En ejecución"
    if "suspend" in lowered:
        return "Suspendido"
    return text or "Sin registro"


def first_value(row: Any, *names: str) -> object:
    for name in names:
        value = row.get(name, "")
        if clean_text(value):
            return value
    return ""


def collect_values(row: Any, *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        values.extend(split_multi_value(row.get(name, "")))
    return sorted(set(values))


def collect_list_values(row: Any, *names: str) -> list[str]:
    values: list[str] = []
    for name in names:
        values.extend(split_list_value(row.get(name, "")))
    return sorted(set(values))


def year_from_date(value: object) -> int | None:
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    return int(date.year)


def load_master() -> pd.DataFrame:
    df = pd.read_excel(MASTER_FILE)
    df.columns = [slugify(col) for col in df.columns]
    return df.fillna("")


def build_projects(df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for idx, row in df.iterrows():
        codigo = clean_text(row.get("codigo_hermes"))
        resumen = clean_text(row.get("resumen"))
        objetivo = clean_text(row.get("objetivo_general"))
        productos_propuestos = collect_values(
            row,
            "tipo_de_producto_propuesto",
            "tipo_producto_propuesto",
            "producto_propuesto",
            "productos_propuestos",
        )
        productos_logrados = collect_values(
            row,
            "tipo_de_producto_logrado",
            "tipo_producto_logrado",
            "nombre_de_producto_logrado",
            "nombre_producto_logrado",
        )
        productos = sorted(set(productos_propuestos + productos_logrados))
        proteccion_producto = clean_text(
            first_value(
                row,
                "es_suceptible_de_proteccion_producto",
                "es_suceptible_de_proteccion___producto",
                "es_susceptible_de_proteccion_producto",
                "es_susceptible_de_proteccion___producto",
                "es_succeptible_de_proteccion_producto",
                "es_succeptible_de_proteccion___producto",
                "susceptible_de_proteccion_producto",
                "succeptible_de_proteccion_producto",
                "suceptible_de_proteccion_producto",
                "proteccion_producto",
            )
        )
        palabras_clave = collect_list_values(
            row,
            "palabras_clave",
            "palabra_clave",
            "keywords",
            "key_words",
            "palabras_clave_proyecto",
            "palabras_clave_del_proyecto",
        )
        if not palabras_clave:
            palabras_clave = infer_keywords(
                row.get("nombre_proyecto"),
                objetivo,
                resumen,
                row.get("area_ocde"),
                row.get("tipo_proyecto"),
                limit=3,
            )
        else:
            palabras_clave = palabras_clave[:3]

        records.append(
            {
                "id": idx + 1,
                "codigo_hermes": codigo,
                "nombre": clean_text(row.get("nombre_proyecto")),
                "objetivo": objetivo,
                "resumen": resumen,
                "departamento": normalize_department(
                    first_value(
                        row,
                        "departamento_o_instituto_investigador_principal",
                        "departamento_o_instituto___investigador_principal",
                        "departamento_o_instituto_director_a",
                    )
                ),
                "estado": normalize_estado(row.get("estado_actual")),
                "ods_principal": clean_text(row.get("ods_principal")),
                "area_ocde": clean_text(row.get("area_ocde")),
                "tipo_proyecto": clean_text(row.get("tipo_proyecto")),
                "grupo_de_investigacion": clean_text(
                    first_value(
                        row,
                        "grupo_de_investigacion",
                        "nombre_grupo_de_investigacion",
                        "grupo_investigacion",
                        "nombre_del_grupo_de_investigacion",
                    )
                ),
                "proteccion_producto": proteccion_producto,
                "año_inicio": year_from_date(row.get("fecha_propuesto")),
                "año_fin": None,
                "facultad": "Ingeniería",
                "macrocategoria_id": "M00",
                "macrocategoria": "Sin asignar",
                "subcategoria_id": "M00-S00",
                "subcategoria": "Sin asignar",
                "clasificacion_origen": "sin_asignar",
                "clasificacion_confianza": None,
                "clasificacion_revisada": 0,
                "clasificacion_actualizada_en": None,
                "palabras_clave": palabras_clave,
                "productos_propuestos": productos_propuestos,
                "productos_logrados": productos_logrados,
                "productos_esperados": sorted(set(productos)),
                "texto_ml": " ".join(
                    part
                    for part in [
                        clean_text(row.get("nombre_proyecto")),
                        objetivo,
                        resumen,
                        clean_text(row.get("area_ocde")),
                        clean_text(row.get("tipo_proyecto")),
                    ]
                    if part
                ),
            }
        )
    return pd.DataFrame(records)


def write_database(projects: pd.DataFrame) -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(
            """
            DROP TABLE IF EXISTS project_products;
            DROP TABLE IF EXISTS projects;

            CREATE TABLE projects (
              id INTEGER PRIMARY KEY,
              codigo_hermes TEXT UNIQUE,
              nombre TEXT NOT NULL,
              objetivo TEXT,
              resumen TEXT,
              departamento TEXT,
              facultad TEXT,
              grupo_de_investigacion TEXT,
              estado TEXT,
              ods_principal TEXT,
              area_ocde TEXT,
              tipo_proyecto TEXT,
              proteccion_producto TEXT,
              año_inicio INTEGER,
              año_fin INTEGER,
              macrocategoria_id TEXT,
              macrocategoria TEXT,
              subcategoria_id TEXT,
              subcategoria TEXT,
              clasificacion_origen TEXT DEFAULT 'sin_asignar',
              clasificacion_confianza REAL,
              clasificacion_revisada INTEGER DEFAULT 0,
              clasificacion_actualizada_en TEXT,
              texto_ml TEXT
            );

            CREATE TABLE project_products (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              project_id INTEGER NOT NULL,
              tipo TEXT NOT NULL DEFAULT 'esperado',
              producto TEXT NOT NULL,
              FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE TABLE project_keywords (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              project_id INTEGER NOT NULL,
              palabra TEXT NOT NULL,
              FOREIGN KEY (project_id) REFERENCES projects(id) ON DELETE CASCADE
            );

            CREATE INDEX idx_projects_estado ON projects(estado);
            CREATE INDEX idx_projects_departamento ON projects(departamento);
            CREATE INDEX idx_projects_macro ON projects(macrocategoria_id);
            """
        )

        project_columns = [
            col
            for col in projects.columns
            if col
            not in {
                "palabras_clave",
                "productos_propuestos",
                "productos_logrados",
                "productos_esperados",
            }
        ]
        projects[project_columns].to_sql("projects", conn, if_exists="append", index=False)

        product_rows = []
        keyword_rows = []
        for project in projects.to_dict("records"):
            for product in project["productos_propuestos"]:
                product_rows.append(
                    {"project_id": project["id"], "tipo": "propuesto", "producto": product}
                )
            for product in project["productos_logrados"]:
                product_rows.append(
                    {"project_id": project["id"], "tipo": "logrado", "producto": product}
                )
            for keyword in project["palabras_clave"]:
                keyword_rows.append({"project_id": project["id"], "palabra": keyword})
        if product_rows:
            pd.DataFrame(product_rows).to_sql(
                "project_products", conn, if_exists="append", index=False
            )
        if keyword_rows:
            pd.DataFrame(keyword_rows).to_sql(
                "project_keywords", conn, if_exists="append", index=False
            )


def write_dashboard_json(projects: pd.DataFrame) -> None:
    fields = [
        "id",
        "codigo_hermes",
        "nombre",
        "objetivo",
        "departamento",
        "facultad",
        "año_inicio",
        "año_fin",
        "estado",
        "ods_principal",
        "proteccion_producto",
        "grupo_de_investigacion",
        "macrocategoria_id",
        "macrocategoria",
        "subcategoria_id",
        "subcategoria",
        "palabras_clave",
        "productos_propuestos",
        "productos_logrados",
        "productos_esperados",
    ]
    records = projects[fields].where(pd.notna(projects[fields]), None).to_dict("records")
    DASHBOARD_JSON.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_ml_dataset(projects: pd.DataFrame) -> None:
    columns = [
        "id",
        "codigo_hermes",
        "nombre",
        "objetivo",
        "resumen",
        "area_ocde",
        "tipo_proyecto",
        "proteccion_producto",
        "texto_ml",
        "macrocategoria_id",
        "subcategoria_id",
    ]
    projects[columns].to_csv(ML_DATASET, index=False)


def main() -> None:
    df = load_master()
    projects = build_projects(df)
    write_database(projects)
    write_dashboard_json(projects)
    write_ml_dataset(projects)
    print(f"Database: {DB_FILE}")
    print(f"Dashboard JSON: {DASHBOARD_JSON}")
    print(f"ML dataset: {ML_DATASET}")
    print(f"Projects imported: {len(projects)}")


if __name__ == "__main__":
    main()
