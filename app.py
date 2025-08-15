import io
import sys
import shutil
import contextlib
from pathlib import Path
from typing import List

import streamlit as st
import pandas as pd

# ------------------ Config de página ------------------
st.set_page_config(page_title="Notebook → App (sin descargar la base)", layout="wide")
st.title("Ejecutar tu notebook como app (sin que el usuario suba nada)")
st.caption("Usa el CSV incluido en el repo y ejecuta **notebook_code.py** tal cual. Capturamos stdout/stderr y mostramos gráficos.")

# ================== Ajustes ==================
# Nombre EXACTO que tu script usa:
FIXED_CSV_NAME = "HDHI Admission data.csv"

# Si prefieres guardar el CSV dentro de una carpeta, añádela aquí:
LOCAL_CANDIDATES = [
    Path(FIXED_CSV_NAME),               # raíz
    Path("data") / FIXED_CSV_NAME,      # data/HDHI Admission data.csv
    Path("dataset") / FIXED_CSV_NAME,   # dataset/...
    Path("datasets") / FIXED_CSV_NAME,  # datasets/...
]

# Si quieres un fallback desde un raw de GitHub, pon aquí la URL (opcional):
RAW_FALLBACK_URL = ""  # ejemplo: "https://raw.githubusercontent.com/tuusuario/turepo/main/HDHI%20Admission%20data.csv"

# Ruta del script a ejecutar (tu código del notebook sin cambios)
CODE_PATH = Path("notebook_code.py")

# ------------------ Utilidades ------------------
def ensure_csv_available() -> Path:
    """
    Garantiza que exista un archivo con el nombre exacto FIXED_CSV_NAME en la raíz.
    1) Si ya existe en la raíz → ok.
    2) Si existe en data/ u otras rutas → lo copia a la raíz con el nombre exacto.
    3) Si RAW_FALLBACK_URL está definido → lo descarga con pandas y lo guarda.
    4) Si nada de lo anterior, ofrece un uploader opcional.
    """
    root_csv = Path(FIXED_CSV_NAME)
    if root_csv.exists():
        return root_csv

    # Buscar en rutas candidatas y copiar
    for cand in LOCAL_CANDIDATES:
        if cand.exists():
            shutil.copy(cand, root_csv)
            return root_csv

    # Fallback remoto (opcional)
    if RAW_FALLBACK_URL:
        try:
            df_fallback = pd.read_csv(RAW_FALLBACK_URL)
            df_fallback.to_csv(root_csv, index=False)
            return root_csv
        except Exception as e:
            st.warning(f"No se pudo leer desde RAW_FALLBACK_URL: {e}")

    # Como última opción, permitir subir un CSV manualmente
    with st.expander("Opcional: subir otro CSV (si no incluiste el archivo en el repo)"):
        up = st.file_uploader("Sube un CSV", type=["csv"])
        if up is not None:
            root_csv.write_bytes(up.getbuffer())
            return root_csv

    return root_csv  # puede no existir; lo validamos fuera

def _patch_matplotlib_for_streamlit():
    """Redirige plt.show() a Streamlit para mostrar todas las figuras."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception:
        return

    def _st_show(*args, **kwargs):
        try:
            managers = list(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())
            if not managers:
                fig = plt.gcf()
                if fig:
                    st.pyplot(fig)
                    plt.close(fig)
                return
            for m in managers:
                fig = m.canvas.figure
                st.pyplot(fig)
                plt.close(fig)
        except Exception:
            try:
                st.pyplot(plt.gcf())
            except Exception:
                pass

    plt.show = _st_show  # type: ignore

def _inject_display_stub(ns: dict):
    """Hace que display(obj) funcione fuera de Jupyter."""
    def display(obj):
        try:
            if isinstance(obj, (pd.DataFrame, pd.Series)):
                st.dataframe(obj)
            else:
                st.write(obj)
        except Exception:
            st.write(obj)
    ns['display'] = display

def _list_images(path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]

# ------------------ Carga automática del CSV ------------------
csv_path = ensure_csv_available()
if not csv_path.exists():
    st.error(
        f"No se encontró `{FIXED_CSV_NAME}` en el repo ni en {', '.join(str(c) for c in LOCAL_CANDIDATES)}.\n"
        f"📌 Sube ese archivo al repositorio (raíz o data/) o define RAW_FALLBACK_URL."
    )
    st.stop()
else:
    st.success(f"Usando dataset incluido: `{csv_path}`")

# ------------------ Ejecutar el notebook_code.py ------------------
if not CODE_PATH.exists():
    st.error("No se encontró `notebook_code.py` en el directorio actual.")
    st.stop()

# Parche para gráficos
_patch_matplotlib_for_streamlit()

# Namespace global para ejecutar tu script “tal cual”
ns = {"__name__": "__main__", "__builtins__": __builtins__, "st": st, "Path": Path}

# Capturamos stdout/stderr
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()

# Inventario de imágenes antes de ejecutar (para detectar nuevas)
before_imgs = set(p.name for p in _list_images(Path(".")))

# Inyectamos display()
_ inject_display_stub = _inject_display_stub  # evitar sombras de nombre con exec
_inject_display_stub(ns)

# Ejecutamos
try:
    user_code = CODE_PATH.read_text(encoding="utf-8")
except Exception as e:
    st.error(f"No se pudo leer `{CODE_PATH}`: {e}")
    st.stop()

try:
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        exec(compile(user_code, str(CODE_PATH), "exec"), ns, ns)
except Exception as e:
    st.error(f"Ocurrió un error durante la ejecución: {e}")

# ------------------ Salidas organizadas ------------------
with st.expander("Salida (stdout)", expanded=False):
    out = stdout_buffer.getvalue()
    if out.strip():
        st.code(out, language="text")
    else:
        st.info("No hubo salida por `print()` o stdout.")

with st.expander("Errores / advertencias (stderr)", expanded=False):
    err = stderr_buffer.getvalue()
    if err.strip():
        st.code(err, language="text")
    else:
        st.info("Sin salida en stderr.")

st.divider()

# Mostrar imágenes nuevas guardadas por tu script (e.g., plt.savefig)
after_imgs = set(p.name for p in _list_images(Path(".")))
new_imgs = sorted(list(after_imgs - before_imgs))
if new_imgs:
    st.subheader("Imágenes guardadas por tu script")
    cols = st.columns(2)
    for i, name in enumerate(new_imgs):
        with cols[i % 2]:
            st.image(str(Path(name)), caption=name, use_container_width=True)
else:
    st.info("No se detectaron nuevas imágenes guardadas por tu script.")

st.divider()
st.subheader("Archivos en el directorio actual")
try:
    files = sorted([p for p in Path('.').iterdir() if p.is_file()])
    if files:
        for p in files:
            st.write(f"• `{p.name}` — {p.stat().st_size} bytes")
    else:
        st.write("No hay archivos.")
except Exception:
    pass
