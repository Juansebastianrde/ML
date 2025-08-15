
import io
import sys
import contextlib
from pathlib import Path
from typing import List

import streamlit as st

# Opcional para el shim de display()
try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

# --- Config de página ---
st.set_page_config(page_title="Runner de tu notebook (con gráficos)", layout="wide")
st.title("Notebook → App (ejecución sin tocar tu código)")
st.caption("Muestra gráficos de Matplotlib/Seaborn y organiza salidas.")

INFO = st.sidebar.empty()
INFO.info("Sube tu CSV; ejecutamos `notebook_code.py` tal cual.")

# Nombre fijo que espera tu código
FIXED_CSV_NAME = "HDHI Admission data.csv"

uploaded = st.file_uploader("Sube tu CSV (lo renombraremos automáticamente)", type=["csv"])
code_path = Path("notebook_code.py")
st.write("Script que se ejecutará:", f"`{code_path}`")

run = st.button("Ejecutar código", use_container_width=True)

stdout_box = st.empty()
stderr_box = st.empty()
plots_box = st.container()
images_box = st.container()

def _patch_matplotlib_for_streamlit():
    """Intercambia plt.show() para que pinte en Streamlit y cierre figuras."""
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except Exception:
        return None

    def _st_show(*args, **kwargs):
        # Captura TODAS las figuras vivas y las pinta
        try:
            managers = list(matplotlib._pylab_helpers.Gcf.get_all_fig_managers())
            if not managers:
                fig = plt.gcf()
                if fig:
                    st.pyplot(fig)
                    try:
                        plt.close(fig)
                    except Exception:
                        pass
                return
            for m in managers:
                fig = m.canvas.figure
                st.pyplot(fig)
                try:
                    plt.close(fig)
                except Exception:
                    pass
        except Exception as _:
            # fallback: mejor intentar pintar la actual
            try:
                st.pyplot(plt.gcf())
            except Exception:
                pass

    # Monkey-patch
    try:
        import matplotlib.pyplot as plt
        plt.show = _st_show  # type: ignore
    except Exception:
        pass
    return True

def _inject_display_stub(ns: dict):
    """Hace que 'display(obj)' funcione fuera de Jupyter."""
    def display(obj):
        try:
            if pd is not None and isinstance(obj, (pd.DataFrame, pd.Series)):
                st.dataframe(obj)
            else:
                st.write(obj)
        except Exception:
            st.write(obj)
    ns['display'] = display

def _list_images(path: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in exts]

if run:
    # Guardar CSV con el nombre esperado
    if uploaded is not None:
        with open(FIXED_CSV_NAME, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"CSV guardado como `{FIXED_CSV_NAME}`.")
    else:
        if not Path(FIXED_CSV_NAME).exists():
            st.error(f"No se encontró `{FIXED_CSV_NAME}`. Sube el CSV o deja ese archivo en la raíz.")
            st.stop()

    if not code_path.exists():
        st.error("No se encontró `notebook_code.py` en el directorio actual.")
        st.stop()

    # Parches para que los gráficos se vean en Streamlit
    _patch_matplotlib_for_streamlit()

    # Preparar namespace con builtins y utilidades
    ns = {"__name__": "__main__", "__builtins__": __builtins__}
    # Exponer streamlit y Path por si tu código los usa
    ns["st"] = st
    ns["Path"] = Path
    # Hacer que display() funcione
    _inject_display_stub(ns)

    # Capturar stdout/stderr
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    # Inventario de imágenes antes de ejecutar (para detectar nuevas)
    before_imgs = set(p.name for p in _list_images(Path(".")))

    # Ejecutar el script
    try:
        user_code = code_path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"No se pudo leer `{code_path}`: {e}")
        st.stop()

    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compile(user_code, str(code_path), "exec"), ns, ns)
    except Exception as e:
        st.error(f"Ocurrió un error durante la ejecución: {e}")

    # Mostrar stdout / stderr dentro de expanders para que no ensucie la pantalla
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

    # Mostrar imágenes nuevas guardadas por el script (e.g., con plt.savefig)
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
        files = sorted([p for p in Path(".").iterdir() if p.is_file()])
        if files:
            for p in files:
                st.write(f"• `{p.name}` — {p.stat().st_size} bytes")
        else:
            st.write("No hay archivos.")
    except Exception:
        pass
