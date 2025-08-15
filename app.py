import io
import sys
import contextlib
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Runner de tu notebook (CSV + código intacto)", layout="wide")
st.title("Ejecutar tu notebook como app (sin modificar tu código)")
st.caption("Sube el CSV y se ejecuta **notebook_code.py** exactamente como está. Capturamos stdout/stderr.")

st.markdown(
    """
**Cómo usar**
1) Sube tu CSV (lo renombramos automáticamente a `HDHI Admission data.csv` para que tu script lo encuentre).
2) Verifica que exista `notebook_code.py` en la raíz del repo (es tu código tal cual).
3) Pulsa **Ejecutar código**.
"""
)

# Nombre de archivo que tu script espera
FIXED_CSV_NAME = "HDHI Admission data.csv"

# Uploader del CSV (si ya lo subiste antes, puedes omitirlo)
uploaded = st.file_uploader("Sube tu CSV", type=["csv"])

# Ruta del script a ejecutar (tu código del notebook sin cambios)
code_path = Path("notebook_code.py")
st.write("Script que se ejecutará:", f"`{code_path}`")

# Botón de ejecución
run = st.button("Ejecutar código", use_container_width=True)

# Área para logs
stdout_box = st.empty()
stderr_box = st.empty()

if run:
    # 1) Guardar el CSV con el nombre exacto que tu código espera
    if uploaded is not None:
        with open(FIXED_CSV_NAME, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success(f"CSV guardado como `{FIXED_CSV_NAME}`.")
    else:
        # Si no subieron en este intento, seguimos si el archivo ya existe de antes
        if not Path(FIXED_CSV_NAME).exists():
            st.error(
                f"No se encontró `{FIXED_CSV_NAME}`. Sube el CSV o deja un archivo con ese nombre en la raíz."
            )
            st.stop()

    # 2) Verificar script
    if not code_path.exists():
        st.error("No se encontró `notebook_code.py` en el directorio actual.")
        st.stop()

    # 3) Ejecutar el código EXACTO del usuario capturando stdout y stderr
    try:
        user_code = code_path.read_text(encoding="utf-8")
    except Exception as e:
        st.error(f"No se pudo leer `{code_path}`: {e}")
        st.stop()

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()

    try:
        ns = {"__name__": "__main__", "__builtins__": __builtins__}
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            exec(compile(user_code, str(code_path), "exec"), ns, ns)
    except Exception as e:
        # Mostramos el error y luego igualmente volcamos stdout/stderr
        st.error(f"Ocurrió un error durante la ejecución: {e}")

    # 4) Mostrar salidas
    out = stdout_buffer.getvalue()
    err = stderr_buffer.getvalue()

    st.subheader("Salida (stdout)")
    if out.strip():
        stdout_box.code(out, language="text")
    else:
        stdout_box.info("No hubo salida por `print()` o stdout.")

    st.subheader("Errores / advertencias (stderr)")
    if err.strip():
        stderr_box.code(err, language="text")
    else:
        stderr_box.info("Sin salida en stderr.")

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
