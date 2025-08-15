
import io
import sys
import contextlib
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="Ejecutar código del notebook (sin cambios)", layout="wide")
st.title("Runner de tu notebook en Streamlit")
st.caption("Usa **exactamente el mismo código** que nos diste. Solo provee el CSV y ejecuta.")

st.markdown("""
**Cómo funciona**  
1) Sube el **CSV** que espera tu código.  
2) Indica el **nombre de archivo** con el que tu código lo busca (por ejemplo, `HDHI Admission data.csv` o `data.csv`).  
3) (Opcional) Si tu script usa una ruta relativa distinta, cámbiala aquí.  
4) Dale a **Ejecutar** y veremos la salida de `print()` y `stdout` de tu script.  


⚠️ **Importante**: No modificamos tu código. Si tu script usa `matplotlib.pyplot.show()` u otros backends gráficos, es posible que las figuras no aparezcan automáticamente en Streamlit.  
Si quieres ver imágenes, guarda las figuras a archivos (p. ej. `plt.savefig(...)`) y luego podrás mostrarlas desde el panel de 'Archivos generados'.
""")

# --- Parámetros de ejecución ---
default_csv_name = "data.csv"
target_csv_name = "HDHI Admission data.csv"
uploaded = st.file_uploader("Sube tu CSV", type=["csv"])

code_path = Path("notebook_code.py")
st.write("Script que se ejecutará:", f"`{code_path}`")

run = st.button("Ejecutar código")

# Área para logs
log_area = st.empty()

# Guardar CSV y ejecutar
if run:
    if uploaded is None:
        st.error("Primero sube el CSV.")
        st.stop()

    # Guardamos el CSV con el nombre exacto que tu script espera
    with open(target_csv_name, "wb") as f:
        f.write(uploaded.getbuffer())

    if not code_path.exists():
        st.error("No se encontró 'notebook_code.py' en el directorio. Sube/añade tu código al repositorio con ese nombre.")
        st.stop()

    # Ejecutamos el código EXACTO sin modificar, capturando stdout/stderr
    with open(code_path, "r", encoding="utf-8") as f:
        user_code = f.read()

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
            # Ejecución en un nuevo namespace pero compartiendo globals de Streamlit si es necesario
            exec(compile(user_code, str(code_path), "exec"), {} , {})
    except Exception as e:
        # Mostrar cualquier error
        st.error(f"Ocurrió un error durante la ejecución: {e}")
    finally:
        out = stdout_buffer.getvalue()
        err = stderr_buffer.getvalue()

        if out.strip():
            st.subheader("Salida (stdout)")
            st.code(out, language="text")
        else:
            st.info("No hubo salida por `print()` o stdout.")

        if err.strip():
            st.subheader("Errores / advertencias (stderr)")
            st.code(err, language="text")

# Mostrar archivos generados en el directorio de trabajo
st.divider()
st.subheader("Archivos en el directorio actual")
try:
    files = sorted([p for p in Path('.').iterdir() if p.is_file()])
    if files:
        for p in files:
            st.write(f"• `{p.name}`  —  {p.stat().st_size} bytes")
    else:
        st.write("No hay archivos aún.")
except Exception as _:
    pass
