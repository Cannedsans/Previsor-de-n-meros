import streamlit as st
from streamlit_drawable_canvas import st_canvas
from modules.model import preverImagem

st.title("Previsor de número")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### Desenhe aqui:")
    canvas = st_canvas(
        background_color="#FFFFFF", 
        stroke_color="#000000",     
        stroke_width=18,            #(essencial para o resize!)
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )

with col2:
    st.markdown("### Resultado:")
    if canvas.image_data is not None and canvas.image_data.any():
        with st.spinner("Prevendo resultado..."):
            num, prob = preverImagem(canvas.image_data)

        st.metric("Número previsto", num)
        st.metric("Certeza", f"{prob}%")