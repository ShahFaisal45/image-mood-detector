import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(
    page_title="Image Mood Detector",
    page_icon="😊",
    layout="centered"
)

# ---------- CSS ----------
st.markdown("""
<style>

.block-container{
    padding-top:2rem;
}

.title{
    text-align:center;
    font-size:55px;
    font-weight:800;
    margin-bottom:10px;
}

.subtitle{
    text-align:center;
    color:gray;
    margin-bottom:30px;
}

.upload-box{
    padding:25px;
    border-radius:20px;
    background:#f7f7f7;
}

.stButton>button{
    width:100%;
    border-radius:15px;
    height:55px;
    font-size:20px;
    font-weight:bold;
    background-color:#1f77ff;
    color:white;
    border:none;
}

.stButton>button:hover{
    background-color:#005ce6;
    color:white;
}

.result-card{
    padding:20px;
    border-radius:15px;
    background:#eef7ee;
    text-align:center;
    font-size:25px;
    font-weight:bold;
    margin-top:20px;
}

</style>
""", unsafe_allow_html=True)

# ---------- Title ----------
st.markdown(
    "<div class='title'>😊 Image Mood Detector</div>",
    unsafe_allow_html=True
)

st.markdown(
    "<div class='subtitle'>Upload a face image and let AI detect emotions</div>",
    unsafe_allow_html=True
)


@st.cache_resource
def load_model():
    return pipeline(
        "image-classification",
        model="trpakov/vit-face-expression"
    )

classifier = load_model()

# Upload
uploaded_file = st.file_uploader(
    "",
    type=["jpg","jpeg","png"]
)

if uploaded_file:

    image = Image.open(uploaded_file)

    col1,col2,col3=st.columns([1,2,1])

    with col2:
        st.image(image,width=350)

    col1,col2,col3=st.columns([1,2,1])

    with col2:

        if st.button("🔍 Detect Emotion"):

            with st.spinner("Analyzing image..."):

                results=classifier(image)

                best=max(results,key=lambda x:x["score"])

            st.markdown(
            f"""
            <div class='result-card'>
            😊 Emotion: {best['label']} <br><br>
            📊 Confidence: {best['score']:.2%}
            </div>
            """,
            unsafe_allow_html=True
            )