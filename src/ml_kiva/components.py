import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
import shap

def render_exploration(df):
    st.header("📊 Loan Data Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.pie(df, names="status", title="Funding Status"))
    with col2:
        st.bar_chart(df["sector_name"].value_counts().head(10))
    
    st.plotly_chart(px.box(df, x="sector_name", y="loan_amount", color="status"), use_container_width=True)

def render_shap_plot(pipeline, input_df, df_explore, get_explainer_func):
    st.subheader("🔬 SHAP Interpretability")
    
    # 1. Access components
    prep = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["rf"]
    
    # 2. Transform and Force Dense
    # Added .toarray() if it's sparse, or .astype(float) to ensure uniform types
    input_transformed = prep.transform(input_df)
    if hasattr(input_transformed, "toarray"):
        input_transformed = input_transformed.toarray()
        
    # Do the same for the background sample
    background = prep.transform(df_explore.sample(100))
    if hasattr(background, "toarray"):
        background = background.toarray()

    # 3. Explain
    explainer = get_explainer_func(model, background)
    shap_values = explainer(input_transformed)
    
    # 4. Plot
    fig, ax = plt.subplots()
    # Ensure we use class index [1] for 'Funded'
    shap.plots.bar(shap_values[0, :, 1], max_display=10, show=False)
    st.pyplot(plt.gcf())
    plt.clf()