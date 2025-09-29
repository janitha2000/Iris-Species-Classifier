import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Iris Species Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .info {
        background-color: #d1ecf1;
        color: #0c5460;
        border: 1px solid #bee5eb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the Iris dataset"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    # Create DataFrame
    df = pd.DataFrame(X, columns=iris.feature_names)
    df['target'] = y
    df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # Feature engineering
    df['sepal_ratio'] = df['sepal length (cm)'] / df['sepal width (cm)']
    df['petal_ratio'] = df['petal length (cm)'] / df['petal width (cm)']
    df['sepal_area'] = df['sepal length (cm)'] * df['sepal width (cm)']
    df['petal_area'] = df['petal length (cm)'] * df['petal width (cm)']
    
    return df, iris

@st.cache_resource
def train_models():
    """Train multiple models and return the best one"""
    df, iris = load_and_prepare_data()
    
    # Prepare features
    feature_cols = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 
                   'petal width (cm)', 'sepal_ratio', 'petal_ratio', 'sepal_area', 'petal_area']
    
    X = df[feature_cols].values
    y = df['target'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'SVM': SVC(random_state=42, probability=True)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        # Use scaled data for LR and SVM
        if name in ['Logistic Regression', 'SVM']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = accuracy
        trained_models[name] = model
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x])
    best_model = trained_models[best_model_name]
    
    return {
        'models': trained_models,
        'results': results,
        'best_model': best_model,
        'best_model_name': best_model_name,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'iris': iris
    }

def create_prediction_visualization(features, prediction, probabilities, iris_data):
    """Create visualization for prediction results"""
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Feature Values', 'Prediction Probabilities', 
                       'Feature Comparison', 'Species Distribution'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "pie"}]]
    )
    
    # Feature values
    feature_names = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width']
    fig.add_trace(
        go.Bar(x=feature_names, y=features[:4], name="Your Input", marker_color="skyblue"),
        row=1, col=1
    )
    
    # Prediction probabilities
    species_names = iris_data.target_names
    fig.add_trace(
        go.Bar(x=species_names, y=probabilities, name="Probability", 
               marker_color=["red" if i == prediction else "lightgray" for i in range(3)]),
        row=1, col=2
    )
    
    # Feature comparison with dataset averages
    df, _ = load_and_prepare_data()
    avg_features = df.groupby('species')[iris_data.feature_names].mean()
    
    for i, species in enumerate(species_names):
        fig.add_trace(
            go.Scatter(x=feature_names, y=avg_features.iloc[i], 
                      mode='lines+markers', name=f'{species} (avg)',
                      line=dict(dash='dash')),
            row=2, col=1
        )
    
    fig.add_trace(
        go.Scatter(x=feature_names, y=features[:4], 
                  mode='lines+markers', name='Your Input',
                  line=dict(width=3, color='red')),
        row=2, col=1
    )
    
    # Species distribution in dataset
    species_counts = df['species'].value_counts()
    fig.add_trace(
        go.Pie(labels=species_counts.index, values=species_counts.values,
               name="Dataset Distribution"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=True, 
                     title_text="Iris Classification Analysis")
    
    return fig

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🌸 Iris Species Classifier 🌸</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.2rem; color: #666;">
        Classify Iris flowers into Setosa, Versicolor, or Virginica using machine learning!
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data and train models
    with st.spinner('Loading data and training models...'):
        model_data = train_models()
        df, iris = load_and_prepare_data()
    
    # Sidebar for navigation
    st.sidebar.title("🎛️ Navigation")
    page = st.sidebar.selectbox("Choose a page:", 
                               ["🔮 Make Prediction", "📊 Model Performance", 
                                "📈 Data Analysis", "🧠 About Models"])
    
    if page == "🔮 Make Prediction":
        st.markdown('<h2 class="sub-header">Make a Prediction</h2>', 
                   unsafe_allow_html=True)
        
        # Create two columns for input
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌿 Sepal Measurements")
            sepal_length = st.slider("Sepal Length (cm)", 
                                   min_value=4.0, max_value=8.0, value=5.1, step=0.1)
            sepal_width = st.slider("Sepal Width (cm)", 
                                  min_value=2.0, max_value=4.5, value=3.5, step=0.1)
        
        with col2:
            st.subheader("🌺 Petal Measurements")
            petal_length = st.slider("Petal Length (cm)", 
                                   min_value=1.0, max_value=7.0, value=1.4, step=0.1)
            petal_width = st.slider("Petal Width (cm)", 
                                  min_value=0.1, max_value=2.5, value=0.2, step=0.1)
        
        # Calculate engineered features
        sepal_ratio = sepal_length / sepal_width
        petal_ratio = petal_length / petal_width
        sepal_area = sepal_length * sepal_width
        petal_area = petal_length * petal_width
        
        # Prepare input for prediction
        input_features = np.array([[sepal_length, sepal_width, petal_length, petal_width,
                                  sepal_ratio, petal_ratio, sepal_area, petal_area]])
        
        # Make prediction button
        if st.button("🎯 Classify Iris Species", type="primary"):
            
            best_model = model_data['best_model']
            best_model_name = model_data['best_model_name']
            scaler = model_data['scaler']
            
            # Scale input if necessary
            if best_model_name in ['Logistic Regression', 'SVM']:
                input_scaled = scaler.transform(input_features)
                prediction = best_model.predict(input_scaled)[0]
                probabilities = best_model.predict_proba(input_scaled)[0]
            else:
                prediction = best_model.predict(input_features)[0]
                probabilities = best_model.predict_proba(input_features)[0]
            
            # Display prediction
            predicted_species = iris.target_names[prediction]
            confidence = probabilities[prediction]
            
            # Color coding for different species
            colors = {'setosa': '#ff9999', 'versicolor': '#66b3ff', 'virginica': '#99ff99'}
            
            st.markdown(f"""
            <div class="prediction-result success">
                🌸 Predicted Species: <strong>{predicted_species.title()}</strong><br>
                🎯 Confidence: <strong>{confidence:.2%}</strong><br>
                🤖 Model Used: <strong>{best_model_name}</strong>
            </div>
            """, unsafe_allow_html=True)
            
            # Show all probabilities
            st.subheader("📊 Prediction Probabilities")
            prob_df = pd.DataFrame({
                'Species': iris.target_names,
                'Probability': probabilities
            }).sort_values('Probability', ascending=False)
            
            st.bar_chart(prob_df.set_index('Species'))
            
            # Create and display visualization
            st.subheader("📈 Prediction Analysis")
            fig = create_prediction_visualization(input_features[0], prediction, 
                                                probabilities, iris)
            st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📊 Model Performance":
        st.markdown('<h2 class="sub-header">Model Performance Comparison</h2>', 
                   unsafe_allow_html=True)
        
        results = model_data['results']
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        
        for i, (model_name, accuracy) in enumerate(results.items()):
            with [col1, col2, col3][i]:
                st.metric(
                    label=f"🤖 {model_name}",
                    value=f"{accuracy:.4f}",
                    delta=f"{(accuracy - min(results.values())):.4f}" if accuracy != min(results.values()) else None
                )
        
        # Best model highlight
        best_model_name = model_data['best_model_name']
        st.success(f"🏆 Best Model: **{best_model_name}** with {results[best_model_name]:.4f} accuracy")
        
        # Performance comparison chart
        st.subheader("📈 Accuracy Comparison")
        results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
        
        fig = px.bar(results_df, x='Model', y='Accuracy', 
                    title="Model Accuracy Comparison",
                    color='Accuracy', color_continuous_scale='viridis')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "📈 Data Analysis":
        st.markdown('<h2 class="sub-header">Iris Dataset Analysis</h2>', 
                   unsafe_allow_html=True)
        
        # Dataset overview
        st.subheader("📋 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(iris.feature_names))
        with col3:
            st.metric("Classes", len(iris.target_names))
        with col4:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        # Species distribution
        st.subheader("🌸 Species Distribution")
        species_counts = df['species'].value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.pie(values=species_counts.values, names=species_counts.index,
                        title="Species Distribution")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.dataframe(species_counts.to_frame().T, use_container_width=True)
        
        # Feature distributions
        st.subheader("📊 Feature Distributions")
        
        feature_to_plot = st.selectbox("Select feature to analyze:", iris.feature_names)
        
        fig = px.histogram(df, x=feature_to_plot, color='species', 
                          title=f"Distribution of {feature_to_plot}",
                          marginal="box")
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("🔥 Feature Correlations")
        corr_matrix = df[iris.feature_names].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title="Feature Correlation Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Pairwise relationships
        st.subheader("🔍 Pairwise Relationships")
        
        fig = px.scatter_matrix(df, dimensions=iris.feature_names, color='species',
                               title="Pairwise Feature Relationships")
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
    
    elif page == "🧠 About Models":
        st.markdown('<h2 class="sub-header">About the Models</h2>', 
                   unsafe_allow_html=True)
        
        st.markdown("""
        ### 🤖 Machine Learning Models Used
        
        This application uses three different machine learning algorithms to classify Iris species:
        """)
        
        # Model descriptions
        models_info = {
            "🌳 Random Forest": {
                "description": "An ensemble method that creates multiple decision trees and combines their predictions.",
                "pros": ["High accuracy", "Handles overfitting well", "Feature importance"],
                "cons": ["Less interpretable", "Can be slow on large datasets"]
            },
            "📈 Logistic Regression": {
                "description": "A linear model that uses logistic function for classification.",
                "pros": ["Fast and efficient", "Highly interpretable", "No tuning required"],
                "cons": ["Assumes linear relationship", "Sensitive to outliers"]
            },
            "🎯 Support Vector Machine": {
                "description": "Finds optimal boundary to separate different classes.",
                "pros": ["Effective in high dimensions", "Memory efficient", "Versatile"],
                "cons": ["Slow on large datasets", "Sensitive to feature scaling"]
            }
        }
        
        for model_name, info in models_info.items():
            with st.expander(model_name):
                st.write(f"**Description:** {info['description']}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Pros:**")
                    for pro in info['pros']:
                        st.write(f"✅ {pro}")
                
                with col2:
                    st.write("**Cons:**")
                    for con in info['cons']:
                        st.write(f"❌ {con}")
        
        # Feature engineering section
        st.markdown("""
        ### ⚙️ Feature Engineering
        
        In addition to the original 4 features, we created 4 new engineered features:
        
        - **Sepal Ratio**: Sepal Length / Sepal Width
        - **Petal Ratio**: Petal Length / Petal Width  
        - **Sepal Area**: Sepal Length × Sepal Width
        - **Petal Area**: Petal Length × Petal Width
        
        These engineered features help capture relationships between measurements that might be important for classification.
        """)
        
        # Model performance
        st.markdown("### 🏆 Current Best Model")
        best_model_name = model_data['best_model_name']
        best_accuracy = model_data['results'][best_model_name]
        
        st.info(f"**{best_model_name}** is currently the best performing model with **{best_accuracy:.4f}** accuracy on the test set.")

if __name__ == "__main__":
    # Run the main app
    main()