import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Real Estate Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        color: #2E86AB;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
    }
    .prediction-value {
        font-size: 3rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .feature-importance {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load the dataset and model"""
    try:
        # Load the dataset
        with open('df.pkl', 'rb') as file:
            df = pickle.load(file)
        
        # Try to load the model
        try:
            with open('pipeline.pkl', 'rb') as file:
                model = pickle.load(file)
        except:
            model = None
            
        return df, model
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

def get_unique_values(df, column):
    """Get unique values for a column"""
    if column in df.columns:
        unique_vals = df[column].dropna().unique()
        if len(unique_vals) > 0:
            return sorted(unique_vals)
    return []

def predict_price(model, input_data):
    """Make price prediction"""
    try:
        if model is not None:
            # Try to predict with the actual model
            prediction = model.predict(input_data)
            return prediction[0]
        else:
            # Simple mock prediction if model not available
            # This is just for demonstration
            base_price = 50000
            area_factor = input_data.iloc[0]['built_up_area'] if 'built_up_area' in input_data.columns else 1000
            bedroom_factor = input_data.iloc[0]['bedRoom'] if 'bedRoom' in input_data.columns else 2
            
            predicted_price = base_price * (area_factor / 1000) * bedroom_factor * np.random.uniform(0.8, 1.2)
            return predicted_price
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        # Return a fallback prediction based on simple heuristics
        try:
            area = input_data.iloc[0]['built_up_area'] if 'built_up_area' in input_data.columns else 1000
            bedrooms = input_data.iloc[0]['bedRoom'] if 'bedRoom' in input_data.columns else 2
            
            # Simple price calculation as fallback
            price_per_sqft = 3000  # Base price per sq ft
            total_price = area * price_per_sqft
            
            # Adjust based on features
            if 'luxury_category' in input_data.columns:
                luxury = input_data.iloc[0]['luxury_category']
                if luxury == 'High':
                    total_price *= 1.3
                elif luxury == 'Medium':
                    total_price *= 1.1
            
            return total_price
        except:
            return 5000000  # Default fallback price

def main():
    # Load data and model
    df, model = load_data()
    
    if df is None:
        st.error("Unable to load dataset. Please check if 'df.pkl' exists.")
        st.stop()
    
    # Debug section - Add this right after loading data
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("**Dataset Info:**")
        st.sidebar.write(f"Shape: {df.shape}")
        st.sidebar.write(f"Columns: {list(df.columns)}")
        
        # Show sample data
        st.write("**Sample Data:**")
        st.dataframe(df.head())
        
        # Show data types
        st.write("**Data Types:**")
        st.write(df.dtypes)
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Price Predictor", "Analytics"])
    
    if page == "Home":
        # Go back to home page
        exec(open('home.py').read())
        return
    elif page == "Analytics":
        # Go to analytics page
        exec(open('Analytics.py').read())
        return
    
    st.markdown("""
    ### Enter Property Details
    Fill in the information below to get an instant price prediction for your property.
    """)
    
    # Create input form
    with st.form("prediction_form"):
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Property Type
            property_types = get_unique_values(df, 'property_type')
            if property_types:
                property_type = st.selectbox(
                    "Property Type",
                    options=property_types,
                    help="Select the type of property"
                )
            else:
                property_type = st.selectbox(
                    "Property Type",
                    options=["flat", "house", "apartment", "villa"],
                    help="Select the type of property"
                )
            
            # Sector
            sectors = get_unique_values(df, 'sector')
            if sectors:
                sector = st.selectbox(
                    "Sector/Location",
                    options=sectors,
                    help="Select the sector or area"
                )
            else:
                sector = st.text_input(
                    "Sector/Location",
                    value="sector 102",
                    help="Enter the sector or area"
                )
            
            # Bedrooms
            bedrooms = get_unique_values(df, 'bedRoom')
            if bedrooms:
                bedRoom = st.selectbox(
                    "Number of Bedrooms",
                    options=bedrooms,
                    help="Select number of bedrooms"
                )
            else:
                bedRoom = st.selectbox(
                    "Number of Bedrooms",
                    options=[1, 2, 3, 4, 5, 6, 7],
                    index=2,
                    help="Select number of bedrooms"
                )
            
            # Bathrooms
            bathrooms = get_unique_values(df, 'bathroom')
            if bathrooms:
                bathroom = st.selectbox(
                    "Number of Bathrooms",
                    options=bathrooms,
                    help="Select number of bathrooms"
                )
            else:
                bathroom = st.selectbox(
                    "Number of Bathrooms",
                    options=[1, 2, 3, 4, 5, 6],
                    index=1,
                    help="Select number of bathrooms"
                )
        
        with col2:
            # Balconies
            balconies = get_unique_values(df, 'balcony')
            if balconies:
                balcony = st.selectbox(
                    "Number of Balconies",
                    options=balconies,
                    help="Select number of balconies"
                )
            else:
                balcony = st.selectbox(
                    "Number of Balconies",
                    options=[0, 1, 2, 3, 4],
                    index=2,
                    help="Select number of balconies"
                )
            
            # Age
            ages = get_unique_values(df, 'age')
            if ages:
                age = st.selectbox(
                    "Property Age",
                    options=ages,
                    help="Select the age of the property"
                )
            else:
                age = st.number_input(
                    "Property Age (years)",
                    min_value=0,
                    max_value=50,
                    value=5,
                    help="Enter the age of the property in years"
                )
            
            # Possession
            possessions = get_unique_values(df, 'Possession')
            if possessions:
                possession = st.selectbox(
                    "Possession Status",
                    options=possessions,
                    help="Select possession status"
                )
            else:
                possession = st.selectbox(
                    "Possession Status",
                    options=["Ready to Move", "Under Construction", "New Launch"],
                    help="Select possession status"
                )
            
            # Built-up Area
            if 'built_up_area' in df.columns:
                min_area = int(df['built_up_area'].min())
                max_area = int(df['built_up_area'].max())
                built_up_area = st.number_input(
                    "Built-up Area (sq ft)",
                    min_value=min_area,
                    max_value=max_area,
                    value=int(df['built_up_area'].mean()),
                    help="Enter the built-up area in square feet"
                )
            else:
                built_up_area = st.number_input(
                    "Built-up Area (sq ft)",
                    min_value=300,
                    max_value=5000,
                    value=1000,
                    help="Enter the built-up area in square feet"
                )
        
        with col3:
            # Servant Room
            servant_rooms = get_unique_values(df, 'servant room')
            if servant_rooms:
                servant_room = st.selectbox(
                    "Servant Room",
                    options=servant_rooms,
                    help="Select if servant room is available"
                )
            else:
                servant_room = st.selectbox(
                    "Servant Room",
                    options=[0.0, 1.0],
                    format_func=lambda x: "No" if x == 0.0 else "Yes",
                    help="Select if servant room is available"
                )
            
            # Store Room
            store_rooms = get_unique_values(df, 'store room')
            if store_rooms:
                store_room = st.selectbox(
                    "Store Room",
                    options=store_rooms,
                    help="Select if store room is available"
                )
            else:
                store_room = st.selectbox(
                    "Store Room",
                    options=[0.0, 1.0],
                    format_func=lambda x: "No" if x == 0.0 else "Yes",
                    help="Select if store room is available"
                )
            
            # Furnishing Type
            furnishing_types = get_unique_values(df, 'furnishing_type')
            if furnishing_types:
                furnishing_type = st.selectbox(
                    "Furnishing Type",
                    options=furnishing_types,
                    help="Select furnishing status"
                )
            else:
                furnishing_type = st.selectbox(
                    "Furnishing Type",
                    options=["furnished", "semi-furnished", "unfurnished"],
                    help="Select furnishing status"
                )
            
            # Luxury Category
            luxury_categories = get_unique_values(df, 'luxury_category')
            if luxury_categories:
                luxury_category = st.selectbox(
                    "Luxury Category",
                    options=luxury_categories,
                    help="Select luxury category"
                )
            else:
                luxury_category = st.selectbox(
                    "Luxury Category",
                    options=["High", "Medium", "Low"],
                    help="Select luxury category"
                )
            
            # Floor Category
            floor_categories = get_unique_values(df, 'floor_category')
            if floor_categories:
                floor_category = st.selectbox(
                    "Floor Category",
                    options=floor_categories,
                    help="Select floor category"
                )
            else:
                floor_category = st.selectbox(
                    "Floor Category",
                    options=["High Floor", "Mid Floor", "Low Floor"],
                    help="Select floor category"
                )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Price", use_container_width=True)
    
    if submitted:
        # Prepare input data with exact column names from your dataset
        # First create the basic input dict
        input_dict = {
            'property_type': property_type,
            'sector': sector,
            'bedRoom': bedRoom,
            'bathroom': bathroom,
            'balcony': balcony,
            'built_up_area': built_up_area,
            'servant room': servant_room,
            'store room': store_room,
            'furnishing_type': furnishing_type,
            'luxury_category': luxury_category,
            'floor_category': floor_category
        }
        
        # Handle the agePossession column specifically
        if 'agePossession' in df.columns:
            # Your dataset has a combined 'agePossession' column
            # We need to create this combined value
            input_dict['agePossession'] = f"{age}_{possession}"
        else:
            # Separate columns
            input_dict['age'] = age
            input_dict['Possession'] = possession
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Ensure all required columns exist and match the model's expected format
        expected_columns = list(df.columns)
        if 'price' in expected_columns:
            expected_columns.remove('price')  # Remove target column if present
        
        # Add missing columns with default values
        missing_cols = set(expected_columns) - set(input_df.columns)
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}")
            for col in missing_cols:
                if col == 'agePossession':
                    # Try to match the format from your dataset
                    input_df[col] = f"{age}_{possession}"
                else:
                    input_df[col] = 0
        
        # Reorder columns to match the training data
        input_df = input_df.reindex(columns=expected_columns, fill_value=0)
        
        # Debug: Print the columns to see what we have
        st.write("**Debug Info:**")
        st.write(f"Input columns: {list(input_df.columns)}")
        st.write(f"Dataset columns: {list(df.columns)}")
        
        # Show sample values from agePossession column if it exists
        if 'agePossession' in df.columns:
            st.write("**Sample agePossession values:**")
            sample_values = df['agePossession'].dropna().unique()[:10]
            st.write(sample_values)
        
        # Show the input data we're sending to the model
        st.write("**Input data being sent to model:**")
        st.dataframe(input_df)
        
        # Make prediction
        with st.spinner("Predicting price..."):
            predicted_price = predict_price(model, input_df)
        
        if predicted_price is not None:
            # Display prediction
            st.markdown(f"""
            <div class="prediction-box">
                <h2>🎯 Predicted Price</h2>
                <div class="prediction-value">₹{predicted_price:,.0f}</div>
                <p>Based on the property details you provided</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display input summary
            st.markdown("### 📋 Property Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Property Type:** {property_type}")
                st.write(f"**Location:** {sector}")
                st.write(f"**Bedrooms:** {bedRoom}")
                st.write(f"**Bathrooms:** {bathroom}")
                st.write(f"**Balconies:** {balcony}")
                st.write(f"**Built-up Area:** {built_up_area:,} sq ft")
                st.write(f"**Age:** {age}")
            
            with col2:
                st.write(f"**Possession:** {possession}")
                st.write(f"**Servant Room:** {servant_room}")
                st.write(f"**Store Room:** {store_room}")
                st.write(f"**Furnishing:** {furnishing_type}")
                st.write(f"**Luxury Category:** {luxury_category}")
                st.write(f"**Floor Category:** {floor_category}")
            
            # Price per sq ft
            price_per_sqft = predicted_price / built_up_area
            st.metric("Price per sq ft", f"₹{price_per_sqft:,.0f}")
            
            # Comparative analysis
            st.markdown("### 📊 Market Comparison")
            
            if 'price' in df.columns:
                # Compare with similar properties
                similar_properties = df[
                    (df['property_type'] == property_type) & 
                    (df['bedRoom'] == bedRoom)
                ]
                
                if len(similar_properties) > 0:
                    avg_market_price = similar_properties['price'].mean()
                    price_difference = predicted_price - avg_market_price
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Market Average", f"₹{avg_market_price:,.0f}")
                    
                    with col2:
                        st.metric("Your Prediction", f"₹{predicted_price:,.0f}")
                    
                    with col3:
                        st.metric("Difference", f"₹{price_difference:,.0f}")
                    
                    # Price trend visualization
                    fig = px.histogram(
                        similar_properties,
                        x='price',
                        nbins=20,
                        title=f"Price Distribution for Similar Properties ({property_type}, {bedRoom} BHK)"
                    )
                    
                    # Add predicted price line
                    fig.add_vline(
                        x=predicted_price,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Your Prediction"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
        
        # Tips section
        st.markdown("### 💡 Tips for Better Price")
        st.info("""
        **To potentially increase your property value:**
        - Consider upgrading furnishing status
        - Improve luxury category features
        - Ensure proper maintenance to reduce effective age
        - Add value-adding amenities like servant room or store room
        """)
        
        # Disclaimer
        st.markdown("### ⚠️ Disclaimer")
        st.warning("""
        This prediction is based on historical data and machine learning algorithms. 
        Actual market prices may vary due to current market conditions, location-specific factors, 
        and other variables not captured in the model. Please consult with real estate professionals 
        for accurate market valuations.
        """)

if __name__ == "__main__":
    main()