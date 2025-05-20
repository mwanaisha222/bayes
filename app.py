from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model
model = joblib.load("model/model.joblib")

# Define mappings for categorical features
# These mappings should be based on your training data encoding
# The values below are examples - you should replace them with actual mappings from your dataset

# Species mapping
species_mapping = {
    'Pseudomonas aeruginosa': 0,
    'Acinetobacter pitii': 1,
    'Acinetobacter baumannii': 2,
    'Staphylococcus aureus': 3,
    'Enterococcus faecium': 4,
    'Enterococcus faecalis': 5,
    'Streptococcus agalactiae': 6
    # Add all other species from your dataset
}

# Family mapping
family_mapping = {
    'Non-Enterobacteriaceae': 0,
    'Staphylococcus spp': 1,
    'Enterococcus spp': 2,
    'Streptococcus spp (no S. pneumo)': 3
    # Add all other families from your dataset
}

# Country mapping
country_mapping = {
    'France': 0
    # Add all other countries from your dataset
}

# State mapping (if relevant)
state_mapping = {
    '': 0  # Empty state seems to be common in your dataset
    # Add any actual states if present in your dataset
}

# Gender mapping
gender_mapping = {
    'Male': 0,
    'Female': 1,
    'Unknown': 2
    # Add any other gender categories if present
}

# Age Group mapping
age_group_mapping = {
    '85 and Over': 0,
    '13 to 18 Years': 1,
    '65 to 84 Years': 2,
    '19 to 64 Years': 3,
    'Unknown': 4
    # Add any other age groups from your dataset
}

# Speciality mapping
speciality_mapping = {
    'Emergency Room': 0,
    'Nursing Home / Rehab': 1,
    'Medicine General': 2,
    'Medicine ICU': 3,
    'Surgery General': 4,
    'Pediatric General': 5
    # Add all other specialities from your dataset
}

# Source mapping
source_mapping = {
    'Urine': 0,
    'Ear': 1,
    'Skin': 2,
    'Bronchus': 3,
    'Sputum': 4,
    'Peritoneal Fluid': 5,
    'Bone': 6,
    'Wound': 7,
    'Blood': 8,
    'Gastric Abscess': 9,
    'Stomach': 10,
    'Vagina': 11,
    'Lungs': 12,
    'Nose': 13
    # Add all other sources from your dataset
}

# Phenotype mapping (if relevant)
phenotype_mapping = {
    '': 0,  # Empty phenotype
    'MSSA': 1,
    'MRSA': 2
    # Add all other phenotypes from your dataset
}

# Define the home route to render the form
@app.route('/')
def home():
    # You could pass these mappings to the template if you want to show dropdowns
    return render_template("form.html", 
                          species_options=list(species_mapping.keys()),
                          family_options=list(family_mapping.keys()),
                          country_options=list(country_mapping.keys()),
                          gender_options=list(gender_mapping.keys()),
                          age_group_options=list(age_group_mapping.keys()),
                          speciality_options=list(speciality_mapping.keys()),
                          source_options=list(source_mapping.keys()),
                          phenotype_options=list(phenotype_mapping.keys()))

# Handle form submission
@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.form:  # Form submission
            # Get raw input data from form
            raw_input_data = {
                'Species': request.form['Species'],
                'Family': request.form['Family'],
                'Country': request.form['Country'],
                'State': request.form['State'],
                'Gender': request.form['Gender'],
                'Age.Group': request.form['Age.Group'],
                'Speciality': request.form['Speciality'],
                'Source': request.form['Source'],
                'Year': int(request.form['Year']),
                'Phenotype': request.form['Phenotype']
            }
            
            # Convert raw input data to encoded values using mappings
            encoded_input_data = {
                'Species': species_mapping.get(raw_input_data['Species'], -1),  # -1 as default for unknown values
                'Family': family_mapping.get(raw_input_data['Family'], -1),
                'Country': country_mapping.get(raw_input_data['Country'], -1),
                'State': state_mapping.get(raw_input_data['State'], -1),
                'Gender': gender_mapping.get(raw_input_data['Gender'], -1),
                'Age.Group': age_group_mapping.get(raw_input_data['Age.Group'], -1),
                'Speciality': speciality_mapping.get(raw_input_data['Speciality'], -1),
                'Source': source_mapping.get(raw_input_data['Source'], -1),
                'Year': raw_input_data['Year'],  # Numeric value, keep as is
                'Phenotype': phenotype_mapping.get(raw_input_data['Phenotype'], -1)
            }
            
            # Check for unknown values (mapped to -1)
            unknown_features = [key for key, value in encoded_input_data.items() if value == -1]
            if unknown_features:
                error_message = f"The following features contain unknown values: {', '.join(unknown_features)}"
                return render_template("form.html", error=error_message)
            
            # Create DataFrame for prediction
            input_df = pd.DataFrame([encoded_input_data])
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Return result
            return render_template("form.html", 
                                  prediction=prediction, 
                                  input_data=raw_input_data,
                                  species_options=list(species_mapping.keys()),
                                  family_options=list(family_mapping.keys()),
                                  country_options=list(country_mapping.keys()),
                                  gender_options=list(gender_mapping.keys()),
                                  age_group_options=list(age_group_mapping.keys()),
                                  speciality_options=list(speciality_mapping.keys()),
                                  source_options=list(source_mapping.keys()),
                                  phenotype_options=list(phenotype_mapping.keys()))

        elif request.is_json:  # API request
            data = request.get_json()
            
            # If JSON request already contains encoded values, use directly
            if all(isinstance(value, (int, float)) for key, value in data.items() if key != 'Year'):
                input_df = pd.DataFrame([data])
            else:
                # Otherwise, encode the JSON data
                encoded_data = {
                    'Species': species_mapping.get(data.get('Species', ''), -1),
                    'Family': family_mapping.get(data.get('Family', ''), -1),
                    'Country': country_mapping.get(data.get('Country', ''), -1),
                    'State': state_mapping.get(data.get('State', ''), -1),
                    'Gender': gender_mapping.get(data.get('Gender', ''), -1),
                    'Age.Group': age_group_mapping.get(data.get('Age.Group', ''), -1),
                    'Speciality': speciality_mapping.get(data.get('Speciality', ''), -1),
                    'Source': source_mapping.get(data.get('Source', ''), -1),
                    'Year': int(data.get('Year', 0)),
                    'Phenotype': phenotype_mapping.get(data.get('Phenotype', ''), -1)
                }
                
                # Check for unknown values
                unknown_features = [key for key, value in encoded_data.items() if value == -1]
                if unknown_features:
                    return jsonify({'error': f"Unknown values for features: {', '.join(unknown_features)}"})
                
                input_df = pd.DataFrame([encoded_data])
            
            prediction = model.predict(input_df)
            return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == "__main__":
    app.run(debug=True)