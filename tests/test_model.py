from ml_model.train import train_model

def test_train_model():
	model = train_model()
	assert model is not None  # Ensure the model is trained
# 1
'''def test_prediction():

	# Test if prediction works with the model
	from ml_model.predict import predict
	from sklearn.model_selection import train_test_split
	import pandas as pd

	df = pd.read_csv("C:\\Users\\shashank\\Desktop\\model deployment\\Mall_Customers.csv")
	data = df[['Annual Income (k$)', 'Spending Score (1-100)']]
	train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
	prediction = predict(test_data)
	assert len(prediction) == 1 '''

# 2
'''def test_prediction():
    from ml_model.predict import predict
    from sklearn.model_selection import train_test_split
    import pandas as pd

    # Load data and select relevant features
    df = pd.read_csv("C:\\Users\\shashank\\Desktop\\model deployment\\Mall_Customers.csv")
    data = df[['Annual Income (k$)', 'Spending Score (1-100)']]
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    
    # Convert test_data to the correct format for prediction
    test_data_array = test_data #.to_numpy()
    
    # Predict using the model, checking each sample separately
    for sample in test_data_array:
        prediction = predict(sample)
        # Add assertions if needed to validate the prediction
        assert len(prediction) == 1 
'''
# 3
def test_prediction():
    from ml_model.predict import predict
    from sklearn.model_selection import train_test_split
    import pandas as pd
    test_data = [77,60]
    prediction = predict(test_data)
    assert len(prediction) == 1 
