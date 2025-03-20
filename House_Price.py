import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
df=pd.read_csv(r"housing.csv")
mean=df["total_bedrooms"].mean()
df["total_bedrooms"]=df["total_bedrooms"].fillna(mean)
df["House_Price"]=df["median_house_value"]
df=df.drop("median_house_value",axis=1)
#Encoding
ct=ColumnTransformer([("encode",OneHotEncoder(drop="first"),[-2])],remainder="passthrough")
df_new=ct.fit_transform(df)
#Scaling
x=df_new[:,:-1]
y=df_new[:,-1]
sc=StandardScaler()
x=sc.fit_transform(x)
#data spliting
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#Prediction
regressor=RandomForestRegressor(n_estimators=15,random_state=42)
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(r2_score(y_test,y_pred))
