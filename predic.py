import pandas as pd
from prophet import Prophet

print(f"inputX: {inputX}")
print(f"inputY: {inputY}")
print(f"outputX: {outputX}")

df = pd.DataFrame()
df['ds'] = pd.to_datetime(inputX, unit='ms')
df['y'] = inputY

model = Prophet()
model.fit(df)

future = pd.DataFrame()
future['ds'] = pd.to_datetime(outputX, unit='ms')

forecast = model.predict(future)
outputY = forecast['yhat'].tolist()
print(f"result: {outputY}")
return outputY

import pandas as pd
import numpy as np
from prophet import Prophet

futureRegressors = []
regressorsCount = len(historyRegressors)

for i in range(0, regressorsCount):
	regressorInputX = inputX
	regressorOutputX = outputX
	regressorInputY = historyRegressors[i]
	regressorOutputY = []

	df = pd.DataFrame()
	df['ds'] = pd.to_datetime(regressorInputX, unit='ms')
	df['y'] = regressorInputY

	model = Prophet()
	model.fit(df)

	future = pd.DataFrame()
	future['ds'] = pd.to_datetime(regressorOutputX, unit='ms')

	forecast = model.predict(future)
	regressorOutputY = forecast['yhat'].tolist()
	futureRegressors.append(regressorOutputY)


for i in range(0, regressorsCount):
	print(f"historyRegressors{i} = {historyRegressors[i]}")
for i in range(0, regressorsCount):
	print(f"futureRegressors{i} = {futureRegressors[i]}")

print(f"inputX: {inputX}")
print(f"inputY: {inputY}")
print(f"outputX: {outputX}")

df = pd.DataFrame()
df['ds'] = pd.to_datetime(inputX, unit='ms')
df['y'] = np.array(inputY)
for i in range(0, regressorsCount):
	df['regressor' + str(i)] = np.array(historyRegressors[i]) 

model = Prophet()
for i in range(0, regressorsCount):
	model.add_regressor('regressor' + str(i), standardize=False) 
model.fit(df)

future = pd.DataFrame()
future['ds'] = pd.to_datetime(outputX, unit='ms')
for i in range(0, regressorsCount):
	future['regressor' + str(i)] = np.array(futureRegressors[i])

forecast = model.predict(future)
outputY = forecast['yhat'].tolist()
print(f"result: {outputY}")
return outputY