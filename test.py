import pandas as pd

featureTime = ['idDay', 'Going-out','Cooking', 'Eating', 'Bathing',  'Other','Sleeping','pred', 'timeLeft']    
data = [[1,4,1,56,45,45,"aa",'aasss',67]]
idDay=5656
print(data)

df = pd.DataFrame(data,columns = featureTime)
f = open('result/' + str(idDay) + '.csv', 'w')
df.to_csv(f)
f.close()