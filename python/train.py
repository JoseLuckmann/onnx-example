# Importar bibliotecas necessárias
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from skl2onnx import __max_supported_opset__, __version__

print("documentation for version:", __version__)
print("Last supported opset:", __max_supported_opset__)

# Carregar dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar um modelo de RandomForest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Converter o modelo para o formato ONNX
initial_type = [('float_input', FloatTensorType([None, X.shape[1]]))]

options = {id(model): {'zipmap': False}}#Precisa disso aqui para funcionar no node
onnx_model = convert_sklearn(model, initial_types=initial_type, options=options)

# Salvar o modelo ONNX em disco
with open("random_forest_iris.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

print("Modelo salvo como random_forest_iris.onnx")
