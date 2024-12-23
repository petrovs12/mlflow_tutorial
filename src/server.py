import joblib
from seldon_core.user_model import SeldonComponent

class ModelServer(SeldonComponent):
    def __init__(self):
        self.model = joblib.load('models/sklearn_model.pkl')

    def predict(self, X, features_names=None):
        return self.model.predict(X)

if __name__ == "__main__":
    from seldon_core.seldon_server import create_rest_server
    server = create_rest_server(ModelServer())
    server.run(host='0.0.0.0', port=9000) 