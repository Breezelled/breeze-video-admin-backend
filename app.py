from flask import Flask
from flask_cors import *
from controller.budget_revenue_controller import budget_revenue_bp
from controller.type_controller import type_bp
from controller.rating_controller import rating_bp
from controller.other_controller import other_bp
from controller.revenue_prediction_controller import revenue_prediction_bp

app = Flask(__name__)
CORS(app, supports_credentials=True, resources=r"/*")
app.register_blueprint(budget_revenue_bp)
app.register_blueprint(type_bp)
app.register_blueprint(rating_bp)
app.register_blueprint(other_bp)
app.register_blueprint(revenue_prediction_bp)


if __name__ == "__main__":
    app.run()
