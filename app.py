from flask import Flask
from flask_cors import *
from flask_sqlalchemy import SQLAlchemy
from common.config import Config


app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = Config.DATABASE_URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

from controller.budget_revenue_controller import budget_revenue_bp
from controller.type_controller import type_bp
from controller.rating_controller import rating_bp
from controller.other_controller import other_bp
from controller.revenue_prediction_controller import revenue_prediction_bp
from controller.display_controller import display_bp
from controller.review_data_controller import review_bp

CORS(app, supports_credentials=True, resources=r"/*")
app.config.from_object(Config())
app.register_blueprint(budget_revenue_bp)
app.register_blueprint(type_bp)
app.register_blueprint(rating_bp)
app.register_blueprint(other_bp)
app.register_blueprint(revenue_prediction_bp)
app.register_blueprint(display_bp)
app.register_blueprint(review_bp)


if __name__ == "__main__":
    app.run()
