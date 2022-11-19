from flask import Flask
from flask_cors import *
from controller.budget_revenue_controller import budget_revenue_bp

app = Flask(__name__)
CORS(app, supports_credentials=True, resources=r"/*")
app.register_blueprint(budget_revenue_bp)


if __name__ == "__main__":
    app.run()
