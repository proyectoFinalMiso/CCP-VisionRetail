from dotenv import load_dotenv
from flask import Flask
from sys import argv
from waitress import serve

from flask_config.blueprints import blueprint

load_dotenv('.env')

app = Flask(__name__)
app.register_blueprint(blueprint)

if __name__ == "__main__":
    if argv[1] == "dev":
        app.run(host="0.0.0.0", port=3000, debug=True)

    else:
        serve(app, host="0.0.0.0", port=3000, threads=4)