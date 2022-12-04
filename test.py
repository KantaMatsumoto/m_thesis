from flask import Flask, render_template
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sleep.db'
db = SQLAlchemy(app)


class sleep(db.Model):
    __tablename__ = 'sleep'
    id = db.Column(db.Integer, primary_key=True)
    sleepHourStart = db.Column(db.String(5), nullable=False)
    sleepMinStart = db.Column(db.String(5), nullable=False)
    sleepHourEnd = db.Column(db.String(5), nullable=False)
    sleepMinEnd = db.Column(db.String(5), nullable=False)
    due = db.Column(db.DateTime, nullable=False)

@app.before_first_request
def init():
    db.create_all()

if __name__ == "__main__":
    app.run(debug=True)




