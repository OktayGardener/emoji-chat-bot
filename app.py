# imports
import os

from flask import (Flask, request, render_template, flash, jsonify)

from flask_sqlalchemy import SQLAlchemy
from bot import (Bot, INTENT_JSON)
import json

# get the folder where this file runs
basedir = os.path.abspath(os.path.dirname(__file__))

# configuration
DATABASE = 'users.db'
DEBUG = True
SECRET_KEY = 'precious'
USERNAME = 'admin'
PASSWORD = 'admin'

# define the full path for the database
DATABASE_PATH = os.path.join(basedir, DATABASE)

# database config
SQLALCHEMY_DATABASE_URI = 'sqlite:///' + DATABASE_PATH
SQLALCHEMY_TRACK_MODIFICATIONS = False


def load_intents():
    with open(INTENT_JSON) as json_data:
        return json.load(json_data)


# Create app
app = Flask(__name__)
app.config.from_object(__name__)
db = SQLAlchemy(app)

import models


def handle_save(username, emoji):
    if emoji is None:
        return
    utf8emoji = emoji.encode(encoding='UTF-8')

    if utf8emoji is None:
        return

    new_entry = models.User(username, utf8emoji)
    user_data = db.session.query(models.User).filter_by(username=username).first()
    if not user_data:
        db.session.add(new_entry)
    else:
        user_data.emoji = utf8emoji

    db.session.commit()


def handle_return_saved(username):
    user_data = db.session.query(models.User).filter_by(username=username).first()
    emoji = user_data.emoji
    return emoji


@app.route('/', methods=['GET', 'POST'])
def index():
    intents = load_intents()
    bot = Bot(intents, basedir)
    bot.create_model()
    bot.load_model()

    if request.method == 'POST':
        username = request.form['username']
        query = request.form['query']
        flash(query)

        result = None
        tag, response, emoji = bot.response(query)

        if tag == 'save':
            handle_save(username, emoji)
        elif tag == 'return_saved':
            emoji = handle_return_saved(username)
            result = ('%s %s') % (response, str(emoji, 'utf-8'))

        if not result:
            result = response

        if result:
            flash(result)

    return render_template('index.html')


@app.route('/chat/', methods=['GET'])
def chat():
    intents = load_intents()
    bot = Bot(intents, basedir)
    bot.create_model()
    bot.load_model()

    username = request.args.get("username")
    query = request.args.get("query")

    result = None
    tag, response, emoji = bot.response(query)

    print('tag: %s' % tag)
    print('response: %s' % response)
    print('emoji: %s' % emoji)

    if tag == 'save':
        handle_save(username, emoji)
    elif tag == 'return_saved':
        emoji = handle_return_saved(username)
        res = ('%s %s') % (response, str(emoji, 'utf-8'))
        result = {'response': res}

    if not result:
        result = {'response': response}
    return jsonify(result)


if __name__ == '__main__':
    app.run()
