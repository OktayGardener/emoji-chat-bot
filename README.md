# NL Emoji Conversational Bot API 😻🔥

This is a project that tries to play around with the power of a very simple NL engine, defined as a
feedforward (fully connected) neural network in Tensorflow, using softmax activation with a regression in the end. So, we're basically transforming the NL problem into an intent classification problem. Cool, right?

This project has a 'bot' in its engine, and uses Flask for exposing an API through /chat for interacting with the bot. Currently, you're able to tell the bot what mood you're in, what your favourite emoji is (and store it by doing so in a small SQL DB), and ask the bot what your favourite emoji is.

## Requirements
💻 Computer
🐍 Python
🦄 APIs in requirements.txt

This project contains:
## Bot
A conversational 'bot', trained on data defined in intents.json. Able to infer the intent of a sentence/conversation and reply in a manner you define.

Known issues:
Mixing up the save/return_saved domains. Currently, I've only seen correct mapping by saying 'What's my favourite' for retreiving favourite emoji, everything else maps to save 🤷🏻‍♂️

## Flask Server
Flask microservice for creating and exposing API through /chat.

###API
Example call:
```
/chat?username=oktay&query=hi my favourite emoji is 😈
```

## Database
SQLAlchemy DB for ease, storing user favourite emojis.

## Model + Training
There's a trained model in the /bot catalog.
If you want to train again with new data, run training in
```bot.py``` by executing ```python bot.py```

## How to Run
First off, create the database:
```python create_db.py```

Then you're ready to go!
Create the webserver:
```python app.py```

There's a very simple UI that you can find at index to talk to the bot,
```http://127.0.0.1:5000/```
