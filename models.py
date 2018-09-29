from app import db


class User(db.Model):

    __tablename__ = "user"

    username = db.Column(db.String, primary_key=True)
    emoji = db.Column(db.String, nullable=False)

    def __init__(self, username, emoji):
        self.username = username
        self.emoji = emoji

    def __repr__(self):
        return '<title {}>'.format(self.body)
