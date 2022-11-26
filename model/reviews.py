from app import db


class Reviews(db.Model):
    __tablename__ = 'reviews'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    movie_id = db.Column(db.Integer)
    author = db.Column(db.Text)
    date = db.Column(db.Text)
    title = db.Column(db.Text)
    content = db.Column(db.Text)

    def save(self):
        db.session.add(self)
        db.session.commit()

    def to_json(self):
        item = self.__dict__
        if "_sa_instance_state" in item:
            del item["_sa_instance_state"]
        return item

    def __str__(self):
        return self.id
