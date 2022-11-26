from app import db


class Info(db.Model):
    __tablename__ = 'info'
    id = db.Column(db.Integer, autoincrement=True, primary_key=True)
    name = db.Column(db.Text)
    runtime = db.Column(db.Text)
    type = db.Column(db.Text)
    release_date = db.Column(db.Text)
    intro = db.Column(db.Text)
    director = db.Column(db.Text)
    writer = db.Column(db.Text)
    star = db.Column(db.Text)
    budget = db.Column(db.Text)
    revenue = db.Column(db.Text)
    language = db.Column(db.Text)
    company = db.Column(db.Text)
    country = db.Column(db.Text)
    rating = db.Column(db.Text)
    rating_num = db.Column(db.Text)
    tag = db.Column(db.Text)

    def save(self):
        db.session.add(self)
        db.session.commit()

    def to_json(self):
        item = self.__dict__
        if "_sa_instance_state" in item:
            del item["_sa_instance_state"]
        return item

    def __str__(self):
        return self.name
