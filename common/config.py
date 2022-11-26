class Config(object):
    DEBUG = False
    DB_SERVER = 'localhost:5432'
    ENV = 'development'
    DATABASE_URI = 'postgresql://breeze:225579qq@{}/breeze_video'.format(DB_SERVER)

    # @property
    # def DATABASE_URI(self):
    #     return
