# Basic configuration file containing paths.


class Config(object):
    DEBUG = False
    DEVELOPMENT = False
    ALLOWED_EXTENSIONS = set(['wav'])


class ProductionConfig(Config):
    pass


class DevelopmentConfig(Config):
    DEBUG = True


class TestingConfig(Config):
    TESTING = True
