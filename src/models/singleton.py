class SingletonModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SingletonModel, cls).__new__(cls)
            # initialisation unique ici
        return cls._instance
