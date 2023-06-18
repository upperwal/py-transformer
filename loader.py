


class Loader():
    def __init__(self, link):
        with open(link) as f:
            self.data = f.read()

    def get_data(self):
        return self.data