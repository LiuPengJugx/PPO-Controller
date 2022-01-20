import psycopg2
class Pg:

    def __init__(self):
        self.database="postgres"
        self.user = "postgres"
        self.password = "postgres"
        self.host = "10.77.10.172"
        self.port = "5432"

    def get_conn(self):
        conn = psycopg2.connect(database=self.database, user=self.user, password=self.password, host=self.host, port=self.port)
        return conn