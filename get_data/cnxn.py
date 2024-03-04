import json
from sqlalchemy import create_engine

def server_access():
    with open('config/creds.json') as f:
        config = json.load(f)
    server = config["db_server"]
    username = config["db_username"]
    password = config["db_password"]
    server_cnxn = create_engine('mysql+pymysql://{}:{}@{}/stocks'.format(username, password, server))
    return server_cnxn