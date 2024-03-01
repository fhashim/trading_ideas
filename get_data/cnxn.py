from sqlalchemy import create_engine

def server_access():
    server ='localhost'
    username = 'sa'
    password = 'Yasin0334'
    server_cnxn = create_engine('mysql+pymysql://{}:{}@{}/stocks'.format(username, password, server))
    return server_cnxn