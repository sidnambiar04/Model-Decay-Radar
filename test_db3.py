import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    conn = psycopg2.connect(dbname="postgres", user="postgres", password="postgres", host="localhost", port=5432)
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cursor = conn.cursor()
    cursor.execute("CREATE DATABASE model_decay_radar;")
    cursor.close()
    conn.close()
    print("SUCCESS_CREATED")
except Exception as e:
    print(f"FAILED: {e}")

