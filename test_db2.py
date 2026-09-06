import psycopg2

try:
    conn = psycopg2.connect(dbname="model_decay_radar", user="postgres", password="postgres", host="localhost", port=5432)
    print("SUCCESS_DB")
    conn.close()
except Exception as e:
    print(f"FAILED: {e}")

