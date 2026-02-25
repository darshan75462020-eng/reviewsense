import sqlite3
import os
from werkzeug.security import generate_password_hash

db_path = "instance/site.db"
username = "admin"

# Ask password securely instead of storing it
new_password = input("Enter new admin password: ")

hashed_pw = generate_password_hash(new_password)

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

cursor.execute(
    "UPDATE user SET password = ? WHERE username = ?",
    (hashed_pw, username)
)

conn.commit()
conn.close()

print(f"Password for '{username}' updated successfully.")