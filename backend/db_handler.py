import sqlite3
import os


DB_PATH = "database/internify.db"  # SQLite database file path


def get_connection():
    """
    Ensures the database folder exists and returns a SQLite connection.
    """
    os.makedirs("database", exist_ok=True)
    return sqlite3.connect(DB_PATH)


def create_tables():
    """
    Creates application tables if they do not exist:
    - users: basic auth (name/email/password)
    - resumes: raw parsed resume storage
    - internships: catalog of internships (optional for persistence)
    - matches: user-to-internship match scores and cluster ids
    """
    conn = get_connection()
    cur = conn.cursor()

    # Users table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        email TEXT UNIQUE,
        password TEXT
    )
    """)

    # Resumes table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS resumes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        parsed_text TEXT,
        skills TEXT,
        FOREIGN KEY(user_id) REFERENCES users(id)
    )
    """)

    # Internships table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS internships (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        description TEXT,
        skills TEXT
    )
    """)

    # Matches table
    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        internship_id INTEGER,
        score REAL,
        cluster INTEGER,
        FOREIGN KEY(user_id) REFERENCES users(id),
        FOREIGN KEY(internship_id) REFERENCES internships(id)
    )
    """)

    conn.commit()
    conn.close()


def add_user(name, email, password):
    """
    Inserts a new user. Returns True on success, False if email exists.
    """
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(email, password):
    """
    Fetches a user row (tuple) by email/password or None if not found.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    user = cur.fetchone()
    conn.close()
    return user


def save_resume(user_id, parsed_text, skills):
    """
    Persists a parsed resume for a user.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO resumes (user_id, parsed_text, skills) VALUES (?, ?, ?)", (user_id, parsed_text, skills))
    conn.commit()
    conn.close()


def save_match(user_id, internship_id, score, cluster):
    """
    Persists a single match result (score + cluster) for a user and internship.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("INSERT INTO matches (user_id, internship_id, score, cluster) VALUES (?, ?, ?, ?)",
                (user_id, internship_id, score, cluster))
    conn.commit()
    conn.close()


def get_matches_for_user(user_id):
    """
    Returns a list of (company, title, score, cluster) tuples for a given user_id,
    ordered by score descending.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Note: 'internships' table schema does not include a 'company' column.
    # To avoid errors, return an empty company string and the stored internship title if present.
    cur.execute(
        """
        SELECT '' as company, internships.title, matches.score, matches.cluster
        FROM matches
        JOIN internships ON matches.internship_id = internships.id
        WHERE matches.user_id = ?
        ORDER BY matches.score DESC
        """,
        (user_id,)
    )
    results = cur.fetchall()
    conn.close()
    return results


if __name__ == "__main__":
    create_tables()
    print("✅ Database and tables initialized successfully!")



