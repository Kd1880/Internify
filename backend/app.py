from flask import Flask, request, jsonify
from flask_cors import CORS
from backend.recommender_pipeline import process_resume
from backend.db_handler import add_user, get_user, get_matches_for_user, create_tables
import os


app = Flask(__name__)
CORS(app)  # Enable CORS so you can call the API from Postman or a web UI


UPLOAD_FOLDER = "uploads"  # Project-root uploads for API file saves
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Ensure folder exists

# Ensure required tables exist at startup (idempotent)
create_tables()


@app.route("/")
def home():
    """
    Health check endpoint.
    Returns a simple message to confirm the API is live.
    """
    return jsonify({"message": "Internify API is running"})


@app.route("/signup", methods=["POST"])
def signup():
    """
    Registers a new user in SQLite database.
    Expects JSON: { name, email, password }.
    """
    data = request.json
    name = data.get("name")
    email = data.get("email")
    password = data.get("password")

    success = add_user(name, email, password)
    if success:
        return jsonify({"message": "User registered successfully"}), 201
    else:
        return jsonify({"error": "Email already exists"}), 400


@app.route("/login", methods=["POST"])
def login():
    """
    Authenticates a user by email/password.
    Expects JSON: { email, password }.
    Returns the user_id on success.
    """
    data = request.json
    email = data.get("email")
    password = data.get("password")
    user = get_user(email, password)
    if user:
        return jsonify({"message": "Login successful", "user_id": user[0]})
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@app.route("/upload_resume", methods=["POST"])
def upload_resume():
    """
    Accepts a PDF resume file and user_id (multipart form-data).
    Runs the ML pipeline and returns top matches.
    """
    user_id = request.form.get("user_id")
    file = request.files.get("file")

    if not user_id or not file:
        return jsonify({"error": "Missing user_id or file"}), 400

    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    try:
        results_df = process_resume(file_path, int(user_id))
        return jsonify({
            "message": "Resume processed successfully",
            "results": results_df.to_dict(orient="records")
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/matches", methods=["GET"])
def matches():
    """
    Returns saved matches for a given user_id.
    Response is a JSON list of {company, title, final_score, cluster}.
    """
    user_id = request.args.get("user_id")
    if not user_id:
        return jsonify({"error": "Missing user_id"}), 400
    try:
        rows = get_matches_for_user(int(user_id))
        payload = [
            {"company": r[0], "title": r[1], "final_score": float(r[2]), "cluster": int(r[3])}
            for r in rows
        ]
        return jsonify(payload)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Start Flask development server
    app.run(debug=True, use_reloader=False)
