# src/webapp/app.py
from pathlib import Path
import logging
import hmac
import os
import secrets
import sys
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from flask import Flask, g, request, jsonify, render_template, session

from agents.router import handle_message
from patient_storage.data_crypto import get_encryption_key
from webapp.auth import (
    authenticate_user,
    can_access_patient,
    get_accessible_patient_ids,
    get_user,
)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("DARJA_SESSION_SECRET"),
    MAX_CONTENT_LENGTH=8 * 1024,
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=20),
    SESSION_REFRESH_EACH_REQUEST=False,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    SESSION_COOKIE_SECURE=os.environ.get("DARJA_HTTPS_ONLY") == "1",
)
logger = logging.getLogger(__name__)


@app.after_request
def add_security_headers(response):
    response.headers["Cache-Control"] = "no-store"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "base-uri 'self'; "
        "form-action 'self'; "
        "frame-ancestors 'none'; "
        "object-src 'none'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "connect-src 'self'; "
        "img-src 'self' data:"
    )
    return response


@app.before_request
def load_current_user():
    user_id = session.get("user_id")
    g.current_user = get_user(user_id) if user_id is not None else None
    if user_id is not None and g.current_user is None:
        session.clear()


def json_body():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "request body must be a JSON object"}), 400)
    return data, None


def csrf_error():
    token = session.get("csrf_token")
    supplied = request.headers.get("X-CSRF-Token", "")
    if not token or not supplied or not hmac.compare_digest(token, supplied):
        return jsonify({"error": "invalid or missing CSRF token"}), 400
    return None


def require_session_secret():
    if not app.config["SECRET_KEY"]:
        return jsonify({"error": "authentication is not configured"}), 503
    return None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/auth/csrf", methods=["GET"])
def csrf_token():
    configuration_error = require_session_secret()
    if configuration_error:
        return configuration_error
    if not session.get("csrf_token"):
        session["csrf_token"] = secrets.token_urlsafe(32)
    return jsonify({"csrf_token": session["csrf_token"]})


@app.route("/api/auth/login", methods=["POST"])
def login():
    configuration_error = require_session_secret()
    if configuration_error:
        return configuration_error
    csrf_failure = csrf_error()
    if csrf_failure:
        return csrf_failure
    data, error = json_body()
    if error:
        return error
    username = data.get("username")
    password = data.get("password")
    if not isinstance(username, str) or not isinstance(password, str):
        return jsonify({"error": "username and password must be strings"}), 400

    user = authenticate_user(username, password)
    if user is None:
        return jsonify({"error": "invalid username or password"}), 401

    session.clear()
    session.permanent = True
    session["user_id"] = user["id"]
    session["csrf_token"] = secrets.token_urlsafe(32)
    return jsonify({
        "username": user["username"],
        "role": user["role"],
        "csrf_token": session["csrf_token"],
    })


@app.route("/api/auth/me", methods=["GET"])
def current_user():
    configuration_error = require_session_secret()
    if configuration_error:
        return configuration_error
    if g.current_user is None:
        return jsonify({"error": "authentication required"}), 401
    return jsonify({
        "username": g.current_user["username"],
        "role": g.current_user["role"],
    })


@app.route("/api/auth/logout", methods=["POST"])
def logout():
    configuration_error = require_session_secret()
    if configuration_error:
        return configuration_error
    if g.current_user is None:
        return jsonify({"error": "authentication required"}), 401
    csrf_failure = csrf_error()
    if csrf_failure:
        return csrf_failure
    session.clear()
    return jsonify({"logged_out": True})


@app.route("/api/patients", methods=["GET"])
def patients():
    configuration_error = require_session_secret()
    if configuration_error:
        return configuration_error
    if g.current_user is None:
        return jsonify({"error": "authentication required"}), 401
    if g.current_user["role"] != "staff":
        return jsonify({"error": "staff account required"}), 403
    return jsonify({"patient_ids": get_accessible_patient_ids(g.current_user["id"])})


@app.route("/api/chat", methods=["POST"])
def chat():
    configuration_error = require_session_secret()
    if configuration_error:
        return configuration_error
    if g.current_user is None:
        return jsonify({"error": "authentication required"}), 401
    csrf_failure = csrf_error()
    if csrf_failure:
        return csrf_failure
    data, error = json_body()
    if error:
        return error

    message = data.get("message", "")
    if not isinstance(message, str):
        return jsonify({"error": "message must be a string"}), 400
    message = message.strip()

    if not message:
        return jsonify({"error": "empty message"}), 400

    if g.current_user["role"] == "patient":
        patient_id = g.current_user["patient_id"]
        requested_patient_id = data.get("patient_id")
        if requested_patient_id is not None and not isinstance(requested_patient_id, str):
            return jsonify({"error": "patient_id must be a string"}), 400
        if requested_patient_id is not None and requested_patient_id != patient_id:
            return jsonify({"error": "patient access denied"}), 403
    else:
        patient_id = data.get("patient_id")
        if not isinstance(patient_id, str) or not patient_id.strip():
            return jsonify({"error": "staff must select an assigned patient"}), 400
        patient_id = patient_id.strip()
        if not can_access_patient(g.current_user, patient_id):
            return jsonify({"error": "patient access denied"}), 403

    try:
        result = handle_message(message, patient_id=patient_id)
    except Exception as exc:
        request_id = secrets.token_hex(8)
        logger.error(
            "Failed to process chat request (request_id=%s, error_type=%s)",
            request_id,
            type(exc).__name__,
        )
        return jsonify({
            "error": "internal server error",
            "request_id": request_id,
        }), 500

    return jsonify({
        "response": result.get("response", ""),
        "route": result.get("route"),
        "intent": result.get("intent"),
        "confidence": result.get("confidence"),
    })


if __name__ == "__main__":
    get_encryption_key()
    app.run(host="127.0.0.1", debug=False, port=5000)