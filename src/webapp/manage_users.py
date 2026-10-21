import argparse
import getpass
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

from webapp.auth import (
    assign_patient_to_staff,
    create_user,
    initialize_auth_db,
)


def main():
    parser = argparse.ArgumentParser(
        description="Provision accounts for the local/private pilot."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Create the local authentication database")

    create_parser = subparsers.add_parser("create-user", help="Create a patient or staff account")
    create_parser.add_argument("--username", required=True)
    create_parser.add_argument("--role", choices=("patient", "staff"), required=True)
    create_parser.add_argument("--patient-id")

    assign_parser = subparsers.add_parser(
        "assign-patient",
        help="Allow a staff account to access an existing patient account",
    )
    assign_parser.add_argument("--staff-username", required=True)
    assign_parser.add_argument("--patient-id", required=True)

    args = parser.parse_args()
    initialize_auth_db()

    if args.command == "init":
        print("Authentication database initialized.")
    elif args.command == "create-user":
        password = getpass.getpass("New password (minimum 12 characters): ")
        confirmation = getpass.getpass("Confirm password: ")
        if password != confirmation:
            parser.error("Passwords do not match.")
        try:
            user = create_user(
                args.username,
                password,
                args.role,
                patient_id=args.patient_id,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(f"Created {user['role']} account '{user['username']}'.")
        if user["patient_id"]:
            print(f"Patient ID: {user['patient_id']}")
    else:
        try:
            assign_patient_to_staff(args.staff_username, args.patient_id)
        except ValueError as exc:
            parser.error(str(exc))
        print("Patient access assigned to staff account.")


if __name__ == "__main__":
    main()
