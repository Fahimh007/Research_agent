import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


def main():
    print("Research Agent started.")


if __name__ == "__main__":
    main()
