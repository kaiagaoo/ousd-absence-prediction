import os

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    print("Project folders initialized.")

if __name__ == "__main__":
    main()
