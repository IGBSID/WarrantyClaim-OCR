import os

# Authenticate header

def authenticate(token):
        # auth = headers.get("X-Api-Key")
    print("Token = " + os.environ["AUTH_TOKEN"])
    if token == os.environ["AUTH_TOKEN"]:
        return True
    else:
        return False
