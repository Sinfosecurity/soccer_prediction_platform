def authenticate_user(username: str, password: str):
    # Dummy implementation for testing
    if username == "username" and password == "password":
        return {"username": username}
    return None
