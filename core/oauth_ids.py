"""Sign-in with GitHub and Google needs an "app" registered by the developer. Paste the public IDs here (or put them in
data/oauth.json / environment variables BV_GITHUB_CLIENT_ID, BV_GOOGLE_CLIENT_ID, BV_GOOGLE_CLIENT_SECRET). See docs/SIGN_IN_SETUP.md.
Client IDs are public by design. Do not put a GitHub client *secret* anywhere: the device flow used here does not need one."""
GITHUB_CLIENT_ID = ""
GOOGLE_CLIENT_ID = ""
GOOGLE_CLIENT_SECRET = ""   # for Google "Desktop app" clients this value is not confidential
