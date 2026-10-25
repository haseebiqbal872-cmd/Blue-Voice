# Switching on "Continue with GitHub" and "Continue with Google"

Blue Voice has no server. To sign people in, GitHub and Google need to know about your app once. This takes about 10 minutes and costs nothing.
Until you do it, the buttons say sign-in is not set up and people can continue without signing in.

## GitHub (device flow, no secret needed)
1. GitHub → Settings → Developer settings → OAuth Apps → **New OAuth App**.
2. Name: `Blue Voice`. Homepage: your repo URL. Callback URL: `http://127.0.0.1:8765/` (not used, but required).
3. After creating it, tick **Enable Device Flow** and save.
4. Copy the **Client ID** (it is public; do not create or share a client secret).

## Google (browser + PKCE)
1. https://console.cloud.google.com → create a project → **APIs & Services → OAuth consent screen** (External, add your email as a test user while testing).
2. **Credentials → Create credentials → OAuth client ID → Desktop app**.
3. Copy the **Client ID** and **Client secret**. For Desktop apps Google says the secret is not confidential, but keep it out of public commits if you can (use `data/oauth.json`).

## Tell Blue Voice
Either edit `core/oauth_ids.py`, or create `data/oauth.json` (git-ignored):

    {"github_client_id": "Ov23...", "google_client_id": "1234...apps.googleusercontent.com", "google_client_secret": "GOCSPX-..."}

or set the environment variables `BV_GITHUB_CLIENT_ID`, `BV_GOOGLE_CLIENT_ID`, `BV_GOOGLE_CLIENT_SECRET`.

## What Blue Voice reads
Only your name and email, to show who is signed in. No token is kept. Nothing is sent to any Blue Voice server because there is none.
