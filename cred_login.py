import webbrowser
from fyers_apiv3 import fyersModel

# Replace these with your Fyers app credentials
client_id = "39SPDCHRQG-100"
secret_key = "9MHJVUI3O8"
redirect_uri = "https://trade.fyers.in/api-login/redirect-uri/index.html"
response_type = "code"
grant_type = "authorization_code"

# Step 1: Create session
session = fyersModel.SessionModel(
    client_id=client_id,
    secret_key=secret_key,
    redirect_uri=redirect_uri,
    response_type=response_type,
    grant_type=grant_type
)

# Step 2: Generate Auth Code URL and open in browser
auth_url = session.generate_authcode()
print(f"\n🔗 Open this URL in your browser to log in and get the auth code:\n{auth_url}\n")

# Automatically open the URL in your default browser
webbrowser.open(auth_url)

# Step 3: After login, you’ll be redirected to the redirect_uri with ?code=AUTH_CODE in URL
# Paste that code here manually
auth_code = input("📥 Paste the auth code from the redirect URL: ").strip()

# Step 4: Set token and generate access token
session.set_token(auth_code)
response = session.generate_token()

# Step 5: Print the final access token response
print("\n✅ Access Token Response:")
print(response)
