import ngrok
import time

# Authenticate (optional, can also use NGROK_AUTHTOKEN environment variable)
# ngrok.set_auth_token("<YOUR_AUTHTOKEN>") 

# Establish connectivity to port 9000
listener = ngrok.forward(9000, authtoken_from_env=True)

# Output ngrok url to console
print(f"Ingress established at {listener.url()}")

# Keep the listener alive
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Closing listener")
