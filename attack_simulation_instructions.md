# Instructions for Simulating HTTP-Based Attacks on the Honeypot

These instructions and example scripts will help your friend simulate HTTP-based attacks that will be logged by the honeypot application.

## 1. Access Non-Existent URLs (Trigger 404 Logging)

Use curl or a browser to access URLs that do not exist on the honeypot server. This will trigger 404 errors and log the attempts.

Example curl command:
```bash
curl -i http://<honeypot-ip>:5000/nonexistentpage
```

## 2. Simulate SQL Injection Attempts

Send HTTP requests with typical SQL injection payloads in the URL or POST data.

Example curl command:
```bash
curl -i "http://<honeypot-ip>:5000/login?username=admin' OR '1'='1&password=anything"
```

Or using POST:
```bash
curl -i -X POST http://<honeypot-ip>:5000/login -d "username=admin' OR '1'='1&password=anything"
```

## 3. Simulate Cross-Site Scripting (XSS) Attempts

Send requests with typical XSS payloads in parameters.

Example curl command:
```bash
curl -i "http://<honeypot-ip>:5000/search?q=<script>alert('XSS')</script>"
```

## 4. Simulate Brute Force Login Attempts

Send multiple login attempts with invalid credentials to trigger brute force detection and IP blocking.

Example bash script:
```bash
for i in {1..5}
do
  curl -i -X POST http://<honeypot-ip>:5000/login -d "username=admin&password=wrongpassword"
done
```

## 5. Use the Attack Simulator UI

If your friend has admin access, they can use the attack simulator page at:

```
http://<honeypot-ip>:5000/attack-simulator
```

From there, they can select attack types such as SQL injection, brute force, path traversal, XSS, port scan, and command injection, and submit payloads to simulate attacks.

## 6. Python Script Example for Simulating Attacks

```python
import requests

honeypot_url = "http://<honeypot-ip>:5000"

def simulate_sql_injection():
    payload = {"username": "admin' OR '1'='1", "password": "anything"}
    response = requests.post(f"{honeypot_url}/login", data=payload)
    print("SQL Injection simulation response:", response.status_code)

def simulate_brute_force():
    for i in range(5):
        payload = {"username": "admin", "password": "wrongpassword"}
        response = requests.post(f"{honeypot_url}/login", data=payload)
        print(f"Brute force attempt {i+1} response:", response.status_code)

def simulate_xss():
    params = {"q": "<script>alert('XSS')</script>"}
    response = requests.get(f"{honeypot_url}/search", params=params)
    print("XSS simulation response:", response.status_code)

if __name__ == "__main__":
    simulate_sql_injection()
    simulate_brute_force()
    simulate_xss()
```

Replace `<honeypot-ip>` with the actual IP address or hostname of the honeypot server.

---

These simulated attacks will be logged by the honeypot and can be viewed in the admin dashboard or attack logs.

If you want, I can help you create more advanced or customized attack simulation scripts.
