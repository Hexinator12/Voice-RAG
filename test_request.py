import requests
print(requests.post("http://127.0.0.1:8001/query",
                    json={"q":"test body","top_k":1}).status_code,
      requests.post("http://127.0.0.1:8001/query",
                    json={"q":"test body","top_k":1}).text)
