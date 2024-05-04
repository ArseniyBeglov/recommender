import requests

url = 'http://127.0.0.1:8000/user/api/v1/recommendations/make/'
data = {}
response_1 = requests.post(url, json=data)
url = 'http://127.0.0.1:8000/user/api/v1/recommendations/make/'
response_2 = requests.get(url)
