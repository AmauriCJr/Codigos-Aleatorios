import requests
import json


r = requests.get('https://httpbin.org/get')

d = r.json()

x = d['origin']

print (d)

print(x)


payload = {'naipe': 'copas', 'numero': '4'}
t = requests.post("http://httpbin.org/post", data=payload)
print t.text


url = 'https://api.github.com/some/endpoint'
payload = {'some': 'data'}

r = requests.post(url, data=json.dumps(payload))

d = r.json()

print(d)
