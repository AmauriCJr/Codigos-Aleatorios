import requests
import json

i = 1
while i < 6:
  	print(i)

	r = requests.get('https://httpbin.org/get')

	d = r.json()

	x = d['origin']

  	i += 1

	print (d)

	print(x)

