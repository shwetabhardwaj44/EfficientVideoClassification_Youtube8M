import json, md5, os

f = open('frame_level_test.json')
hashes = json.load(f)
hashes = hashes['files']
address = 'http://us.data.yt8m.org/1/frame_level/test/'
i = 0
for f, h in hashes.items():
	i += 1
	if i % 10 == 0:
		print i
	m = md5.new()
	m.update(open(f).read())
	hs = m.hexdigest()
	if h != hs:
		download_url = address+f
		os.system('curl %s > %s' % (download_url, f))
