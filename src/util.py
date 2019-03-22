import os
ROOT = '/data/jrgillick/projects/assisted_orchestration/TinySOL/TinySOL'

def wavFileGen(rootdir=ROOT):
	for root, dirs, files in os.walk(rootdir):
		for filename in files:
			if os.path.splitext(filename)[1] == ".wav":
				yield os.path.join(root, filename)
