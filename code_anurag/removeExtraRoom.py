def removeExtraRoom(F, roomNo):
	f = open(F,'r')
	lines = f.readlines()
	f.close()

	lines = filter(lambda x: roomNo not in x and "feat64ms" in x, lines)

	f = open(F,'w')
	for line in lines:
		f.write(line)
	f.close()

removeExtraRoom("results/SV/fold_3/O/P/results.txt", "1018")