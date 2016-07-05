Reference for labeling and terms used herein:
	- BR: Bathroom
	- LH: Lecture Hall
	- P: Pantry
	- O: Office
	- LB: Library		|
	- FC: Food Court	|  Used to refer to location of the bathrooms
	- C: Center			|
	- Smartphone 1: Galaxy SIII mini
	- Smartphone 2: Sony Xperia Z1 compact
	- Professional Recorder: Olympus Linear PCM LS -10
	- Intra room locations:
		 ___________________________________________
		|*L0								   	 L1*|
		|											|
		|											|
		|											|
		|											|
		|					*L4						|
		|											|
		|											|
		|											|
		|											|
		|*L2								   	 L3*|                                
		|_______________________________/  _________|
	- S0: "Hello, How are you doing?"
	- S1: "I am fine, thank you"
	- S2: "Have a good day sir"
	- S3: "So dark the con of man"

Method:
	Recordings of 4 sentences were obtained from each room at 5 different locations within the room. The recordings were done with smartphone 1 in the speaker's hand and with smartphone 2 in the speaker's front pocket. A professional recorder was also used to record 20 samples at LOC4 within each room.

Recording Naming Scheme:
	- RoomType_RoomIdentifier_Pocket/Hand_Location_sentenceNumber_recordingsNumber.wav
	- eg: Recording from Lecture Hall 1190 with phone in hand with the speaker saying sentence 0 at location 4 for the 3rd time.
				LH_1190_H_L4_S0_2.wav
	- Extention Definitions:
		> .mfcc: has 20D MFCC features
		> .fbe: has 20D Mel-Filterbank energy values
		> .feat: has 20D MFCC features but excluding NaN present in the .mfcc file
		> .Dfeat: the nth MFCC feature vector is augmented by appending the delta w.r.t the n-1th and n+1th feature vectors producing a 60D vector.
		> .Nfeat: The feature values are normalized by subtracting taking the difference from the mean of the values the respective feature takes withing the corresponding .mfcc file.
		> .Cfeat: The process is same as for the .Nfeat but only the mean of the first feature is used.
		> .quant: Histogram values based on the count of feature vectors in the file that are assigned to a particular centroid.
											[c1,...,cn]
		c1 are the number of features assigned to centroid at index 0 in our codebook
			* .quant is preceeded by one of the feat extextion to indicate which type of features were used in the generating the histogram.
			* .Nquant, Cquant and Dquant also refer to the type of .feat files used
		> .svm: files are used to train svms




