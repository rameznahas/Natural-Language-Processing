import re, sys, time, pickle
from math import log10

class Model():
	"""
	This class is a natural language processor based on the Naive Bayes classification and n-gram model. 
	It classifies tweets according to their languages, as determined by this model.
	"""

	link_pattern = re.compile(r"(http|[@#])\S+")
	cc_pattern = re.compile(r"\b[cC]{2}\b")
	rt_pattern = re.compile(r"\b[rR][tT]\b")
	punctuation_pattern = re.compile(r"[^\w\s]|_")
	extra_ws_pattern = re.compile(r"\s{2,}")

	def __init__(self, v, n, delta, training_file, testing_file):
		"""
		Parameterized constructor.

		Parameters
		----------
		v : int
			indicates the vocabulary to use
		n : int
			indicates the size of the n-gram
		delta : float
			indicates the smoothing value used for additive smoothing
		training_file : str
			indicates the file used for training the model
		testing_file : str
			indicates the file used for testing the model
		"""
		self.v = v
		self.n = n
		self.delta = delta
		self.training_file = training_file
		self.testing_file = testing_file
		self.prefix_idx = n - 1
		self.ngrams = {"eu": {}, "ca": {}, "gl": {}, "es": {}, "en": {}, "pt": {}}
		self.ngrams_total = {"eu": {}, "ca": {}, "gl": {}, "es": {}, "en": {}, "pt": {}}

		self.langs = {"eu": 0, "ca": 0, "gl": 0, "es": 0, "en": 0, "pt": 0}
		self.confusion_matrix = {
			"eu": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}, 
			"ca": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}, 
			"gl": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}, 
			"es": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}, 
			"en": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}, 
			"pt": {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
			}
		self.__set_vocab()
		self.ngrams_total_increment = self.delta * self.vocab_size
		self.trace_output = ''
		self.correct_classifications = 0

	def __set_vocab(self):
		"""
		Determines the pattern (for regex) and the size of the chosen vocabulary.
		"""
		vocab = ''
		a, z, A, Z = ord('a'), ord('z'), ord('A'), ord('Z')

		if self.v == 0:
			vocab = r"[a-z]"
			self.vocab_size = 1 + z - a

		elif self.v == 1: 
			vocab = r"[a-zA-Z]"
			self.vocab_size = 2 + z - a + Z - A

		elif self.v == 2:
			# here vocab is any none-whitespace character since we will
			# use the function isalpha() instead of regex to find all the ngrams
			vocab = r"\S"
			self.vocab_size = 0
			unicode_planes = 17
			code_points_per_plane = 2**16

			for code_point in range(unicode_planes * code_points_per_plane):
				char = chr(code_point)
				if char.isalpha():
					self.vocab_size += 1

		else:
			diacritics = r"\U000000C0-\U000000C3\U000000C7-\U000000CA\U000000CD\U000000CF\U000000D1-\U000000D5\U000000DA\U000000DC\U000000E1-\U000000E4\U000000E7-\U000000EA\U000000ED\U000000EF\U000000F1-\U000000F5\U000000FA\U000000FC"
			vocab = r"[a-zA-Z" + diacritics
			diacritics_count = 34
			self.vocab_size = 2 + z - a + Z - A + diacritics_count

			if self.v == 3:
				vocab += "]"
			else:
				vocab += " ]"
				self.vocab_size += 1
			
		self.vocab_pattern = re.compile(r"(?=(" + vocab + "{" + str(self.n) + "}))")

	def __clean(self, tweet):
		"""
		Cleans the tweet by removing links (URLS, mentions and hashtags), carbon copies, retweets and extra whitespaces.

		Parameters
		----------
		tweet : str
			the tweet to clean

		Returns
		-------
		str
			the cleaned tweet
		"""
		tweet = self.link_pattern.sub('', tweet)
		tweet = self.cc_pattern.sub('', tweet)
		tweet = self.rt_pattern.sub('', tweet)
		tweet = self.punctuation_pattern.sub(' ', tweet)
		tweet = self.extra_ws_pattern.sub(' ', tweet)
		tweet = ' ' + tweet.strip() + ' '

		return tweet

	def train(self):
		"""
		Trains the model using the self.training_file as the training corpus and self.vocab_pattern as the ngram model.
		"""
		file = open(self.training_file, encoding="utf-8")
		file_buffer = file.readlines()
		file.close()

		tweet_count = len(file_buffer)
		# if true, make all tweets lowercase
		if self.v == 0: 
			file_buffer = [file_buffer[i].lower() for i in range(tweet_count)]

		# for each line in the file_buffer
		for line in file_buffer:
			# split the line into a list of data
			data = line.split('\t')
			# language of the tweet
			lang = data[2]
			# increment the language count
			self.langs[lang] += 1

			# get the cleaned tweet
			tweet = self.__clean(data[3])
			# get the list of all possible ngrams from the tweet
			ngrams = self.vocab_pattern.findall(tweet)
			for ngram in ngrams:
				if self.v != 2 or ngram.isalpha():
					# get the ngram prefix
					prefix = ngram[0 : self.prefix_idx]
					# increment the count of this ngram
					self.ngrams[lang][ngram] = self.ngrams[lang].get(ngram, self.delta) + 1
					# increment the total
					self.ngrams_total[lang][prefix] = self.ngrams_total[lang].get(prefix, self.ngrams_total_increment) + 1

		# convert values to probabilities
		self.langs = {lang: count / tweet_count for lang, count in self.langs.items()}

		# convert values to probabilities
		for lang in self.ngrams:
			for ngram in self.ngrams[lang]:
				prefix = ngram[0 : self.prefix_idx]
				self.ngrams[lang][ngram] /= self.ngrams_total[lang][prefix]

	def __trace_output(self, id, classification, score, actual):
		"""
		This method updates the buffer that will be written to the trace file.

		Parameters
		----------
		id : str
			id of the tweet
		classification : str
			this model's classification the tweet
		score : float
			the score of the classification
		actual : str
			the actual classification of the tweet
		"""
		if classification == actual:
			self.trace_output += id + "  " + classification + "  " + "{:.2e}".format(score) + "  " + actual + "  correct\n"
			self.correct_classifications += 1
		else:
			if score == float("-inf"):
				score = -999999999 
			self.trace_output += id + "  " + classification + "  " + "{:.2e}".format(score) + "  " + actual + "  wrong\n"

	def __eval_output(self, actual_lang, classification):
		"""
		This method updates the confusion matrix which is used for the evaluation file.

		Parameters
		----------
		actual_lang : str
			the actual classification of the tweet
		classification : str
			this model's classification of the tweet
		"""
		for lang in self.confusion_matrix:
			if actual_lang != lang:
				if classification != lang: self.confusion_matrix[lang]["tn"] += 1
				else: self.confusion_matrix[lang]["fp"] += 1
			else:
				if classification != lang: self.confusion_matrix[lang]["fn"] += 1
				else: self.confusion_matrix[lang]["tp"] += 1


	def __generate_output_files(self, tweet_count):
		"""
		This method generates the trace and evaluation files.

		Parameters
		----------
		tweet_count : int
			number of tweets in the testing file
		"""
		#file = open("trace_" + str(self.v) + "_" + str(self.n) + "_" + str(self.delta) + ".txt", 'w', encoding="utf-8")
		#file.write(self.trace_output)
		#file.close

		precision = {"eu": 0, "ca": 0, "gl": 0, "es": 0, "en": 0, "pt": 0}
		recall = {"eu": 0, "ca": 0, "gl": 0, "es": 0, "en": 0, "pt": 0}
		f1 = {"eu": 0, "ca": 0, "gl": 0, "es": 0, "en": 0, "pt": 0}
		weights = {"eu": 0, "ca": 0, "gl": 0, "es": 0, "en": 0, "pt": 0}

		# populate the precision, recall, f1 and weights dictionaries from self.confusion matrix
		for lang, cm in self.confusion_matrix.items():
			classifications_count = cm["tp"] + cm["fp"]
			actual_count = cm["tp"] + cm["fn"]

			if classifications_count > 0:
				precision[lang] = cm["tp"] / classifications_count

			if actual_count > 0:
				recall[lang] = cm["tp"] / actual_count

			denom = precision[lang] + recall[lang]
			if denom > 0:
				f1[lang] = 2 * precision[lang] * recall[lang] / denom

			weights[lang] = actual_count / tweet_count

		macro_f1 = sum(f1.values()) / len(f1)
		weighted_avg_f1 = 0
		for lang, weight in weights.items():
			weighted_avg_f1 += weight * f1[lang]

		number = "{:.4f}"
		eval_output = str(number.format(self.correct_classifications / tweet_count)) + '\n'
		eval_output += str(number.format(precision["eu"])) + "  " + str(number.format(precision["ca"])) + "  " + str(number.format(precision["gl"])) + "  " + str(number.format(precision["es"])) + "  " + str(number.format(precision["en"])) + "  " + str(number.format(precision["pt"])) + '\n'
		eval_output += str(number.format(recall["eu"])) + "  " + str(number.format(recall["ca"])) + "  " + str(number.format(recall["gl"])) + "  " + str(number.format(recall["es"])) + "  " + str(number.format(recall["en"])) + "  " + str(number.format(recall["pt"])) + '\n'
		eval_output += str(number.format(f1["eu"])) + "  " + str(number.format(f1["ca"])) + "  " + str(number.format(f1["gl"])) + "  " + str(number.format(f1["es"])) + "  " + str(number.format(f1["en"])) + "  " + str(number.format(f1["pt"])) + '\n'
		eval_output += str(number.format(macro_f1)) + "  " + str(number.format(weighted_avg_f1))

		#file = open("eval_" + str(self.v) + "_" + str(self.n) + "_" + str(self.delta) + ".txt", 'w', encoding="utf-8")
		#file.write(eval_output)
		#file.close()

		file = open("test.txt", 'a', encoding="utf-8")
		file.write(str(self.v) + "\t\t" + str(self.n) + "\t\t" + str(self.delta) + "\t\t" + str(number.format(self.correct_classifications * 100 / tweet_count)) + "%\n")
		file.close()

	def __argmax(self, tweet):
		"""
		This method applies Naives Bayes to classify the tweet.

		Parameters
		----------
		tweet : str
			the tweet to classify

		Returns
		-------
		str, float
			the classification of the tweet, the score of the classification
		"""
		score = float("-inf")
		# default value of the classification in case we cannot classify the tweet,
		# which can happen when the smoothing value is 0 and an ngram
		# was found in the test file but not the training file, for all classes
		classification = "xx" 
		# get the list of all possible ngrams from the tweet
		ngrams = self.vocab_pattern.findall(tweet)
		smoothed_prob = 0
		for lang in self.ngrams:
			likelihood = 0

			try:
				for ngram in ngrams:
					if self.v != 2 or ngram.isalpha():
						prefix = ngram[0 : self.prefix_idx]
						smoothed_prob = self.delta / self.ngrams_total[lang].get(prefix, self.ngrams_total_increment)
						likelihood += log10(self.ngrams[lang].get(ngram, smoothed_prob))
			except:
				likelihood = float("-inf")

			probability = log10(self.langs[lang]) + likelihood

			# if true, this is the best classification so far, save it.
			if probability > score:
				score = probability
				classification = lang

		return classification, score

	def test(self):
		"""
		Test the model using the self.testing_file.
		"""
		file = open(self.testing_file, encoding="utf-8")
		file_buffer = file.readlines()
		file.close()

		# for each line in the file_buffer
		for line in file_buffer:
			# split the line into a list of data
			data = line.split('\t')
			# get the cleaned tweet
			tweet = self.__clean(data[3])
			# get the classification and the score of the classification
			classification, score = self.__argmax(tweet)
			self.__trace_output(data[0], classification, score, data[2])
			self.__eval_output(data[2], classification)

		self.__generate_output_files(len(file_buffer))

if __name__ == "__main__":
	#model = Model(int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5])
	#model.train()
	#model.test()

	###########################################testing/debugging#########################################
	file = open("test.txt", 'w', encoding="utf-8")
	file.write("v\t\tn\t\td\t\taccuracy\n")
	file.close()
	performance = ''

	#d = 0.001
	#while d < 0.1:
	#	model = Model(4, 3, d, "D:/Downloads/OriginalDataSet/training-tweets.txt", "D:/Downloads/OriginalDataSet/test-tweets-given.txt")
	#	start = time.time()
	#	model.train()
	#	model.test()
	#	end = time.time()
	#	performance += str(model.v) + " " + str(model.n) + " " + str(model.delta) + "\ttime: " + str(end - start) + '\n'
	#	d += 0.001

	for v in range(5):
		for n in range(1, 4):
			model = Model(v, n, 0.0, "D:/Downloads/OriginalDataSet/training-tweets.txt", "D:/Downloads/OriginalDataSet/test-tweets-given.txt")
			start = time.time()
			model.train()
			model.test()
			end = time.time()
			performance += str(v) + " " + str(n) + " " + str(model.delta) + "\ttime: " + str(end - start) + '\n'
			model = Model(v, n, 0.1, "D:/Downloads/OriginalDataSet/training-tweets.txt", "D:/Downloads/OriginalDataSet/test-tweets-given.txt")
			start = time.time()
			model.train()
			model.test()
			end = time.time()
			performance += str(v) + " " + str(n) + " " + str(model.delta) + "\ttime: " + str(end - start) + '\n'
			model = Model(v, n, 0.2, "D:/Downloads/OriginalDataSet/training-tweets.txt", "D:/Downloads/OriginalDataSet/test-tweets-given.txt")
			start = time.time()
			model.train()
			model.test()
			end = time.time()
			performance += str(v) + " " + str(n) + " " + str(model.delta) + "\ttime: " + str(end - start) + '\n'

	file = open("optim_perf.txt", 'w', encoding="utf-8")
	file.write(performance)
	file.close()