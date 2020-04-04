<!-- ![alt text](https://github.com/rameznahas/Natural-Language-Processing/blob/master/Project%20Instructions/COMP_472_2020_Winter_Project_2-v2-1.png)
![alt text](https://github.com/rameznahas/Natural-Language-Processing/blob/master/Project%20Instructions/COMP_472_2020_Winter_Project_2-v2-2.png)
![alt text](https://github.com/rameznahas/Natural-Language-Processing/blob/master/Project%20Instructions/COMP_472_2020_Winter_Project_2-v2-3.png)
![alt text](https://github.com/rameznahas/Natural-Language-Processing/blob/master/Project%20Instructions/COMP_472_2020_Winter_Project_2-v2-4.png)
![alt text](https://github.com/rameznahas/Natural-Language-Processing/blob/master/Project%20Instructions/COMP_472_2020_Winter_Project_2-v2-5.png)
![alt text](https://github.com/rameznahas/Natural-Language-Processing/blob/master/Project%20Instructions/COMP_472_2020_Winter_Project_2-v2-6.png) -->
## GitHub repo:  
https://github.com/rameznahas/Natural-Language-Processing

## How to run the program:  
(from the CLI) python Model.py [v] [n] [delta] [training_file] [testing_file] [word_boundary]  
* [v]: (int) vocabulary to use, v = [0-3] where v = 3 is our own vocabulary consisting of [a-zA-Z] + all the diacritics of the languages
* [n]: (int) size of the n-grams, n > 0  
* [delta]: (float) smoothing value, delta = [0.0-1.0]  
* [training_file]: (str) absolute path of the file used for training  
* [testing_file]: (str) absolute path of the file used for testing  
* [word_boundary]: (str) flag indicating if word boundary should be part of the n-grams, word_boundary = ["on", "off"]
