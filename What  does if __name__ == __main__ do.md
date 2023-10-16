I have always this in Python but never actually knew what it does.

[link to stackoverflow answer](https://stackoverflow.com/a/419185)
### Short Answer
boilerplate code that protects from accidentally invoking the script when they didn't intend to
- if you import guardless script in another srit, then the latter script will trigger the former to run at import time and using the second script's command line arguments. 

### Long Answer
This is what one of my proffs told me
Whenever the python interpreter reads a source file, it does two things:
- it sets a few special variables like `__name__` 
- executes all of the code found in the file 
and `__main__` is like the int main file in c++ and will look for it as it is the main function and will start executing code blocks from that 
