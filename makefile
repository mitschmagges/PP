CFLAGS+= -std=c99

# by uncommenting this line the preprocessor will see #ifdef DEBUG as true
# CFLAGS+= -DDEBUG
objects = bitmap.o julia.o
SHELL := /bin/bash

# When running "make" in your terminal, the first target will be chosen.
# The syntax of make is basically:
# name : [stuff I need to make this target]

# In this case the dependencies are easy to figure out, so I will not elaborate further
all : $(objects)
	gcc $(objects) -o julia

# In this target [stuff I need to make this target] is two other targets, namely clean and all
# This command simply runs the clean target, then the all target, thus recompiling everything.
remake : clean all

# We add .PHONY when a target doesn't actually create any output. In this case we just run a shell
# command, removing all object files, i.e files ending on .o
# the * syntax means [anything].o
.PHONY : clean
clean :
	rm -f *.o && rm -f *.gch
	# remove for force (no error, if doesent exsist) all files with .o
	# remove for force all files with .gch

# Finally, the test target. Builds the 'all' target, then runs the test script on the output
.PHONY : test #"this doesn reals produce anything"
test : all
	./test.sh myProgram
	#runs the test.sh programm (shellscript)