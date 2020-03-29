Going to have to setup your directories to get this working.

Open the project

Right click the project on the left tab
	>Options..>C/C++ Compiler>Preprocessor

Look at the include directories
$PROJ_DIR$ is the EWARM directory
/../ means you go back one directory
With this logic, you can setup the corresponding libraries
the first /../Inc works for you
You're going to have to setup the Drivers and Utilities dirs

Mine are in a folder called Libraries, which is 2 directories back from EWARM

Good luck.