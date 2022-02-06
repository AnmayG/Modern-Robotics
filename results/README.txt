README

Hello! This is my Modern Robotics Course 6 Capstone project, consisting of a simulated robot with trajectory generation, feedback control, and singularity avoidance.

This is coded in Python using a Conda Virtual Environment, so you'll likely have to set up your own Python interpreter in order to run it.
Otherwise, you can simply read through the logs and work through it.

This project already contains most of the parameters within the code itself, so if you read through the code (especially the final function starting on line 549), you should be able to find all parameters.

The code is contained within one Python class consisting of all 3 milestones, with each milestone clearly delimited through out the code. It's also commented fairly well for easy finding.

The only special feature that I implemented was singularity avoidance using the method described in the wiki, with the function test_joint_limits encompassing most of the logic.
It works by reading through an array of constraints (such as arms 3 and 4 being less than -0.2), checks if the configuration violates those constraints, then iterates through the Jacobian and sets the violating
segments to 0 to prevent further movement. This works pretty well and seemed to avoid most singularities.

One important thing that I may have structured differently was printing out the configuration, the time, and the trajectory step number for each timestep to easily demark what step does what.
This information was saved in the log file.

Thank you for reviewing!