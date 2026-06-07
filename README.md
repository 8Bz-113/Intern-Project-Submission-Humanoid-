# Humanoid AI Intern Project Submission

GitHub repository for **Abhinav Mayura's application to Humanoid AI**.

## Requirements

- Ubuntu
- NVIDIA Isaac Sim installed at:

```text
~/isaacsim
```

To Clone the Git Repository to your home directory:
```text
cd ~
git clone https://github.com/8Bz-113/Intern-Project-Submission-Humanoid-.git
```
This will create a folder called 'Intern-Project-Submission-Humanoid-'.

Then to run the script:
```text
cd ~/Intern-Project-Submission-Humanoid- 
~/isaacsim/python.sh scripts/end_effector_tracking.py
```

If you have IsaacSim installed somewhere else, change:
```text
~/isaacsim/python.sh scripts/end_effector_tracking.py
```
to the folder where IsaacSim is downloaded.

Running the script will create a folder called rl_orbit_logs within 'Intern-Project-Submission-Humanoid-', which contains csv. files containing data for each episode.

It will also create a file called episode_summary.csv which contains high level episode-by-episode data for the simulation.

The script will terminate automatically upon 3 successful runs. This can take between ~15-30minutes. 

A video explaining the mechanism of the system is provided below:
https://youtu.be/OX2dDgMpsKg 
